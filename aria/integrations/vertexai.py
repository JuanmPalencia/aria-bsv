"""
aria.integrations.vertexai — Google Vertex AI prediction endpoint wrapper for ARIA auditing.

Wraps the Vertex AI online prediction REST API using ``httpx`` directly — no
``google-cloud-aiplatform`` SDK is required on the client side.  Every call to
:meth:`ARIAVertexAI.predict` (or its async counterpart
:meth:`ARIAAsyncVertexAI.predict`) is automatically audited via ARIA.

Usage::

    from aria.integrations.vertexai import ARIAVertexAI

    client = ARIAVertexAI(
        project="my-gcp-project",
        location="us-central1",
        endpoint_id="1234567890",
        auditor=auditor,            # InferenceAuditor instance
        model_id="my-vertex-model", # overrides the model label in ARIA records
        credentials_json="Bearer ya29.xxx",  # raw Bearer token
    )

    response = client.predict([{"feature_1": 0.5, "feature_2": 1.2}])
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-vertex-app")
    client = ARIAVertexAI(
        project="my-gcp-project",
        location="us-central1",
        endpoint_id="1234567890",
        aria=aria,
    )

Async support::

    from aria.integrations.vertexai import ARIAAsyncVertexAI

    client = ARIAAsyncVertexAI(
        project="my-gcp-project",
        location="us-central1",
        endpoint_id="1234567890",
        aria=aria,
    )
    response = await client.predict([{"x": 1}])

Notes:
    - No Google Cloud SDK is required; only ``httpx`` (already a core ARIA
      dependency) is used.
    - If *credentials_json* is provided it is sent as the value of the
      ``Authorization`` header verbatim (e.g. ``"Bearer <token>"``).  When it
      is ``None`` no ``Authorization`` header is added, which is useful for
      local development or IAP-bypassed endpoints.
    - Only the first 3 instances (truncated to 400 chars) are stored in the
      ARIA record — keeping large feature vectors off the audit trail while
      preserving traceability.
    - Raw inference data never goes on-chain; only SHA-256 hashes are anchored
      per BRC-121 requirements.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIARecorder:
    """Shared recording logic for sync and async Vertex AI wrappers."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs: Any) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    kwargs["model_id"],
                    kwargs["input_data"],
                    kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=int(kwargs.get("latency_ms") or 0),
                    metadata=kwargs.get("metadata") or {},
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=kwargs["model_id"],
                    input_data=kwargs["input_data"],
                    output_data=kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=kwargs.get("latency_ms"),
                    metadata=kwargs.get("metadata") or {},
                )
        except Exception as exc:
            _log.warning("ARIAVertexAI: record error: %s", exc)


def _build_url(project: str, location: str, endpoint_id: str) -> str:
    """Build the Vertex AI online prediction URL."""
    return (
        f"https://{location}-aiplatform.googleapis.com/v1"
        f"/projects/{project}/locations/{location}"
        f"/endpoints/{endpoint_id}:predict"
    )


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class ARIAVertexAI:
    """Synchronous Vertex AI prediction client with automatic ARIA auditing.

    Sends prediction requests to a Vertex AI online endpoint via the REST API
    (no Google Cloud SDK required).

    Args:
        project:          GCP project ID.
        location:         GCP region where the endpoint is deployed
                          (e.g. ``"us-central1"``).
        endpoint_id:      Numeric or named Vertex AI endpoint ID.
        auditor:          ``InferenceAuditor`` instance (mutually exclusive
                          with *aria*).
        aria:             ``ARIAQuick`` instance (alternative to *auditor*).
        model_id:         Override for the model_id label in ARIA records.  If
                          ``None``, *endpoint_id* is used.
        credentials_json: Value to use for the ``Authorization`` header
                          (e.g. ``"Bearer ya29.xxx"``).  If ``None``, no auth
                          header is sent.

    Raises:
        httpx.HTTPStatusError: on HTTP 4xx / 5xx responses.

    Example::

        client = ARIAVertexAI(
            project="acme-prod",
            location="europe-west4",
            endpoint_id="9876543210",
            model_id="fraud-v2",
            credentials_json="Bearer " + access_token,
            auditor=auditor,
        )

        result = client.predict([{"amount": 42.5, "merchant_id": "m-001"}])
        print(result["predictions"])
    """

    def __init__(
        self,
        project: str,
        location: str,
        endpoint_id: str,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        credentials_json: str | None = None,
    ) -> None:
        self._project = project
        self._location = location
        self._endpoint_id = endpoint_id
        self._credentials_json = credentials_json
        self._url = _build_url(project, location, endpoint_id)
        self._recorder = _ARIARecorder(
            auditor=auditor,
            aria=aria,
            model_id=model_id or endpoint_id,
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if credentials_json:
            headers["Authorization"] = credentials_json
        self._http = httpx.Client(headers=headers)

    def predict(self, instances: list[dict], **kwargs: Any) -> dict:
        """Send a prediction request and record the inference in ARIA.

        Args:
            instances: List of instance dicts to predict on.
            **kwargs:  Additional fields merged into the request body (e.g.
                       ``parameters``).

        Returns:
            Parsed JSON response from Vertex AI (contains ``"predictions"``
            key).

        Raises:
            httpx.HTTPStatusError: on HTTP 4xx / 5xx responses.
        """
        body: dict[str, Any] = {"instances": instances, **kwargs}

        input_data: dict[str, Any] = {
            "instances": repr(instances[:3])[:400],
        }

        t0 = time.time()
        response = self._http.post(self._url, json=body)
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        parsed: dict = response.json()

        predictions = parsed.get("predictions", [])
        output_data: dict[str, Any] = {
            "predictions": repr(predictions[:3])[:400],
        }

        self._recorder.record(
            model_id=self._recorder.model_id,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vertexai",
                "project": self._project,
                "location": self._location,
                "endpoint": self._endpoint_id,
            },
        )
        return parsed

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "ARIAVertexAI":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class ARIAAsyncVertexAI:
    """Asynchronous Vertex AI prediction client with automatic ARIA auditing.

    Drop-in async companion to :class:`ARIAVertexAI`.  Uses
    ``httpx.AsyncClient``.

    Args:
        project:          GCP project ID.
        location:         GCP region where the endpoint is deployed.
        endpoint_id:      Numeric or named Vertex AI endpoint ID.
        auditor:          ``InferenceAuditor`` instance (mutually exclusive
                          with *aria*).
        aria:             ``ARIAQuick`` instance (alternative to *auditor*).
        model_id:         Override for the model_id label in ARIA records.
        credentials_json: Value to use for the ``Authorization`` header.  If
                          ``None``, no auth header is sent.

    Example::

        async with ARIAAsyncVertexAI(
            project="acme-prod",
            location="us-central1",
            endpoint_id="1234567890",
            aria=aria,
        ) as client:
            result = await client.predict([{"x": 1.0}])
            print(result["predictions"])
    """

    def __init__(
        self,
        project: str,
        location: str,
        endpoint_id: str,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        credentials_json: str | None = None,
    ) -> None:
        self._project = project
        self._location = location
        self._endpoint_id = endpoint_id
        self._credentials_json = credentials_json
        self._url = _build_url(project, location, endpoint_id)
        self._recorder = _ARIARecorder(
            auditor=auditor,
            aria=aria,
            model_id=model_id or endpoint_id,
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if credentials_json:
            headers["Authorization"] = credentials_json
        self._http = httpx.AsyncClient(headers=headers)

    async def predict(self, instances: list[dict], **kwargs: Any) -> dict:
        """Async version of :meth:`ARIAVertexAI.predict`.

        Args:
            instances: List of instance dicts to predict on.
            **kwargs:  Additional fields merged into the request body.

        Returns:
            Parsed JSON response from Vertex AI.

        Raises:
            httpx.HTTPStatusError: on HTTP 4xx / 5xx responses.
        """
        body: dict[str, Any] = {"instances": instances, **kwargs}

        input_data: dict[str, Any] = {
            "instances": repr(instances[:3])[:400],
        }

        t0 = time.time()
        response = await self._http.post(self._url, json=body)
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        parsed: dict = response.json()

        predictions = parsed.get("predictions", [])
        output_data: dict[str, Any] = {
            "predictions": repr(predictions[:3])[:400],
        }

        self._recorder.record(
            model_id=self._recorder.model_id,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vertexai",
                "project": self._project,
                "location": self._location,
                "endpoint": self._endpoint_id,
            },
        )
        return parsed

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "ARIAAsyncVertexAI":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
