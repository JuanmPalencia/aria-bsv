"""
aria.integrations.sagemaker — AWS SageMaker real-time endpoint wrapper for ARIA auditing.

Wraps ``boto3.client("sagemaker-runtime")`` so that every call to
:meth:`ARIASageMaker.invoke` is automatically audited via ARIA without
changing your existing endpoint code.

Zero config changes required — just replace your raw boto3 client calls with
:class:`ARIASageMaker`.

Usage::

    from aria.integrations.sagemaker import ARIASageMaker

    # Wrap a SageMaker real-time endpoint
    client = ARIASageMaker(
        endpoint_name="my-endpoint",
        auditor=auditor,          # InferenceAuditor instance
        model_id="my-model-v1",   # overrides the model label in ARIA records
        region_name="us-west-2",
    )

    response = client.invoke({"feature_1": 0.5, "feature_2": 1.2})
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-sagemaker-app")
    client = ARIASageMaker(endpoint_name="my-endpoint", aria=aria)

Notes:
    - ``boto3`` must be installed.  Install with ``pip install aria-bsv[sagemaker]``.
    - The full request payload is never sent to the ARIA record — only the
      first 400 characters of its ``repr`` are stored, keeping sensitive data
      off the audit trail while preserving traceability.
    - Raw inference data never goes on-chain; only SHA-256 hashes are anchored
      per BRC-121 requirements.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIARecorder:
    """Shared recording logic for ARIASageMaker."""

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
            _log.warning("ARIASageMaker: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class ARIASageMaker:
    """ARIA-audited wrapper for an AWS SageMaker real-time inference endpoint.

    Wraps ``boto3.client("sagemaker-runtime")`` and records every invocation
    in ARIA.  The underlying boto3 client is created lazily on first use so
    that construction is cheap even when the endpoint is not immediately
    needed.

    Args:
        endpoint_name: Name of the SageMaker endpoint to invoke.
        auditor:       ``InferenceAuditor`` instance (mutually exclusive with
                       *aria*).
        aria:          ``ARIAQuick`` instance (alternative to *auditor*).
        model_id:      Override for the model_id label in ARIA records.  If
                       ``None``, ``endpoint_name`` is used as the label.
        region_name:   AWS region where the endpoint lives
                       (default: ``"us-east-1"``).
        content_type:  MIME type sent as the ``ContentType`` header to
                       SageMaker (default: ``"application/json"``).

    Raises:
        ImportError: if ``boto3`` is not installed.  Install it with
            ``pip install aria-bsv[sagemaker]``.

    Example::

        client = ARIASageMaker(
            endpoint_name="fraud-detector-v3",
            model_id="fraud-detector-v3",
            region_name="eu-west-1",
            auditor=auditor,
        )

        result = client.invoke({"transaction_id": "tx-123", "amount": 99.99})
        print(result)
    """

    def __init__(
        self,
        endpoint_name: str,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        region_name: str = "us-east-1",
        content_type: str = "application/json",
    ) -> None:
        try:
            import boto3  # noqa: F401 — ensure it is importable
        except ImportError:
            raise ImportError(
                "boto3 not installed. pip install aria-bsv[sagemaker]"
            )

        self._endpoint_name = endpoint_name
        self._region_name = region_name
        self._content_type = content_type
        self._recorder = _ARIARecorder(
            auditor=auditor,
            aria=aria,
            model_id=model_id or endpoint_name,
        )

        import boto3
        self._client = boto3.client(
            "sagemaker-runtime",
            region_name=region_name,
        )

    def invoke(self, payload: dict | str, **kwargs: Any) -> dict:
        """Invoke the SageMaker endpoint and record the inference in ARIA.

        Args:
            payload:  Request body.  A ``dict`` is serialised to JSON; a
                      ``str`` is sent as-is.
            **kwargs: Additional keyword arguments forwarded to
                      ``boto3`` ``invoke_endpoint()``.

        Returns:
            Parsed JSON response from the endpoint as a ``dict``.

        Raises:
            botocore.exceptions.ClientError: on AWS-level errors.
            json.JSONDecodeError: if the endpoint returns non-JSON body.
        """
        body: str = json.dumps(payload) if isinstance(payload, dict) else payload

        input_data: dict[str, Any] = {
            "payload": repr(payload)[:400],
            "endpoint": self._endpoint_name,
        }

        t0 = time.time()
        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType=self._content_type,
            Body=body,
            **kwargs,
        )
        latency_ms = (time.time() - t0) * 1000

        parsed_response: dict = json.loads(response["Body"].read())

        output_data: dict[str, Any] = {
            "response": repr(parsed_response)[:400],
            "endpoint": self._endpoint_name,
        }

        self._recorder.record(
            model_id=self._recorder.model_id,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "sagemaker",
                "endpoint": self._endpoint_name,
                "region": self._region_name,
            },
        )
        return parsed_response
