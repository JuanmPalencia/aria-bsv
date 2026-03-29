"""
aria.integrations.vllm — vLLM client wrapper for ARIA auditing.

vLLM exposes an OpenAI-compatible REST API.  This module wraps ``httpx``
directly — no vLLM SDK is required on the client side.

Every call to :meth:`ARIAvLLM.chat` or :meth:`ARIAvLLM.complete` is
automatically audited via ARIA.

Usage::

    from aria.integrations.vllm import ARIAvLLM

    client = ARIAvLLM(
        base_url="http://localhost:8000",
        model_id="Llama-3-8B",
        auditor=auditor,        # InferenceAuditor instance
    )

    response = client.chat(
        messages=[{"role": "user", "content": "Explain BSV in one line."}]
    )
    # ↑ Automatically recorded in ARIA; response is the parsed JSON dict.

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-vllm-app")
    client = ARIAvLLM(aria=aria)

Async support::

    from aria.integrations.vllm import ARIAAsyncvLLM

    client = ARIAAsyncvLLM(aria=aria)
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
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
# Helpers
# ---------------------------------------------------------------------------

def _build_chat_output(response: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalised output dict from a chat completions response."""
    try:
        choice = response["choices"][0]
        return {
            "content": choice["message"]["content"],
            "model": response.get("model"),
            "usage": response.get("usage", {}),
        }
    except Exception:
        return {"raw": str(response)[:500]}


def _build_complete_output(response: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalised output dict from a completions response."""
    try:
        choice = response["choices"][0]
        return {
            "content": choice.get("text", ""),
            "model": response.get("model"),
            "usage": response.get("usage", {}),
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIARecorder:
    """Shared recording logic for sync and async vLLM wrappers."""

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
            _log.warning("ARIAvLLM: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class ARIAvLLM:
    """Synchronous vLLM client with automatic ARIA auditing.

    vLLM exposes an OpenAI-compatible HTTP API, so this client POSTs to
    ``/v1/chat/completions`` and ``/v1/completions`` using ``httpx``.

    Args:
        base_url:  Base URL of the vLLM server (default: ``http://localhost:8000``).
        model_id:  Model identifier used in ARIA audit records.
        auditor:   ``InferenceAuditor`` instance (mutually exclusive with *aria*).
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        timeout:   HTTP request timeout in seconds (default: 30.0).

    Example::

        client = ARIAvLLM(
            base_url="http://gpu-host:8000",
            model_id="meta-llama/Llama-3-8B-Instruct",
            auditor=auditor,
        )

        # Chat
        resp = client.chat(messages=[{"role": "user", "content": "Hello"}])
        print(resp["content"])

        # Text completion
        resp = client.complete(prompt="The capital of France is")
        print(resp["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_id: str = "vllm-model",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._http = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completions request to vLLM.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model:    Model name to send in the request body.  Defaults to
                      *model_id* passed to the constructor.
            **kwargs: Additional fields forwarded in the request body (e.g.
                      ``temperature``, ``max_tokens``).

        Returns:
            Parsed response dict::

                {
                    "content": "<assistant reply>",
                    "model":   "<model name>",
                    "usage":   {"prompt_tokens": ..., "completion_tokens": ...},
                }

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        effective_model = model or self._model_id
        payload = {
            "model": effective_model,
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in (messages or [])
            ],
            **kwargs,
        }
        input_data = {"messages": payload["messages"]}

        t0 = time.time()
        response = self._http.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_chat_output(data)

        self._recorder.record(
            model_id=effective_model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vllm",
                "base_url": self._base_url,
                "model": effective_model,
            },
        )
        return output_data

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a text completions request to vLLM.

        Args:
            prompt:   Prompt string (truncated to 500 chars for audit records).
            model:    Model name to send in the request body.
            **kwargs: Additional fields forwarded in the request body.

        Returns:
            Parsed response dict::

                {
                    "content": "<completion text>",
                    "model":   "<model name>",
                    "usage":   {"prompt_tokens": ..., "completion_tokens": ...},
                }

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        effective_model = model or self._model_id
        payload = {
            "model": effective_model,
            "prompt": prompt,
            **kwargs,
        }
        input_data = {"prompt": prompt[:500]}

        t0 = time.time()
        response = self._http.post(
            f"{self._base_url}/v1/completions",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_complete_output(data)

        self._recorder.record(
            model_id=effective_model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vllm",
                "base_url": self._base_url,
                "model": effective_model,
            },
        )
        return output_data

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "ARIAvLLM":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class ARIAAsyncvLLM:
    """Asynchronous vLLM client with automatic ARIA auditing.

    Drop-in async companion to :class:`ARIAvLLM`.  Uses ``httpx.AsyncClient``.

    Args:
        base_url:  Base URL of the vLLM server (default: ``http://localhost:8000``).
        model_id:  Model identifier used in ARIA audit records.
        auditor:   ``InferenceAuditor`` instance (mutually exclusive with *aria*).
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        timeout:   HTTP request timeout in seconds (default: 30.0).

    Example::

        async with ARIAAsyncvLLM(base_url="http://gpu-host:8000", aria=aria) as client:
            resp = await client.chat(
                messages=[{"role": "user", "content": "Hello"}]
            )
            print(resp["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_id: str = "vllm-model",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._http = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`ARIAvLLM.chat`.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model:    Model name to send in the request body.
            **kwargs: Additional fields forwarded in the request body.

        Returns:
            Parsed response dict with ``content``, ``model``, and ``usage``.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        effective_model = model or self._model_id
        payload = {
            "model": effective_model,
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in (messages or [])
            ],
            **kwargs,
        }
        input_data = {"messages": payload["messages"]}

        t0 = time.time()
        response = await self._http.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_chat_output(data)

        self._recorder.record(
            model_id=effective_model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vllm",
                "base_url": self._base_url,
                "model": effective_model,
            },
        )
        return output_data

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`ARIAvLLM.complete`.

        Args:
            prompt:   Prompt string (truncated to 500 chars for audit records).
            model:    Model name to send in the request body.
            **kwargs: Additional fields forwarded in the request body.

        Returns:
            Parsed response dict with ``content``, ``model``, and ``usage``.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        effective_model = model or self._model_id
        payload = {
            "model": effective_model,
            "prompt": prompt,
            **kwargs,
        }
        input_data = {"prompt": prompt[:500]}

        t0 = time.time()
        response = await self._http.post(
            f"{self._base_url}/v1/completions",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_complete_output(data)

        self._recorder.record(
            model_id=effective_model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "vllm",
                "base_url": self._base_url,
                "model": effective_model,
            },
        )
        return output_data

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "ARIAAsyncvLLM":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
