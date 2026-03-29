"""
aria.integrations.ollama — Ollama local runtime wrapper for ARIA auditing.

Ollama exposes a simple REST API on ``http://localhost:11434``.  This module
wraps ``httpx`` directly — no Ollama SDK is required on the client side.

Every call to :meth:`ARIAOllama.chat` or :meth:`ARIAOllama.generate` is
automatically audited via ARIA.

Usage::

    from aria.integrations.ollama import ARIAOllama

    client = ARIAOllama(
        model_id="llama3",
        auditor=auditor,        # InferenceAuditor instance
    )

    response = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "What is BSV?"}],
    )
    print(response["content"])

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-ollama-app")
    client = ARIAOllama(aria=aria)

Async support::

    from aria.integrations.ollama import ARIAAsyncOllama

    client = ARIAAsyncOllama(aria=aria)
    response = await client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "Hello!"}],
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
    """Extract a normalised output dict from an Ollama chat response."""
    try:
        content = response["message"]["content"]
        return {
            "content": content[:500],
            "model": response.get("model"),
            "done": response.get("done"),
        }
    except Exception:
        return {"raw": str(response)[:500]}


def _build_generate_output(response: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalised output dict from an Ollama generate response."""
    try:
        return {
            "content": response.get("response", "")[:500],
            "model": response.get("model"),
            "done": response.get("done"),
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIARecorder:
    """Shared recording logic for sync and async Ollama wrappers."""

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
            _log.warning("ARIAOllama: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class ARIAOllama:
    """Synchronous Ollama client with automatic ARIA auditing.

    Args:
        base_url:  Base URL of the Ollama server (default: ``http://localhost:11434``).
        model_id:  Default model identifier used in ARIA audit records when
                   the caller does not pass an explicit *model* argument.
        auditor:   ``InferenceAuditor`` instance (mutually exclusive with *aria*).
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        timeout:   HTTP request timeout in seconds (default: 120.0 — Ollama
                   can be slow on first load).

    Example::

        client = ARIAOllama(
            base_url="http://localhost:11434",
            model_id="llama3",
            auditor=auditor,
        )

        # Multi-turn chat
        resp = client.chat(
            model="llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(resp["content"])

        # Text generation
        resp = client.generate(model="llama3", prompt="Once upon a time")
        print(resp["content"])

        # List available models
        models = client.list_models()
        print(models)   # ["llama3:latest", "mistral:7b", ...]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "ollama-model",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._http = httpx.Client(timeout=timeout)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request to Ollama.

        Args:
            model:    Model name (e.g. ``"llama3"``).
            messages: List of message dicts with ``role`` and ``content`` keys.
            **kwargs: Additional fields forwarded in the request body (e.g.
                      ``stream=False``, ``options``).

        Returns:
            Normalised response dict::

                {
                    "content": "<assistant reply (up to 500 chars)>",
                    "model":   "<model name>",
                    "done":    True,
                }

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in (messages or [])
            ],
            "stream": False,
            **kwargs,
        }
        input_data = {"messages": payload["messages"]}

        t0 = time.time()
        response = self._http.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_chat_output(data)

        self._recorder.record(
            model_id=model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "ollama",
                "base_url": self._base_url,
                "model": model,
            },
        )
        return output_data

    def generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a text generation request to Ollama.

        Args:
            model:    Model name (e.g. ``"llama3"``).
            prompt:   Prompt string (truncated to 500 chars for audit records).
            **kwargs: Additional fields forwarded in the request body (e.g.
                      ``stream=False``, ``options``).

        Returns:
            Normalised response dict::

                {
                    "content": "<generated text (up to 500 chars)>",
                    "model":   "<model name>",
                    "done":    True,
                }

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        input_data = {"prompt": prompt[:500]}

        t0 = time.time()
        response = self._http.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_generate_output(data)

        self._recorder.record(
            model_id=model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "ollama",
                "base_url": self._base_url,
                "model": model,
            },
        )
        return output_data

    def list_models(self) -> list[str]:
        """Return the names of all locally available Ollama models.

        Calls ``GET /api/tags`` and extracts the ``name`` field from each
        entry in ``models``.

        Returns:
            List of model name strings, e.g.
            ``["llama3:latest", "mistral:7b-instruct"]``.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        response = self._http.get(f"{self._base_url}/api/tags")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [m["name"] for m in data.get("models", [])]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def __enter__(self) -> "ARIAOllama":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class ARIAAsyncOllama:
    """Asynchronous Ollama client with automatic ARIA auditing.

    Drop-in async companion to :class:`ARIAOllama`.  Uses ``httpx.AsyncClient``.

    Args:
        base_url:  Base URL of the Ollama server (default: ``http://localhost:11434``).
        model_id:  Default model identifier used in ARIA audit records.
        auditor:   ``InferenceAuditor`` instance (mutually exclusive with *aria*).
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        timeout:   HTTP request timeout in seconds (default: 120.0).

    Example::

        async with ARIAAsyncOllama(aria=aria) as client:
            resp = await client.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp["content"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_id: str = "ollama-model",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._http = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`ARIAOllama.chat`.

        Args:
            model:    Model name.
            messages: List of message dicts with ``role`` and ``content`` keys.
            **kwargs: Additional fields forwarded in the request body.

        Returns:
            Normalised response dict with ``content``, ``model``, and ``done``.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                for m in (messages or [])
            ],
            "stream": False,
            **kwargs,
        }
        input_data = {"messages": payload["messages"]}

        t0 = time.time()
        response = await self._http.post(
            f"{self._base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_chat_output(data)

        self._recorder.record(
            model_id=model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "ollama",
                "base_url": self._base_url,
                "model": model,
            },
        )
        return output_data

    async def generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`ARIAOllama.generate`.

        Args:
            model:    Model name.
            prompt:   Prompt string (truncated to 500 chars for audit records).
            **kwargs: Additional fields forwarded in the request body.

        Returns:
            Normalised response dict with ``content``, ``model``, and ``done``.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        input_data = {"prompt": prompt[:500]}

        t0 = time.time()
        response = await self._http.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        latency_ms = (time.time() - t0) * 1000

        data: dict[str, Any] = response.json()
        output_data = _build_generate_output(data)

        self._recorder.record(
            model_id=model,
            input_data=input_data,
            output_data=output_data,
            confidence=None,
            latency_ms=latency_ms,
            metadata={
                "provider": "ollama",
                "base_url": self._base_url,
                "model": model,
            },
        )
        return output_data

    async def list_models(self) -> list[str]:
        """Async version of :meth:`ARIAOllama.list_models`.

        Returns:
            List of model name strings available locally on the Ollama server.

        Raises:
            httpx.HTTPError: on network or HTTP-level failures.
        """
        response = await self._http.get(f"{self._base_url}/api/tags")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> "ARIAAsyncOllama":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()
