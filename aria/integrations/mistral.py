"""
aria.integrations.mistral — Drop-in Mistral AI client wrapper for ARIA auditing.

Wraps the official ``mistralai`` client so that every call to
``chat.complete()`` (sync) and ``chat.complete_async()`` (async) is
automatically audited via ARIA.

Zero config changes required — just replace ``Mistral()`` with ``ARIAMistral()``.

Usage::

    from aria.integrations.mistral import ARIAMistral

    # Drop-in replacement — same API as mistralai.Mistral
    client = ARIAMistral(
        api_key="your-mistral-api-key",
        auditor=auditor,          # InferenceAuditor instance
        model_id="mistral-large-latest",  # overrides the model field in records
    )

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-app")
    client = ARIAMistral(api_key="your-key", aria=aria)

Async support::

    from aria.integrations.mistral import ARIAAsyncMistral

    client = ARIAAsyncMistral(api_key="your-key", aria=aria)
    response = await client.chat.complete_async(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Note:
    Mistral AI responses do not include logprobs, so ``confidence`` is always
    ``None`` in ARIA records produced by this integration.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_input(messages: list[dict]) -> dict[str, Any]:
    """Convert a messages list to a serialisable input dict.

    Truncates each content field to 500 characters to keep record sizes
    manageable.
    """
    return {
        "messages": [
            {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
            for m in (messages or [])
        ]
    }


def _response_to_output(response: Any, model: str = "") -> dict[str, Any]:
    """Convert a Mistral chat response to a serialisable output dict."""
    try:
        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "model": model,
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            } if hasattr(response, "usage") and response.usage else {},
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIAMistralRecorder:
    """Shared recording logic for sync and async Mistral wrappers."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs) -> None:
        """Forward an inference record to the configured auditor or ARIAQuick instance.

        Errors are swallowed and logged so that auditing failures never
        interrupt the application's normal control flow.
        """
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
            _log.warning("ARIAMistral: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class _ARIAMistralChat:
    """Wraps the ``chat`` namespace of a ``mistralai.Mistral`` client."""

    def __init__(self, original: Any, recorder: _ARIAMistralRecorder) -> None:
        self._orig = original
        self._recorder = recorder

    def complete(self, **kwargs) -> Any:
        """Call ``chat.complete()`` and record the inference in ARIA."""
        model_kwarg = kwargs.get("model", "")
        model_id = self._recorder.model_id or model_kwarg or "mistral"
        messages = kwargs.get("messages", [])
        t0 = time.time()
        response = self._orig.complete(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response, model=model_kwarg),
            confidence=None,  # Mistral does not expose logprobs
            latency_ms=latency_ms,
            metadata={"provider": "mistral", "model": model_kwarg},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class ARIAMistral:
    """Drop-in replacement for ``mistralai.Mistral`` with automatic ARIA auditing.

    Args:
        auditor:   ``InferenceAuditor`` instance.
        aria:      ``ARIAQuick`` instance (alternative to ``auditor``).
        model_id:  Override for the ``model_id`` label stored in ARIA records.
                   If ``None``, the ``model`` field from the API request is used.
        **kwargs:  All keyword arguments are forwarded to ``mistralai.Mistral()``.

    Raises:
        ImportError: if the ``mistralai`` package is not installed.

    Note:
        ``confidence`` is always ``None`` because Mistral AI does not expose
        per-token log-probabilities in its chat completion responses.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import mistralai
            self._client = mistralai.Mistral(**kwargs)
        except ImportError:
            raise ImportError(
                "mistralai package not installed. pip install aria-bsv[mistral]"
            )

        self._recorder = _ARIAMistralRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.chat = _ARIAMistralChat(self._client.chat, self._recorder)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying Mistral client."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class _ARIAAsyncMistralChat:
    """Wraps the ``chat`` namespace for the async Mistral variant."""

    def __init__(self, original: Any, recorder: _ARIAMistralRecorder) -> None:
        self._orig = original
        self._recorder = recorder

    async def complete_async(self, **kwargs) -> Any:
        """Call ``chat.complete_async()`` and record the inference in ARIA."""
        model_kwarg = kwargs.get("model", "")
        model_id = self._recorder.model_id or model_kwarg or "mistral"
        messages = kwargs.get("messages", [])
        t0 = time.time()
        response = await self._orig.complete_async(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response, model=model_kwarg),
            confidence=None,  # Mistral does not expose logprobs
            latency_ms=latency_ms,
            metadata={"provider": "mistral", "model": model_kwarg},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class ARIAAsyncMistral:
    """Async drop-in replacement for ``mistralai.Mistral`` with ARIA auditing.

    Use this variant when running inside an ``asyncio`` event loop.  The
    interface mirrors ``ARIAMistral`` but ``chat.complete_async()`` is an
    ``async`` coroutine.

    Args:
        auditor:   ``InferenceAuditor`` instance.
        aria:      ``ARIAQuick`` instance.
        model_id:  Override for the ``model_id`` label in ARIA records.
        **kwargs:  Forwarded to ``mistralai.Mistral()``.

    Raises:
        ImportError: if the ``mistralai`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import mistralai
            self._client = mistralai.Mistral(**kwargs)
        except ImportError:
            raise ImportError(
                "mistralai package not installed. pip install aria-bsv[mistral]"
            )

        self._recorder = _ARIAMistralRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.chat = _ARIAAsyncMistralChat(self._client.chat, self._recorder)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying Mistral client."""
        return getattr(self._client, name)
