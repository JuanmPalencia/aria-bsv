"""
aria.integrations.anthropic_sdk — Drop-in Anthropic client wrapper for ARIA.

Wraps ``anthropic.Anthropic`` so every call to ``messages.create()`` is
automatically audited. Named ``anthropic_sdk`` to avoid clashing with the
``anthropic`` package namespace.

Usage::

    from aria.integrations.anthropic_sdk import ARIAAnthropic

    client = ARIAAnthropic(aria=aria)   # or auditor=auditor

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # ↑ Automatically audited in ARIA

Async::

    from aria.integrations.anthropic_sdk import ARIAAsyncAnthropic
    client = ARIAAsyncAnthropic(aria=aria)
    response = await client.messages.create(...)
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

def _messages_to_input(messages: list[dict], system: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "messages": [
            {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
            for m in (messages or [])
        ]
    }
    if system:
        result["system"] = system[:500]
    return result


def _response_to_output(response: Any) -> dict[str, Any]:
    try:
        content = ""
        if response.content:
            block = response.content[0]
            content = getattr(block, "text", str(block))
        return {
            "content": content[:1000],
            "stop_reason": getattr(response, "stop_reason", ""),
            "model": getattr(response, "model", ""),
            "usage": {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            } if hasattr(response, "usage") else {},
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class _ARIAMessages:
    def __init__(self, original: Any, recorder: Any) -> None:
        self._orig = original
        self._recorder = recorder

    def create(self, **kwargs) -> Any:
        model_id = self._recorder.model_id or kwargs.get("model", "claude")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        t0 = time.time()
        response = self._orig.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages, system),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata={
                "provider": "anthropic",
                "model": kwargs.get("model", ""),
                "max_tokens": kwargs.get("max_tokens"),
            },
        )
        return response


class _ARIARecorder:
    def __init__(self, auditor=None, aria=None, model_id=None):
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    kwargs["model_id"], kwargs["input_data"], kwargs["output_data"],
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
            _log.warning("ARIAAnthropic: record error: %s", exc)


class ARIAAnthropic:
    """Drop-in replacement for ``anthropic.Anthropic`` with ARIA auditing.

    Args:
        auditor:   ``InferenceAuditor`` instance.
        aria:      ``ARIAQuick`` instance.
        model_id:  Override for the model_id label in ARIA records.
        **kwargs:  Forwarded to ``anthropic.Anthropic()``.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(**kwargs)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. pip install aria-bsv[anthropic]"
            )
        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.messages = _ARIAMessages(self._client.messages, self._recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class _ARIAAsyncMessages:
    def __init__(self, original: Any, recorder: Any) -> None:
        self._orig = original
        self._recorder = recorder

    async def create(self, **kwargs) -> Any:
        model_id = self._recorder.model_id or kwargs.get("model", "claude")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        t0 = time.time()
        response = await self._orig.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages, system),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata={"provider": "anthropic", "model": kwargs.get("model", "")},
        )
        return response


class ARIAAsyncAnthropic:
    """Async drop-in for ``anthropic.AsyncAnthropic``."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(**kwargs)
        except ImportError:
            raise ImportError("anthropic package not installed.")

        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.messages = _ARIAAsyncMessages(self._client.messages, self._recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
