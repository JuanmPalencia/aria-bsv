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
# Streaming helpers
# ---------------------------------------------------------------------------

class _ARIAAnthropicStreamIterator:
    """Wraps a sync Anthropic stream and records to ARIA when exhausted.

    Anthropic streaming chunks have a ``delta.text`` attribute on
    ``content_block_delta`` events.  Other event types are yielded transparently
    without text extraction.
    """

    def __init__(
        self,
        stream: Any,
        model_id: str,
        input_data: Any,
        recorder: Any,
        t0: float,
        metadata: dict,
    ) -> None:
        self._stream = stream
        self._model_id = model_id
        self._input_data = input_data
        self._recorder = recorder
        self._t0 = t0
        self._metadata = metadata
        self._chunks: list[str] = []

    def __iter__(self) -> Any:
        for event in self._stream:
            try:
                # Anthropic SDK v0.20+: RawContentBlockDeltaEvent
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    text = event.delta.text or ""
                    if text:
                        self._chunks.append(text)
            except Exception:
                pass
            yield event
        self._record()

    def _record(self) -> None:
        text = "".join(self._chunks)
        latency_ms = (time.time() - self._t0) * 1000
        self._recorder.record(
            model_id=self._model_id,
            input_data=self._input_data,
            output_data={"text": text, "chunk_count": len(self._chunks), "streamed": True},
            latency_ms=latency_ms,
            metadata={**self._metadata, "streamed": True},
        )

    def __enter__(self) -> "_ARIAAnthropicStreamIterator":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class _ARIAAnthropicAsyncStreamIterator:
    """Async variant of _ARIAAnthropicStreamIterator."""

    def __init__(
        self,
        stream: Any,
        model_id: str,
        input_data: Any,
        recorder: Any,
        t0: float,
        metadata: dict,
    ) -> None:
        self._stream = stream
        self._model_id = model_id
        self._input_data = input_data
        self._recorder = recorder
        self._t0 = t0
        self._metadata = metadata
        self._chunks: list[str] = []

    def __aiter__(self) -> "_ARIAAnthropicAsyncStreamIterator":
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._stream.__anext__()
        except StopAsyncIteration:
            self._record()
            raise
        try:
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                text = event.delta.text or ""
                if text:
                    self._chunks.append(text)
        except Exception:
            pass
        return event

    def _record(self) -> None:
        text = "".join(self._chunks)
        latency_ms = (time.time() - self._t0) * 1000
        self._recorder.record(
            model_id=self._model_id,
            input_data=self._input_data,
            output_data={"text": text, "chunk_count": len(self._chunks), "streamed": True},
            latency_ms=latency_ms,
            metadata={**self._metadata, "streamed": True},
        )

    async def __aenter__(self) -> "_ARIAAnthropicAsyncStreamIterator":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


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
        meta = {
            "provider": "anthropic",
            "model": kwargs.get("model", ""),
            "max_tokens": kwargs.get("max_tokens"),
        }
        t0 = time.time()
        response = self._orig.create(**kwargs)

        # Streaming: wrap iterator — record fires when stream is exhausted
        if kwargs.get("stream"):
            return _ARIAAnthropicStreamIterator(
                iter(response), model_id, _messages_to_input(messages, system),
                self._recorder, t0, meta,
            )

        latency_ms = (time.time() - t0) * 1000
        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages, system),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata=meta,
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
        meta = {"provider": "anthropic", "model": kwargs.get("model", "")}
        t0 = time.time()
        response = await self._orig.create(**kwargs)

        # Streaming: wrap async iterator
        if kwargs.get("stream"):
            return _ARIAAnthropicAsyncStreamIterator(
                response.__aiter__(), model_id, _messages_to_input(messages, system),
                self._recorder, t0, meta,
            )

        latency_ms = (time.time() - t0) * 1000
        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages, system),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata=meta,
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
