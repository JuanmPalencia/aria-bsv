"""
aria.streaming — Streaming-aware audit sessions for ARIA.

LLMs that generate tokens incrementally (OpenAI ``stream=True``, Anthropic
``stream=True``, vLLM, Ollama, etc.) produce output in chunks, not in a
single response.  This module provides context managers that accumulate those
chunks and produce a single :class:`AuditRecord` after the stream ends — with
correct end-to-end latency measurement and the full accumulated text.

Usage (sync)::

    from aria.streaming import ARIAStreamingAuditor

    sa = ARIAStreamingAuditor(auditor=auditor)

    # Option A — context manager
    with sa.start_stream("gpt-4o", input_messages) as session:
        for chunk in openai_stream:
            text = chunk.choices[0].delta.content or ""
            session.add_chunk(text)
    print(session.record_id)   # AuditRecord created on __exit__

    # Option B — manual
    session = sa.start_stream("gpt-4o", input_messages)
    session.__enter__()
    for chunk in openai_stream:
        session.add_chunk(chunk.choices[0].delta.content or "")
    record_id = session.finish()

Usage (async)::

    sa = ARIAStreamingAuditor(aria=aria)

    async with sa.start_async_stream("claude-3-5-sonnet", input_data) as session:
        async for chunk in anthropic_stream:
            session.add_chunk(chunk.delta.text or "")
    print(session.record_id)

The module is backend-agnostic — it works with any object that exposes a
``.record()`` method compatible with :class:`aria.auditor.InferenceAuditor`
or :class:`aria.quick.ARIAQuick`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .auditor import InferenceAuditor
    from .quick import ARIAQuick


# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Configuration for a single streaming session.

    Attributes:
        model_id:    Model identifier (must be registered in the auditor's
                     ``model_hashes`` if using ``InferenceAuditor``).
        input_data:  The input that triggered this stream (messages list,
                     prompt string, or any JSON-serialisable object).
        confidence:  Optional override for the confidence score to store in
                     the AuditRecord.  If ``None``, confidence is omitted.
        metadata:    Additional key-value pairs merged into the AuditRecord
                     metadata.
    """

    model_id: str
    input_data: Any
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StreamingSession (sync)
# ---------------------------------------------------------------------------


class StreamingSession:
    """Sync context manager that accumulates streaming output into one AuditRecord.

    Returned by :meth:`ARIAStreamingAuditor.start_stream`.  Do not instantiate
    directly — use :class:`ARIAStreamingAuditor`.

    Attributes:
        record_id:        Set after :meth:`finish` completes.  ``None`` before that.
        accumulated_text: Current accumulated text from all :meth:`add_chunk` calls.
        chunk_count:      Number of chunks received so far.
    """

    def __init__(self, config: StreamConfig, backend: Any) -> None:
        self._config = config
        self._backend = backend
        self._chunks: list[str] = []
        self._t0: float = 0.0
        self._finished: bool = False
        self._record_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def record_id(self) -> str | None:
        """ARIA record ID, available after :meth:`finish` is called."""
        return self._record_id

    @property
    def accumulated_text(self) -> str:
        """All chunks joined into a single string."""
        return "".join(self._chunks)

    @property
    def chunk_count(self) -> int:
        """Number of chunks added so far."""
        return len(self._chunks)

    def add_chunk(self, text: str) -> None:
        """Append a text chunk to the accumulation buffer.

        Args:
            text: Incremental text from the model (may be empty string —
                  those are silently ignored to avoid inflating ``chunk_count``).
        """
        if text:
            self._chunks.append(text)

    def finish(self, confidence: float | None = None) -> str:
        """Close the stream and record the full output to ARIA.

        Can be called at most once.  Subsequent calls are no-ops and return
        the original ``record_id``.

        Args:
            confidence: Override for the session-level confidence value.
                        If ``None``, falls back to the value set in
                        :class:`StreamConfig`.

        Returns:
            The ``record_id`` string from the underlying auditor.
        """
        if self._finished:
            return self._record_id or ""

        self._finished = True
        output_text = self.accumulated_text
        latency_ms = int((time.time() - self._t0) * 1000)

        output_data: dict[str, Any] = {
            "text": output_text,
            "chunk_count": len(self._chunks),
            "streamed": True,
        }

        conf = confidence if confidence is not None else self._config.confidence
        merged_meta: dict[str, Any] = {**self._config.metadata, "streamed": True}

        try:
            record_id = self._backend.record(
                self._config.model_id,
                self._config.input_data,
                output_data,
                confidence=conf,
                latency_ms=latency_ms,
                metadata=merged_meta,
            )
        except Exception as exc:
            _log.warning(
                "StreamingSession.finish(): record() failed for model=%s: %s",
                self._config.model_id,
                exc,
            )
            import hashlib
            raw = f"{self._config.model_id}:{self._t0}".encode()
            record_id = "stream_err_" + hashlib.sha256(raw).hexdigest()[:16]

        self._record_id = record_id
        return record_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "StreamingSession":
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self.finish()
        else:
            # Record partial output even on error, flagging it in metadata.
            self._config.metadata["stream_error"] = str(exc_val)
            self.finish()


# ---------------------------------------------------------------------------
# AsyncStreamingSession
# ---------------------------------------------------------------------------


class AsyncStreamingSession:
    """Async context manager that accumulates streaming output into one AuditRecord.

    Returned by :meth:`ARIAStreamingAuditor.start_async_stream`.

    All chunk accumulation is synchronous (just list append — no I/O), so
    ``add_chunk`` is a regular method even in the async variant.  Only
    ``__aexit__`` is async because :meth:`finish` dispatches the
    ``record()`` call via ``asyncio.to_thread`` to avoid blocking the event
    loop.
    """

    def __init__(self, config: StreamConfig, backend: Any) -> None:
        self._session = StreamingSession(config, backend)

    # ------------------------------------------------------------------
    # Proxy API
    # ------------------------------------------------------------------

    @property
    def record_id(self) -> str | None:
        return self._session.record_id

    @property
    def accumulated_text(self) -> str:
        return self._session.accumulated_text

    @property
    def chunk_count(self) -> int:
        return self._session.chunk_count

    def add_chunk(self, text: str) -> None:
        """Append a text chunk.  Thread-safe — just a list append."""
        self._session.add_chunk(text)

    async def finish(self, confidence: float | None = None) -> str:
        """Async close: dispatches record() to a thread to avoid blocking.

        Returns:
            The ``record_id`` string.
        """
        import asyncio
        return await asyncio.to_thread(self._session.finish, confidence)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncStreamingSession":
        self._session._t0 = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._session._config.metadata["stream_error"] = str(exc_val)
        await self.finish()


# ---------------------------------------------------------------------------
# ARIAStreamingAuditor
# ---------------------------------------------------------------------------


class ARIAStreamingAuditor:
    """Adds streaming audit sessions to any ARIA backend.

    Wraps an :class:`~aria.auditor.InferenceAuditor` or
    :class:`~aria.quick.ARIAQuick` instance and exposes
    :meth:`start_stream` and :meth:`start_async_stream` factory methods.

    Args:
        auditor: An initialised ``InferenceAuditor`` instance.
        aria:    An ``ARIAQuick`` instance (alternative to *auditor*).

    Raises:
        ValueError: if both *auditor* and *aria* are supplied simultaneously.

    Example::

        sa = ARIAStreamingAuditor(auditor=auditor)

        with sa.start_stream("gpt-4o", messages) as session:
            for chunk in openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, stream=True
            ):
                session.add_chunk(chunk.choices[0].delta.content or "")

        print(f"Record ID: {session.record_id}")
        print(f"Output:    {session.accumulated_text[:80]}...")
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
    ) -> None:
        if auditor is not None and aria is not None:
            raise ValueError("Supply either 'auditor' or 'aria', not both.")
        self._backend = auditor or aria
        if self._backend is None:
            raise ValueError("Supply at least one of 'auditor' or 'aria'.")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def start_stream(
        self,
        model_id: str,
        input_data: Any,
        *,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamingSession:
        """Create a new synchronous streaming session.

        Args:
            model_id:    Model identifier.
            input_data:  Input that triggered the stream.
            confidence:  Optional confidence override for the final AuditRecord.
            metadata:    Additional metadata merged into the AuditRecord.

        Returns:
            A :class:`StreamingSession` ready to be used as a context manager
            or called manually.
        """
        config = StreamConfig(
            model_id=model_id,
            input_data=input_data,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )
        return StreamingSession(config, self._backend)

    def start_async_stream(
        self,
        model_id: str,
        input_data: Any,
        *,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncStreamingSession:
        """Create a new asynchronous streaming session.

        Args:
            model_id:    Model identifier.
            input_data:  Input that triggered the stream.
            confidence:  Optional confidence override for the final AuditRecord.
            metadata:    Additional metadata merged into the AuditRecord.

        Returns:
            An :class:`AsyncStreamingSession` ready to be used as an async
            context manager.
        """
        config = StreamConfig(
            model_id=model_id,
            input_data=input_data,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )
        return AsyncStreamingSession(config, self._backend)
