"""Tests for aria.streaming — StreamingSession, AsyncStreamingSession, ARIAStreamingAuditor."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aria.streaming import (
    ARIAStreamingAuditor,
    AsyncStreamingSession,
    StreamConfig,
    StreamingSession,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backend(record_id: str = "rec-stream-001") -> MagicMock:
    mock = MagicMock()
    mock.record.return_value = record_id
    return mock


# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------

class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig(model_id="gpt-4o", input_data={"q": "hi"})
        assert cfg.confidence is None
        assert cfg.metadata == {}

    def test_explicit_values(self):
        cfg = StreamConfig(
            model_id="gpt-4o",
            input_data="prompt",
            confidence=0.9,
            metadata={"key": "val"},
        )
        assert cfg.confidence == 0.9
        assert cfg.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# StreamingSession — basic API
# ---------------------------------------------------------------------------

class TestStreamingSession:
    def _session(self, backend=None):
        cfg = StreamConfig(model_id="gpt-4o", input_data={"q": "hello"})
        return StreamingSession(cfg, backend or _backend())

    def test_initial_state(self):
        s = self._session()
        assert s.accumulated_text == ""
        assert s.chunk_count == 0
        assert s.record_id is None

    def test_add_chunk_accumulates(self):
        s = self._session()
        s.add_chunk("Hello")
        s.add_chunk(", ")
        s.add_chunk("world")
        assert s.accumulated_text == "Hello, world"
        assert s.chunk_count == 3

    def test_empty_chunks_ignored(self):
        s = self._session()
        s.add_chunk("")
        s.add_chunk("hi")
        s.add_chunk("")
        assert s.chunk_count == 1

    def test_finish_calls_backend_record(self):
        backend = _backend("rec-001")
        s = self._session(backend)
        s._t0 = time.time()
        s.add_chunk("output")
        record_id = s.finish()
        backend.record.assert_called_once()
        assert record_id == "rec-001"

    def test_finish_sets_record_id(self):
        backend = _backend("rec-xyz")
        s = self._session(backend)
        s._t0 = time.time()
        s.finish()
        assert s.record_id == "rec-xyz"

    def test_finish_twice_is_noop(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.finish()
        s.finish()
        assert backend.record.call_count == 1

    def test_finish_output_data_has_streamed_true(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.add_chunk("text")
        s.finish()
        # output_data is the 3rd positional arg to record()
        output_data = backend.record.call_args[0][2]
        assert output_data["streamed"] is True

    def test_finish_output_data_has_accumulated_text(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.add_chunk("Hello")
        s.add_chunk(" world")
        s.finish()
        output_data = backend.record.call_args[0][2]
        assert output_data["text"] == "Hello world"

    def test_finish_output_data_has_chunk_count(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.add_chunk("a")
        s.add_chunk("b")
        s.finish()
        output_data = backend.record.call_args[0][2]
        assert output_data["chunk_count"] == 2

    def test_finish_confidence_override(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.finish(confidence=0.85)
        kwargs = backend.record.call_args[1]
        assert kwargs["confidence"] == 0.85

    def test_finish_confidence_from_config(self):
        cfg = StreamConfig(model_id="m", input_data={}, confidence=0.77)
        backend = _backend()
        s = StreamingSession(cfg, backend)
        s._t0 = time.time()
        s.finish()
        kwargs = backend.record.call_args[1]
        assert kwargs["confidence"] == 0.77

    def test_metadata_streamed_flag_injected(self):
        backend = _backend()
        s = self._session(backend)
        s._t0 = time.time()
        s.finish()
        kwargs = backend.record.call_args[1]
        assert kwargs["metadata"]["streamed"] is True

    def test_backend_error_returns_placeholder(self):
        backend = MagicMock()
        backend.record.side_effect = RuntimeError("BSV offline")
        s = self._session(backend)
        s._t0 = time.time()
        record_id = s.finish()
        assert record_id.startswith("stream_err_")


# ---------------------------------------------------------------------------
# StreamingSession — context manager
# ---------------------------------------------------------------------------

class TestStreamingSessionContextManager:
    def test_enter_returns_self(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = StreamingSession(cfg, backend)
        with s as ctx:
            assert ctx is s

    def test_exit_calls_finish(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = StreamingSession(cfg, backend)
        with s:
            s.add_chunk("output")
        backend.record.assert_called_once()
        assert s.record_id is not None

    def test_latency_measured_over_context(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = StreamingSession(cfg, backend)
        with s:
            time.sleep(0.01)
        kwargs = backend.record.call_args[1]
        assert kwargs["latency_ms"] >= 5

    def test_exception_inside_still_records(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = StreamingSession(cfg, backend)
        try:
            with s:
                s.add_chunk("partial")
                raise ValueError("inner error")
        except ValueError:
            pass
        backend.record.assert_called_once()
        assert "stream_error" in backend.record.call_args[1]["metadata"]


# ---------------------------------------------------------------------------
# AsyncStreamingSession
# ---------------------------------------------------------------------------

class TestAsyncStreamingSession:
    @pytest.mark.asyncio
    async def test_add_chunk_accumulates(self):
        backend = _backend()
        cfg = StreamConfig(model_id="claude", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        s.add_chunk("Hello")
        s.add_chunk(" async")
        assert s.accumulated_text == "Hello async"

    @pytest.mark.asyncio
    async def test_finish_creates_record(self):
        backend = _backend("async-rec-001")
        cfg = StreamConfig(model_id="claude", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        s._session._t0 = time.time()
        record_id = await s.finish()
        assert record_id == "async-rec-001"
        backend.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_records_on_exit(self):
        backend = _backend()
        cfg = StreamConfig(model_id="claude", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        async with s:
            s.add_chunk("async output")
        backend.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_count_property(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        s.add_chunk("a")
        s.add_chunk("b")
        assert s.chunk_count == 2

    @pytest.mark.asyncio
    async def test_record_id_none_before_finish(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        assert s.record_id is None

    @pytest.mark.asyncio
    async def test_exception_inside_still_records(self):
        backend = _backend()
        cfg = StreamConfig(model_id="m", input_data={})
        s = AsyncStreamingSession(cfg, backend)
        try:
            async with s:
                s.add_chunk("partial")
                raise RuntimeError("inner")
        except RuntimeError:
            pass
        backend.record.assert_called_once()


# ---------------------------------------------------------------------------
# ARIAStreamingAuditor
# ---------------------------------------------------------------------------

class TestARIAStreamingAuditor:
    def test_raises_with_both_backends(self):
        with pytest.raises(ValueError, match="not both"):
            ARIAStreamingAuditor(auditor=MagicMock(), aria=MagicMock())

    def test_raises_with_no_backend(self):
        with pytest.raises(ValueError):
            ARIAStreamingAuditor()

    def test_start_stream_returns_streaming_session(self):
        backend = _backend()
        sa = ARIAStreamingAuditor(auditor=backend)
        session = sa.start_stream("gpt-4o", {"q": "hi"})
        assert isinstance(session, StreamingSession)

    def test_start_async_stream_returns_async_session(self):
        backend = _backend()
        sa = ARIAStreamingAuditor(auditor=backend)
        session = sa.start_async_stream("gpt-4o", {"q": "hi"})
        assert isinstance(session, AsyncStreamingSession)

    def test_start_stream_with_aria_backend(self):
        aria = MagicMock()
        sa = ARIAStreamingAuditor(aria=aria)
        session = sa.start_stream("claude", {})
        assert isinstance(session, StreamingSession)

    def test_start_stream_confidence_passed(self):
        backend = _backend()
        sa = ARIAStreamingAuditor(auditor=backend)
        session = sa.start_stream("m", {}, confidence=0.9)
        assert session._config.confidence == 0.9

    def test_start_stream_metadata_passed(self):
        backend = _backend()
        sa = ARIAStreamingAuditor(auditor=backend)
        session = sa.start_stream("m", {}, metadata={"custom": "data"})
        assert session._config.metadata["custom"] == "data"

    def test_full_sync_flow(self):
        backend = _backend("final-rec")
        sa = ARIAStreamingAuditor(auditor=backend)
        with sa.start_stream("gpt-4o", {"messages": []}) as session:
            for token in ["Hello", " ", "world"]:
                session.add_chunk(token)
        assert session.record_id == "final-rec"
        assert session.accumulated_text == "Hello world"

    @pytest.mark.asyncio
    async def test_full_async_flow(self):
        backend = _backend("async-final")
        sa = ARIAStreamingAuditor(auditor=backend)
        async with sa.start_async_stream("claude", {"messages": []}) as session:
            for token in ["Async", " ", "output"]:
                session.add_chunk(token)
        assert session.record_id == "async-final"
        assert session.accumulated_text == "Async output"


# ---------------------------------------------------------------------------
# OpenAI streaming integration
# ---------------------------------------------------------------------------

class TestOpenAIStreamingIntegration:
    def _make_fake_openai_stream(self, texts: list[str]) -> list:
        """Fake OpenAI streaming chunks."""
        chunks = []
        for text in texts:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        return chunks

    def test_stream_iterator_accumulates_and_records(self):
        from aria.integrations.openai import _ARIAStreamIterator
        recorder = MagicMock()
        chunks = self._make_fake_openai_stream(["Hello", " ", "world"])
        it = _ARIAStreamIterator(iter(chunks), "gpt-4o", {}, recorder, time.time(), {"provider": "openai"})
        result = list(it)
        assert len(result) == 3
        recorder.record.assert_called_once()
        kwargs = recorder.record.call_args[1]
        assert kwargs["output_data"]["text"] == "Hello world"

    def test_stream_iterator_empty_content_ignored(self):
        from aria.integrations.openai import _ARIAStreamIterator
        recorder = MagicMock()
        chunk_none = MagicMock()
        chunk_none.choices = [MagicMock()]
        chunk_none.choices[0].delta.content = None
        chunk_text = MagicMock()
        chunk_text.choices = [MagicMock()]
        chunk_text.choices[0].delta.content = "hi"
        it = _ARIAStreamIterator(
            iter([chunk_none, chunk_text]), "m", {}, recorder, time.time(), {}
        )
        list(it)
        kwargs = recorder.record.call_args[1]
        assert kwargs["output_data"]["chunk_count"] == 1

    def test_stream_iterator_context_manager(self):
        from aria.integrations.openai import _ARIAStreamIterator
        recorder = MagicMock()
        it = _ARIAStreamIterator(iter([]), "m", {}, recorder, time.time(), {})
        with it as ctx:
            assert ctx is it


# ---------------------------------------------------------------------------
# Anthropic streaming integration
# ---------------------------------------------------------------------------

class TestAnthropicStreamingIntegration:
    def _make_fake_anthropic_stream(self, texts: list[str]) -> list:
        """Fake Anthropic streaming events."""
        events = []
        for text in texts:
            evt = MagicMock()
            evt.delta = MagicMock()
            evt.delta.text = text
            events.append(evt)
        return events

    def test_anthropic_stream_iterator_accumulates(self):
        from aria.integrations.anthropic_sdk import _ARIAAnthropicStreamIterator
        recorder = MagicMock()
        events = self._make_fake_anthropic_stream(["Hello", " Claude"])
        it = _ARIAAnthropicStreamIterator(
            iter(events), "claude", {}, recorder, time.time(), {"provider": "anthropic"}
        )
        result = list(it)
        assert len(result) == 2
        recorder.record.assert_called_once()
        kwargs = recorder.record.call_args[1]
        assert kwargs["output_data"]["text"] == "Hello Claude"

    def test_anthropic_stream_metadata_has_streamed_true(self):
        from aria.integrations.anthropic_sdk import _ARIAAnthropicStreamIterator
        recorder = MagicMock()
        it = _ARIAAnthropicStreamIterator(
            iter([]), "claude", {}, recorder, time.time(), {"provider": "anthropic"}
        )
        list(it)
        kwargs = recorder.record.call_args[1]
        assert kwargs["metadata"]["streamed"] is True
