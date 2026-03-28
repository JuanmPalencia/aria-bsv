"""Tests for aria.integrations.llamaindex — ARIACallbackHandler."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aria.integrations.llamaindex import ARIACallbackHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handler(auditor=None, aria=None, model_id=None):
    return ARIACallbackHandler(auditor=auditor, aria=aria, model_id=model_id)


def _evt(name: str):
    """Create a mock event_type that behaves like a LlamaIndex CBEventType."""
    evt = MagicMock()
    evt.value = name
    return evt


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestARIACallbackHandlerInit:
    def test_init_no_llama(self):
        with patch.dict("sys.modules", {"llama_index": None, "llama_index.core": None,
                                         "llama_index.core.callbacks": None,
                                         "llama_index.callbacks": None}):
            h = _handler(auditor=MagicMock())
            assert h._base_class is object

    def test_init_with_auditor(self):
        h = _handler(auditor=MagicMock())
        assert h._auditor is not None
        assert h._aria is None

    def test_init_with_aria(self):
        h = _handler(aria=MagicMock())
        assert h._aria is not None

    def test_model_id_stored(self):
        h = _handler(auditor=MagicMock(), model_id="my-llm")
        assert h._model_id == "my-llm"

    def test_start_trace_noop(self):
        h = _handler()
        h.start_trace("trace-1")  # should not raise

    def test_end_trace_noop(self):
        h = _handler()
        h.end_trace("trace-1")  # should not raise


# ---------------------------------------------------------------------------
# on_event_start / on_event_end
# ---------------------------------------------------------------------------

class TestEventStartEnd:
    def test_start_stores_time(self):
        h = _handler(auditor=MagicMock())
        h.on_event_start(_evt("llm_predict"), event_id="evt-1")
        assert "evt-1" in h._start_times

    def test_start_returns_event_id(self):
        h = _handler(auditor=MagicMock())
        result = h.on_event_start(_evt("query"), event_id="my-id")
        assert result == "my-id"

    def test_end_removes_start_time(self):
        h = _handler(auditor=MagicMock())
        h._start_times["evt-2"] = time.time()
        h.on_event_end(_evt("llm_predict"), event_id="evt-2", payload={})
        assert "evt-2" not in h._start_times

    def test_end_without_start(self):
        h = _handler(auditor=MagicMock())
        # Should not raise even without a prior start
        h.on_event_end(_evt("llm_predict"), event_id="unknown", payload={})


# ---------------------------------------------------------------------------
# LLM event recording
# ---------------------------------------------------------------------------

class TestLLMEventRecording:
    def test_llm_predict_records(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="test-llm")
        payload = {
            "messages": [],
            "response": MagicMock(message="hello"),
        }
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()

    def test_llm_chat_records(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="test-llm")
        payload = {"messages": [], "response": "hi"}
        h.on_event_end(_evt("llm_chat"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()

    def test_llm_uppercase_event(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="m")
        payload = {"messages": [], "response": "resp"}
        h.on_event_end(_evt("LLM_PREDICT"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()

    def test_model_name_from_payload(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        payload = {"messages": [], "response": "resp", "model_name": "gpt-4"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        args = auditor.record.call_args[0]
        assert args[0] == "gpt-4"

    def test_model_id_override_wins(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="override")
        payload = {"messages": [], "response": "resp", "model_name": "gpt-4"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        args = auditor.record.call_args[0]
        assert args[0] == "override"

    def test_messages_truncated(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="m")
        msg = MagicMock()
        msg.role = "user"
        msg.content = "x" * 1000
        payload = {"messages": [msg], "response": "resp"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        args = auditor.record.call_args[0]
        assert len(args[1]["messages"][0]["content"]) == 500

    def test_latency_passed(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor, model_id="m")
        h._start_times["e1"] = time.time() - 0.1  # 100ms ago
        payload = {"messages": [], "response": "resp"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] > 0


# ---------------------------------------------------------------------------
# Embedding event recording
# ---------------------------------------------------------------------------

class TestEmbeddingEventRecording:
    def test_embedding_records(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        payload = {
            "chunks": ["text1", "text2"],
            "embeddings": [[0.1] * 768, [0.2] * 768],
        }
        h.on_event_end(_evt("embedding"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()

    def test_embedding_counts_captured(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        payload = {
            "chunks": ["a", "b", "c"],
            "embeddings": [[0.1] * 10, [0.2] * 10, [0.3] * 10],
        }
        h.on_event_end(_evt("EMBEDDING"), event_id="e1", payload=payload)
        args = auditor.record.call_args[0]
        assert args[1]["chunks_count"] == 3
        assert args[2]["embeddings_count"] == 3


# ---------------------------------------------------------------------------
# Query event recording
# ---------------------------------------------------------------------------

class TestQueryEventRecording:
    def test_query_records(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        payload = {
            "query_str": "What is BSV?",
            "response": "BSV is a blockchain",
        }
        h.on_event_end(_evt("query"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()

    def test_query_content_truncated(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        payload = {
            "query_str": "q" * 1000,
            "response": "r" * 2000,
        }
        h.on_event_end(_evt("query"), event_id="e1", payload=payload)
        args = auditor.record.call_args[0]
        assert len(args[1]["query"]) == 500
        assert len(args[2]["response"]) == 1000

    def test_unknown_event_not_recorded(self):
        auditor = MagicMock()
        h = _handler(auditor=auditor)
        h.on_event_end(_evt("unknown_event"), event_id="e1", payload={})
        auditor.record.assert_not_called()


# ---------------------------------------------------------------------------
# Recording with aria vs auditor
# ---------------------------------------------------------------------------

class TestRecordingRouting:
    def test_uses_aria_when_no_auditor(self):
        aria = MagicMock()
        h = _handler(aria=aria, model_id="m")
        payload = {"messages": [], "response": "resp"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        aria.record.assert_called_once()

    def test_prefers_auditor_over_aria(self):
        auditor = MagicMock()
        aria = MagicMock()
        h = _handler(auditor=auditor, aria=aria, model_id="m")
        payload = {"messages": [], "response": "resp"}
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
        auditor.record.assert_called_once()
        aria.record.assert_not_called()

    def test_record_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        h = _handler(auditor=auditor, model_id="m")
        payload = {"messages": [], "response": "resp"}
        # Should not raise
        h.on_event_end(_evt("llm_predict"), event_id="e1", payload=payload)
