"""Tests for aria.integrations.langgraph — ARIALangGraphCallback."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aria.integrations.langgraph import ARIALangGraphCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _callback(auditor=None, aria=None, model_id="langgraph-agent"):
    return ARIALangGraphCallback(auditor=auditor, aria=aria, model_id=model_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNodeStartEnd:
    def test_node_start_end_records_inference(self):
        """on_node_end triggers a record() call on the auditor."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_node_start("my_node", {"x": 1}, run_id="r1")
        cb.on_node_end("my_node", {"y": 2}, run_id="r1")
        auditor.record.assert_called_once()

    def test_latency_tracked_between_start_and_end(self):
        """latency_ms reflects the time elapsed between start and end."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_node_start("slow_node", {}, run_id="r2")
        # Manually backdate the start time to simulate elapsed time
        cb._start_times["r2"] = time.monotonic() - 0.05  # ~50 ms ago
        cb.on_node_end("slow_node", {"result": "ok"}, run_id="r2")
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] >= 40

    def test_error_node_records_with_error_output(self):
        """on_error calls record() and includes error info in output_data."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_node_start("fail_node", {}, run_id="r3")
        cb.on_error("fail_node", ValueError("something broke"), run_id="r3")
        auditor.record.assert_called_once()
        args = auditor.record.call_args[0]
        output_data = args[2]
        assert "error" in output_data
        assert "ValueError" in output_data["error"]

    def test_multiple_nodes_tracked_independently(self):
        """Two concurrent nodes with different run_ids are tracked separately."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_node_start("node_a", {"a": 1}, run_id="ra")
        cb.on_node_start("node_b", {"b": 2}, run_id="rb")
        assert "ra" in cb._start_times
        assert "rb" in cb._start_times
        cb.on_node_end("node_a", {"result": "a"}, run_id="ra")
        cb.on_node_end("node_b", {"result": "b"}, run_id="rb")
        assert auditor.record.call_count == 2

    def test_model_id_uses_node_name(self):
        """model_id passed to record() is '{model_id}/{node_name}'."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor, model_id="my-graph")
        cb.on_node_start("retriever", {}, run_id="r5")
        cb.on_node_end("retriever", {}, run_id="r5")
        args = auditor.record.call_args[0]
        assert args[0] == "my-graph/retriever"

    def test_on_graph_start_does_not_crash(self):
        """on_graph_start is a no-op that must not raise."""
        cb = _callback()
        cb.on_graph_start("graph-1", {"query": "hello"})  # should not raise

    def test_on_graph_end_does_not_crash(self):
        """on_graph_end is a no-op that must not raise."""
        cb = _callback()
        cb.on_graph_end("graph-1", {"answer": "world"})  # should not raise

    def test_aria_quick_integration(self):
        """Works with an ARIAQuick instance passed as aria=."""
        aria = MagicMock()
        cb = _callback(aria=aria)
        cb.on_node_start("node_x", {"in": "val"}, run_id="r6")
        cb.on_node_end("node_x", {"out": "result"}, run_id="r6")
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert "model_id" in kwargs

    def test_record_error_swallowed(self):
        """Exceptions raised inside record() are swallowed and do not propagate."""
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage failure")
        cb = _callback(auditor=auditor)
        cb.on_node_start("n", {}, run_id="r7")
        cb.on_node_end("n", {}, run_id="r7")  # must not raise

    def test_unknown_run_id_on_end_does_not_crash(self):
        """on_node_end with a run_id that was never started must not crash."""
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_node_end("ghost_node", {"result": 42}, run_id="never-started")
        # record is still called, with latency_ms=0
        auditor.record.assert_called_once()
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] == 0
