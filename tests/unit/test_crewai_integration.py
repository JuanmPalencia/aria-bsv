"""Tests for aria.integrations.crewai — ARIACrewCallback."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aria.integrations.crewai import ARIACrewCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _callback(auditor=None, aria=None, model_id="crewai-agent"):
    return ARIACrewCallback(auditor=auditor, aria=aria, model_id=model_id)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_model_id(self):
        cb = ARIACrewCallback()
        assert cb._model_id == "crewai-agent"

    def test_custom_model_id(self):
        cb = ARIACrewCallback(model_id="my-crew")
        assert cb._model_id == "my-crew"

    def test_start_times_empty(self):
        cb = ARIACrewCallback()
        assert cb._start_times == {}


# ---------------------------------------------------------------------------
# on_task_start / on_task_end
# ---------------------------------------------------------------------------

class TestTaskStartEnd:
    def test_task_end_calls_auditor_record(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("Analyse data", agent_role="Analyst")
        cb.on_task_end("Analyse data", agent_role="Analyst", output="done")
        auditor.record.assert_called_once()

    def test_model_id_includes_agent_role(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor, model_id="base")
        cb.on_task_start("task", agent_role="Writer")
        cb.on_task_end("task", agent_role="Writer", output="text")
        args = auditor.record.call_args[0]
        assert args[0] == "base/Writer"

    def test_provider_metadata_is_crewai(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("task", agent_role="A")
        cb.on_task_end("task", agent_role="A", output="out")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "crewai"

    def test_task_description_in_input_data(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("Write a report", agent_role="Writer")
        cb.on_task_end("Write a report", agent_role="Writer", output="report")
        args = auditor.record.call_args[0]
        assert "Write a report" in args[1]["task"]

    def test_output_in_output_data(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("t", agent_role="A")
        cb.on_task_end("t", agent_role="A", output="final output")
        args = auditor.record.call_args[0]
        assert args[2]["output"] == "final output"

    def test_output_truncated_to_400(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("t", agent_role="A")
        cb.on_task_end("t", agent_role="A", output="x" * 500)
        args = auditor.record.call_args[0]
        assert len(args[2]["output"]) == 400

    def test_task_description_truncated_to_400(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        long_task = "t" * 500
        cb.on_task_start(long_task, agent_role="A")
        cb.on_task_end(long_task, agent_role="A", output="out")
        args = auditor.record.call_args[0]
        assert len(args[1]["task"]) == 400

    def test_latency_tracked_between_start_and_end(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_start("slow task", agent_role="Worker")
        # Backdate start time to simulate elapsed time
        key = "slow task"[:40] + ":Worker"
        cb._start_times[key] = time.monotonic() - 0.05  # ~50ms ago
        cb.on_task_end("slow task", agent_role="Worker", output="done")
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] >= 40

    def test_start_time_cleared_after_task_end(self):
        cb = _callback()
        cb.on_task_start("t", agent_role="A")
        key = "t:A"
        assert key in cb._start_times
        cb.on_task_end("t", agent_role="A", output="out")
        assert key not in cb._start_times

    def test_task_end_without_start_uses_zero_latency(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_task_end("no start", agent_role="A", output="out")
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] == 0

    def test_aria_backend(self):
        aria = MagicMock()
        cb = _callback(aria=aria)
        cb.on_task_start("t", agent_role="A")
        cb.on_task_end("t", agent_role="A", output="out")
        aria.record.assert_called_once()

    def test_no_backend_is_noop(self):
        cb = _callback()
        cb.on_task_start("t", agent_role="A")
        cb.on_task_end("t", agent_role="A", output="out")  # must not raise

    def test_auditor_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("down")
        cb = _callback(auditor=auditor)
        cb.on_task_start("t", agent_role="A")
        cb.on_task_end("t", agent_role="A", output="out")  # must not raise


# ---------------------------------------------------------------------------
# on_agent_action
# ---------------------------------------------------------------------------

class TestOnAgentAction:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_action("Analyst", "search_web", "BSV price")
        auditor.record.assert_called_once()

    def test_event_metadata_is_agent_action(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_action("A", "search", "query")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "agent_action"

    def test_action_in_input_data(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_action("A", "my_tool", "input_val")
        args = auditor.record.call_args[0]
        assert args[1]["action"] == "my_tool"

    def test_input_truncated_to_400(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_action("A", "tool", "x" * 500)
        args = auditor.record.call_args[0]
        assert len(args[1]["input"]) == 400

    def test_aria_backend(self):
        aria = MagicMock()
        cb = _callback(aria=aria)
        cb.on_agent_action("A", "tool", "input")
        aria.record.assert_called_once()


# ---------------------------------------------------------------------------
# on_agent_finish
# ---------------------------------------------------------------------------

class TestOnAgentFinish:
    def test_calls_auditor_record(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_finish("Writer", "Final output text")
        auditor.record.assert_called_once()

    def test_event_metadata_is_agent_finish(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_finish("A", "done")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["event"] == "agent_finish"

    def test_output_in_output_data(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_finish("A", "the answer")
        args = auditor.record.call_args[0]
        assert args[2]["output"] == "the answer"

    def test_output_truncated_to_400(self):
        auditor = MagicMock()
        cb = _callback(auditor=auditor)
        cb.on_agent_finish("A", "x" * 500)
        args = auditor.record.call_args[0]
        assert len(args[2]["output"]) == 400
