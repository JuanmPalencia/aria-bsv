"""Tests for aria.agent_trace — AgentTraceAuditor and AgentStep."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.agent_trace import AgentStep, AgentTraceAuditor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auditor_mock(record_id: str = "rec-001") -> MagicMock:
    mock = MagicMock()
    mock.record.return_value = record_id
    return mock


# ---------------------------------------------------------------------------
# AgentStep dataclass
# ---------------------------------------------------------------------------

class TestAgentStep:
    def test_fields_stored_correctly(self):
        step = AgentStep(
            record_id="r1",
            step_type="llm_call",
            model_id="gpt-4",
            trace_id="trace-abc",
            parent_record_id=None,
            sequence=0,
        )
        assert step.record_id == "r1"
        assert step.step_type == "llm_call"
        assert step.model_id == "gpt-4"
        assert step.trace_id == "trace-abc"
        assert step.parent_record_id is None
        assert step.sequence == 0

    def test_parent_record_id_can_be_set(self):
        step = AgentStep(
            record_id="r2",
            step_type="tool_call",
            model_id="tool:search",
            trace_id="trace-abc",
            parent_record_id="r1",
            sequence=1,
        )
        assert step.parent_record_id == "r1"


# ---------------------------------------------------------------------------
# AgentTraceAuditor initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_auto_generates_trace_id(self):
        trace = AgentTraceAuditor()
        assert trace.trace_id.startswith("trace_")

    def test_explicit_trace_id_preserved(self):
        trace = AgentTraceAuditor(trace_id="my-run-123")
        assert trace.trace_id == "my-run-123"

    def test_both_auditor_and_aria_raises(self):
        with pytest.raises(ValueError, match="not both"):
            AgentTraceAuditor(auditor=MagicMock(), aria=MagicMock())

    def test_steps_empty_on_init(self):
        trace = AgentTraceAuditor()
        assert trace.steps == []


# ---------------------------------------------------------------------------
# record_step — auditor backend
# ---------------------------------------------------------------------------

class TestRecordStepAuditor:
    def test_calls_auditor_record(self):
        auditor = _auditor_mock("rec-001")
        trace = AgentTraceAuditor(auditor=auditor, trace_id="t1")
        step = trace.record_step(
            model_id="planner",
            step_type="llm_call",
            input_data={"q": "hello"},
            output_data={"answer": "hi"},
        )
        auditor.record.assert_called_once()
        assert step.record_id == "rec-001"

    def test_trace_id_injected_into_metadata(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor, trace_id="trace-xyz")
        trace.record_step("m", "llm_call", {}, {})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["trace_id"] == "trace-xyz"

    def test_step_type_injected_into_metadata(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.record_step("m", "tool_call", {}, {})
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["step_type"] == "tool_call"

    def test_step_sequence_increments(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.record_step("m", "llm_call", {}, {})
        trace.record_step("m", "tool_call", {}, {})
        assert trace.steps[0].sequence == 0
        assert trace.steps[1].sequence == 1

    def test_parent_record_id_injected(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.record_step("m", "llm_call", {}, {}, parent_record_id="prev-rec")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["parent_record_id"] == "prev-rec"

    def test_caller_metadata_merged_lower_priority(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor, trace_id="T")
        trace.record_step(
            "m", "llm_call", {}, {},
            metadata={"custom_key": "custom_val", "trace_id": "HIJACK"},
        )
        kwargs = auditor.record.call_args[1]
        # ARIA trace_id must win over caller-supplied trace_id
        assert kwargs["metadata"]["trace_id"] == "T"
        assert kwargs["metadata"]["custom_key"] == "custom_val"

    def test_confidence_forwarded(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.record_step("m", "llm_call", {}, {}, confidence=0.9)
        kwargs = auditor.record.call_args[1]
        assert kwargs["confidence"] == 0.9

    def test_latency_ms_forwarded(self):
        auditor = _auditor_mock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.record_step("m", "llm_call", {}, {}, latency_ms=123.0)
        kwargs = auditor.record.call_args[1]
        assert kwargs["latency_ms"] == 123

    def test_step_appended_to_steps_list(self):
        auditor = _auditor_mock("rec-42")
        trace = AgentTraceAuditor(auditor=auditor)
        step = trace.record_step("m", "llm_call", {}, {})
        assert len(trace.steps) == 1
        assert trace.steps[0].record_id == "rec-42"

    def test_steps_returns_copy(self):
        trace = AgentTraceAuditor()
        steps = trace.steps
        steps.append("injected")
        assert len(trace.steps) == 0


# ---------------------------------------------------------------------------
# record_step — ARIAQuick (aria) backend
# ---------------------------------------------------------------------------

class TestRecordStepAria:
    def test_calls_aria_record(self):
        aria = MagicMock()
        aria.record.return_value = "aria-rec-001"
        trace = AgentTraceAuditor(aria=aria, trace_id="t2")
        step = trace.record_step("m", "llm_call", {}, {})
        aria.record.assert_called_once()
        assert step.record_id == "aria-rec-001"


# ---------------------------------------------------------------------------
# record_step — no backend (noop)
# ---------------------------------------------------------------------------

class TestRecordStepNoop:
    def test_noop_returns_placeholder_record_id(self):
        trace = AgentTraceAuditor(trace_id="noop-trace")
        step = trace.record_step("my-model", "llm_call", {}, {})
        assert step.record_id.startswith("noop_")

    def test_noop_step_still_appended(self):
        trace = AgentTraceAuditor()
        trace.record_step("m", "tool_call", {}, {})
        assert len(trace.steps) == 1


# ---------------------------------------------------------------------------
# record_step — auditor error resilience
# ---------------------------------------------------------------------------

class TestRecordStepErrorHandling:
    def test_auditor_error_returns_placeholder_not_raises(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("BSV offline")
        trace = AgentTraceAuditor(auditor=auditor)
        step = trace.record_step("m", "llm_call", {}, {})
        assert step.record_id.startswith("err_")

    def test_trace_continues_after_error(self):
        auditor = MagicMock()
        auditor.record.side_effect = [RuntimeError("fail"), "ok-rec"]
        trace = AgentTraceAuditor(auditor=auditor)
        step1 = trace.record_step("m", "llm_call", {}, {})
        step2 = trace.record_step("m", "llm_call", {}, {})
        assert step1.record_id.startswith("err_")
        assert step2.record_id == "ok-rec"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------

class TestFlush:
    def test_flush_calls_auditor_flush(self):
        auditor = MagicMock()
        trace = AgentTraceAuditor(auditor=auditor)
        trace.flush()
        auditor.flush.assert_called_once()

    def test_flush_calls_aria_close_if_available(self):
        aria = MagicMock(spec=["record", "flush"])
        trace = AgentTraceAuditor(aria=aria)
        trace.flush()
        aria.flush.assert_called_once()

    def test_flush_no_backend_is_noop(self):
        trace = AgentTraceAuditor()
        trace.flush()  # must not raise

    def test_flush_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.flush.side_effect = RuntimeError("offline")
        trace = AgentTraceAuditor(auditor=auditor)
        trace.flush()  # must not raise


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_returns_self(self):
        trace = AgentTraceAuditor()
        with trace as t:
            assert t is trace

    def test_exit_calls_flush(self):
        auditor = MagicMock()
        trace = AgentTraceAuditor(auditor=auditor)
        with trace:
            pass
        auditor.flush.assert_called_once()

    def test_exit_called_on_exception(self):
        auditor = MagicMock()
        trace = AgentTraceAuditor(auditor=auditor)
        try:
            with trace:
                raise ValueError("inner error")
        except ValueError:
            pass
        auditor.flush.assert_called_once()
