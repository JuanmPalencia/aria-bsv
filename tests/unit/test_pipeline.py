"""Tests for aria.pipeline — pipeline/chain auditing."""

from __future__ import annotations

import pytest

from aria.pipeline import PipelineAuditor, PipelineStep, PipelineTrace, _TraceContext
from aria.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    return SQLiteStorage(dsn="sqlite://")


class TestPipelineAuditor:
    """Tests for PipelineAuditor."""

    def test_create(self):
        pa = PipelineAuditor("my-pipeline")
        assert pa._pipeline_name == "my-pipeline"
        assert len(pa.traces) == 0

    def test_trace_records_steps(self, storage):
        pa = PipelineAuditor("rag", storage=storage)

        with pa.trace("query-1") as t:
            t.step("retriever", "bge-large", {"q": "hi"}, {"docs": ["d1"]})
            t.step("generator", "gpt-4", {"docs": ["d1"]}, {"answer": "hello"}, confidence=0.95)

        assert len(pa.traces) == 1
        trace = pa.traces[0]
        assert trace.trace_id == "query-1"
        assert trace.step_count == 2
        assert trace.models_used == ["bge-large", "gpt-4"]

    def test_trace_auto_id(self, storage):
        pa = PipelineAuditor("auto-pipeline", storage=storage)
        with pa.trace() as t:
            t.step("s1", "model-a", "in", "out")
        assert pa.traces[0].trace_id.startswith("trace-")

    def test_multiple_traces(self, storage):
        pa = PipelineAuditor("multi", storage=storage)
        for i in range(3):
            with pa.trace(f"t-{i}") as t:
                t.step(f"step-{i}", "model", f"in-{i}", f"out-{i}")
        assert len(pa.traces) == 3

    def test_step_sequence(self, storage):
        pa = PipelineAuditor("seq-test", storage=storage)
        with pa.trace("t") as t:
            s0 = t.step("a", "m1", "i0", "o0")
            s1 = t.step("b", "m2", "i1", "o1")
            s2 = t.step("c", "m3", "i2", "o2")
        assert s0.sequence == 0
        assert s1.sequence == 1
        assert s2.sequence == 2

    def test_step_hashes(self, storage):
        pa = PipelineAuditor("hash-test", storage=storage)
        with pa.trace("t") as t:
            s = t.step("retriever", "bge", {"query": "test"}, {"docs": [1, 2]})
        assert s.input_hash.startswith("sha256:")
        assert s.output_hash.startswith("sha256:")
        assert s.input_hash != s.output_hash

    def test_end_to_end_confidence(self, storage):
        pa = PipelineAuditor("conf-test", storage=storage)
        with pa.trace("t") as t:
            t.step("s1", "m1", "i", "o")
            t.step("s2", "m2", "i", "o", confidence=0.87)
        assert pa.traces[0].end_to_end_confidence == 0.87

    def test_end_to_end_confidence_none(self, storage):
        pa = PipelineAuditor("conf-none", storage=storage)
        with pa.trace("t") as t:
            t.step("s1", "m1", "i", "o")
        assert pa.traces[0].end_to_end_confidence is None

    def test_summary(self, storage):
        pa = PipelineAuditor("summ-test", storage=storage)
        with pa.trace("t1") as t:
            t.step("retriever", "bge", "q", "d")
            t.step("generator", "gpt-4", "d", "a", confidence=0.9)
        summary = t.summary()
        assert "summ-test" in summary
        assert "retriever" in summary
        assert "gpt-4" in summary

    def test_all_traces_summary(self, storage):
        pa = PipelineAuditor("multi-summ", storage=storage)
        for i in range(2):
            with pa.trace(f"t-{i}") as t:
                t.step(f"s-{i}", "m", f"in-{i}", f"out-{i}")
        summ = pa.all_traces_summary()
        assert "multi-summ" in summ
        assert "Total traces: 2" in summ

    def test_no_storage(self):
        pa = PipelineAuditor("no-db")
        with pa.trace("t") as t:
            t.step("s", "m", "i", "o")
        assert len(pa.traces) == 1

    def test_step_metadata(self, storage):
        pa = PipelineAuditor("meta-test", storage=storage)
        with pa.trace("t") as t:
            s = t.step("s", "m", "i", "o", metadata={"extra": 42})
        assert s.metadata["extra"] == 42
        assert s.metadata["pipeline"] == "meta-test"
        assert s.metadata["trace_id"] == "t"


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_to_dict(self):
        s = PipelineStep(
            step_name="retriever",
            model_id="bge",
            input_hash="a" * 64,
            output_hash="b" * 64,
            record_id="rec-1",
            sequence=0,
            confidence=0.95,
            latency_ms=50,
            started_at=1000.0,
            finished_at=1000.05,
        )
        d = s.to_dict()
        assert d["step_name"] == "retriever"
        assert d["model_id"] == "bge"
        assert d["confidence"] == 0.95
        assert d["duration_ms"] == 50.0

    def test_duration_ms(self):
        s = PipelineStep(
            step_name="s", model_id="m",
            input_hash="", output_hash="",
            record_id="r", sequence=0,
            started_at=100.0, finished_at=100.1,
        )
        assert abs(s.duration_ms - 100.0) < 0.1


class TestPipelineTrace:
    """Tests for PipelineTrace dataclass."""

    def test_to_dict(self):
        t = PipelineTrace(
            trace_id="t-1",
            pipeline_name="test",
            epoch_id="ep-1",
            started_at=100.0,
            finished_at=101.0,
        )
        d = t.to_dict()
        assert d["trace_id"] == "t-1"
        assert d["pipeline_name"] == "test"
        assert d["step_count"] == 0
        assert d["total_duration_ms"] == 1000.0

    def test_models_used_dedup(self):
        t = PipelineTrace(
            trace_id="t", pipeline_name="p", epoch_id="e",
            steps=[
                PipelineStep("s1", "m1", "", "", "r1", 0, started_at=0, finished_at=0),
                PipelineStep("s2", "m1", "", "", "r2", 1, started_at=0, finished_at=0),
                PipelineStep("s3", "m2", "", "", "r3", 2, started_at=0, finished_at=0),
            ],
        )
        assert t.models_used == ["m1", "m2"]

    def test_empty_trace(self):
        t = PipelineTrace(
            trace_id="t", pipeline_name="p", epoch_id="e",
        )
        assert t.step_count == 0
        assert t.models_used == []
        assert t.end_to_end_confidence is None
