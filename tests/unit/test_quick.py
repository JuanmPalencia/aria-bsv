"""Tests for aria.quick — ARIAQuick zero-blockchain-knowledge API."""

from __future__ import annotations

import os
import time

import pytest

from aria.quick import ARIAQuick, DriftSummary, EpochSummary, quick_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aria(tmp_path, **kwargs) -> ARIAQuick:
    """Create an ARIAQuick instance with a temp database."""
    db = str(tmp_path / "aria_test.db")
    defaults = dict(
        system_id="test-sys",
        db_path=db,
        watchdog=False,
        compliance=True,
        batch_ms=3_600_000,   # 1 hour — we flush manually
        batch_size=100_000,
    )
    defaults.update(kwargs)
    return ARIAQuick(**defaults)


# ---------------------------------------------------------------------------
# EpochSummary
# ---------------------------------------------------------------------------

class TestEpochSummary:
    def test_str_shows_anchored(self):
        s = EpochSummary(
            epoch_id="ep-1", system_id="sys",
            records_count=5,
            open_txid="tx-open", close_txid="tx-close",
            merkle_root="a" * 64,
            anchored=True, compliant=True,
        )
        text = str(s)
        assert "ANCHORED" in text
        assert "COMPLIANT" in text
        assert "ep-1" in text

    def test_str_local_only(self):
        s = EpochSummary(
            epoch_id="ep-1", system_id="s",
            records_count=1,
            open_txid="", close_txid="",
            merkle_root="a" * 32,
            anchored=False, compliant=None,
        )
        assert "LOCAL-ONLY" in str(s)
        assert "NOT-CHECKED" in str(s)

    def test_str_non_compliant_shows_count(self):
        s = EpochSummary(
            epoch_id="ep-1", system_id="s",
            records_count=1,
            open_txid="", close_txid="",
            merkle_root="x",
            anchored=False,
            compliant=False,
            compliance_violations=["v1", "v2"],
        )
        text = str(s)
        assert "NON-COMPLIANT" in text
        assert "2" in text


# ---------------------------------------------------------------------------
# ARIAQuick — core lifecycle
# ---------------------------------------------------------------------------

class TestARIAQuickLifecycle:
    def test_start_and_stop(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        assert aria._started is True
        aria.stop()
        assert aria._started is False

    def test_context_manager(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            assert aria._started

    def test_record_returns_record_id(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            rid = aria.record("gpt-4", {"q": "hi"}, {"a": "hello"}, confidence=0.95)
            assert isinstance(rid, str)
            assert len(rid) > 0

    def test_multiple_records(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            ids = []
            for i in range(10):
                rid = aria.record(
                    "model-v1",
                    {"input": i},
                    {"output": i * 2},
                    confidence=0.9,
                    latency_ms=10.0,
                )
                ids.append(rid)
            assert len(ids) == 10
            assert len(set(ids)) == 10  # all unique

    def test_close_returns_epoch_summary(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            for _ in range(5):
                aria.record("gpt-4", {}, {}, confidence=0.9)
            summary = aria.close()
            assert isinstance(summary, EpochSummary)
            assert summary.epoch_id != ""
            assert summary.system_id == "test-sys"

    def test_start_idempotent(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        aria.start()  # Second start should be a no-op
        assert aria._started is True
        aria.stop()

    def test_stop_idempotent(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.stop()  # Not started — should not raise
        aria.start()
        aria.stop()
        aria.stop()  # Already stopped — should not raise

    def test_record_auto_starts_if_not_started(self, tmp_path):
        aria = _make_aria(tmp_path)
        rid = aria.record("model", {}, {})
        assert isinstance(rid, str)
        aria.stop()

    def test_multiple_close_calls(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            for _ in range(5):
                aria.record("m", {}, {}, confidence=0.8)
            s1 = aria.close()

            for _ in range(5):
                aria.record("m", {}, {}, confidence=0.8)
            s2 = aria.close()

            assert isinstance(s1, EpochSummary)
            assert isinstance(s2, EpochSummary)

    def test_close_without_records_works(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            # Flush an empty epoch
            summary = aria.close()
            assert isinstance(summary, EpochSummary)

    def test_records_stored_in_db(self, tmp_path):
        """Records should appear in storage."""
        with _make_aria(tmp_path) as aria:
            for i in range(8):
                aria.record("bert", {"txt": f"t{i}"}, {"score": 0.5}, confidence=0.9)
            aria.close()
            # List epochs and check records
            epochs = aria.storage.list_epochs(system_id="test-sys", limit=10)
            assert len(epochs) >= 1


# ---------------------------------------------------------------------------
# track decorator
# ---------------------------------------------------------------------------

class TestTrackDecorator:
    def test_decorator_audits_call(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()

        calls = []
        original_record = aria.record
        def spy_record(*a, **kw):
            calls.append(True)
            return original_record(*a, **kw)
        aria.record = spy_record  # type: ignore

        @aria.track("test-model")
        def my_func(x):
            return x * 2

        result = my_func(21)
        assert result == 42
        assert len(calls) == 1
        aria.stop()

    def test_decorator_preserves_return_value(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()

        @aria.track("m")
        def compute(n):
            return n ** 2

        assert compute(5) == 25
        assert compute(10) == 100
        aria.stop()

    def test_decorator_preserves_function_name(self, tmp_path):
        aria = _make_aria(tmp_path)

        @aria.track("m")
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"

    def test_decorator_handles_exception_in_record(self, tmp_path, monkeypatch):
        """Even if record() fails, the wrapped function still returns."""
        aria = _make_aria(tmp_path)
        aria.start()

        def broken_record(*a, **kw):
            raise RuntimeError("storage failure")

        monkeypatch.setattr(aria, "record", broken_record)

        @aria.track("m")
        def safe_func():
            return "ok"

        result = safe_func()
        assert result == "ok"
        aria.stop()

    def test_context_manager_with_decorator(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            @aria.track("llm")
            def generate(prompt):
                return f"response to: {prompt}"

            reply = generate("hello")
            assert "hello" in reply


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestARIAQuickProperties:
    def test_system_id_property(self, tmp_path):
        aria = _make_aria(tmp_path, system_id="my-app")
        assert aria.system_id == "my-app"

    def test_current_epoch_id_after_record(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        aria.record("m", {}, {})
        # After a record, current_epoch_id is set by the hook
        assert aria._current_epoch_id is not None
        aria.stop()

    def test_storage_property_auto_starts(self, tmp_path):
        aria = _make_aria(tmp_path)
        storage = aria.storage
        assert storage is not None

    def test_auditor_property_auto_starts(self, tmp_path):
        aria = _make_aria(tmp_path)
        auditor = aria.auditor
        assert auditor is not None


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_no_epochs(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        # No records yet — BatchManager may have started an epoch
        text = aria.summary()
        assert isinstance(text, str)
        aria.stop()

    def test_summary_after_records(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            for _ in range(5):
                aria.record("m", {}, {})
            aria.close()
            text = aria.summary()
            assert "test-sys" in text


# ---------------------------------------------------------------------------
# compliance_report()
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def test_compliance_report_returns_string(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            for i in range(15):
                aria.record("gpt-4", {"q": i}, {"a": i}, confidence=0.9, latency_ms=50.0)
            summary = aria.close()
            report_text = aria.compliance_report(summary.epoch_id)
            assert isinstance(report_text, str)
            assert len(report_text) > 0

    def test_compliance_disabled_returns_message(self, tmp_path):
        aria = _make_aria(tmp_path, compliance=False)
        aria.start()
        text = aria.compliance_report()
        assert "disabled" in text.lower()
        aria.stop()

    def test_no_closed_epochs_message(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        text = aria.compliance_report()
        # Should handle no closed epochs gracefully
        assert isinstance(text, str)
        aria.stop()


# ---------------------------------------------------------------------------
# check_drift()
# ---------------------------------------------------------------------------

class TestCheckDrift:
    def test_drift_check_two_epochs(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            # Epoch A — low confidence
            for i in range(30):
                aria.record("m", {}, {}, confidence=0.2)
            aria.close()

            # Epoch B — high confidence
            for i in range(30):
                aria.record("m", {}, {}, confidence=0.9)
            aria.close()

            ds = aria.check_drift()
            assert isinstance(ds, DriftSummary)
            assert ds.drift_detected is True  # Large shift

    def test_drift_check_requires_two_epochs(self, tmp_path):
        aria = _make_aria(tmp_path)
        aria.start()
        # Only one close (or none)
        with pytest.raises(RuntimeError, match="at least 2"):
            aria.check_drift()
        aria.stop()

    def test_drift_check_explicit_epoch_ids(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            for _ in range(20):
                aria.record("m", {}, {}, confidence=0.5)
            s1 = aria.close()

            for _ in range(20):
                aria.record("m", {}, {}, confidence=0.5)
            s2 = aria.close()

            if s1.epoch_id and s2.epoch_id:
                ds = aria.check_drift(s1.epoch_id, s2.epoch_id, test="ks")
                assert ds.epoch_a == s1.epoch_id
                assert ds.epoch_b == s2.epoch_id
                assert ds.test == "ks"


# ---------------------------------------------------------------------------
# quick_audit() convenience function
# ---------------------------------------------------------------------------

class TestQuickAudit:
    def test_quick_audit_basic(self, tmp_path):
        db = str(tmp_path / "quick.db")
        records = [
            {
                "model_id": "gpt-4",
                "input_data": {"prompt": f"q{i}"},
                "output_data": {"answer": f"a{i}"},
                "confidence": 0.9,
                "latency_ms": 100.0,
            }
            for i in range(10)
        ]
        summary = quick_audit("quick-test", records, db_path=db)
        assert isinstance(summary, EpochSummary)
        assert summary.system_id == "quick-test"

    def test_quick_audit_empty_records(self, tmp_path):
        db = str(tmp_path / "empty.db")
        summary = quick_audit("empty-test", [], db_path=db)
        assert isinstance(summary, EpochSummary)

    def test_quick_audit_minimal_record_keys(self, tmp_path):
        db = str(tmp_path / "minimal.db")
        records = [{"model_id": "m1", "input_data": {}, "output_data": {}}]
        summary = quick_audit("minimal", records, db_path=db)
        assert isinstance(summary, EpochSummary)

    def test_quick_audit_creates_db_file(self, tmp_path):
        db = str(tmp_path / "newfile.db")
        assert not os.path.exists(db)
        quick_audit("test", [{"model_id": "m", "input_data": {}, "output_data": {}}], db_path=db)
        assert os.path.exists(db)

    def test_quick_audit_summary_has_system_id(self, tmp_path):
        db = str(tmp_path / "ep.db")
        summary = quick_audit("ep-test", [
            {"model_id": "m", "input_data": {}, "output_data": {}, "confidence": 0.5}
        ], db_path=db)
        assert summary.system_id == "ep-test"


# ---------------------------------------------------------------------------
# Integration: full workflow
# ---------------------------------------------------------------------------

class TestFullWorkflow:
    def test_full_workflow_three_closes(self, tmp_path):
        """Simulate a real-world multi-epoch workflow."""
        aria = _make_aria(tmp_path)
        aria.start()

        summaries = []
        for epoch_num in range(3):
            for i in range(10):
                aria.record(
                    model_id="bert-v2",
                    input_data={"text": f"sample {i}"},
                    output_data={"label": "pos"},
                    confidence=0.8 + 0.01 * i,
                    latency_ms=50.0,
                )
            summary = aria.close()
            summaries.append(summary)
            assert isinstance(summary, EpochSummary)

        assert len(summaries) == 3
        aria.stop()

    def test_decorator_workflow(self, tmp_path):
        with _make_aria(tmp_path) as aria:
            @aria.track("classifier")
            def classify(text: str) -> dict:
                return {"label": "positive", "confidence": 0.95}

            results = [classify(t) for t in ["good", "bad", "neutral"]]
            assert all("label" in r for r in results)

    def test_compliance_and_drift(self, tmp_path):
        """Full workflow: records + close + compliance + drift."""
        aria = _make_aria(tmp_path)
        aria.start()

        for _ in range(20):
            aria.record("llm", {"in": "x"}, {"out": "y"}, confidence=0.3, latency_ms=100.0)
        s1 = aria.close()

        for _ in range(20):
            aria.record("llm", {"in": "x"}, {"out": "y"}, confidence=0.9, latency_ms=100.0)
        s2 = aria.close()

        # Compliance
        if s1.epoch_id:
            report = aria.compliance_report(s1.epoch_id)
            assert isinstance(report, str)

        # Drift
        ds = aria.check_drift()
        assert isinstance(ds, DriftSummary)
        assert ds.drift_detected is True

        aria.stop()

    def test_system_id_property(self, tmp_path):
        aria = _make_aria(tmp_path, system_id="my-service")
        assert aria.system_id == "my-service"
