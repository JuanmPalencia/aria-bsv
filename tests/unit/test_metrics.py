"""Tests for aria.metrics — Prometheus metrics adapter."""

from __future__ import annotations

import pytest

from aria.metrics import ARIAMetrics, _NoOpCounter, _NoOpGauge, _NoOpHistogram


# ---------------------------------------------------------------------------
# No-op shims
# ---------------------------------------------------------------------------

class TestNoOpShims:
    def test_counter_labels_and_inc(self):
        c = _NoOpCounter()
        c.labels(system_id="x", model_id="y").inc()  # Should not raise

    def test_histogram_observe(self):
        h = _NoOpHistogram()
        h.labels(system_id="x").observe(42.0)  # Should not raise

    def test_gauge_operations(self):
        g = _NoOpGauge()
        g.labels(system_id="x").inc()
        g.labels(system_id="x").dec()
        g.labels(system_id="x").set(5)


# ---------------------------------------------------------------------------
# ARIAMetrics — works even without prometheus_client
# ---------------------------------------------------------------------------

class TestARIAMetricsNoPrometheus:
    """These tests run regardless of whether prometheus_client is installed."""

    def test_instantiate(self):
        m = ARIAMetrics(namespace="test_aria", system_id="sys-1")
        assert m.system_id == "sys-1"

    @property
    def _metrics(self):
        return ARIAMetrics(namespace="test_no_prom", system_id="sys-1")

    def test_record_inference_no_raise(self):
        m = ARIAMetrics(system_id="s")
        m.record_inference("gpt-4", latency_ms=120.0, confidence=0.95)

    def test_record_inference_none_values(self):
        m = ARIAMetrics(system_id="s")
        m.record_inference("gpt-4", latency_ms=None, confidence=None)

    def test_record_inference_hook(self):
        m = ARIAMetrics(system_id="s")

        class FakeRecord:
            model_id = "gpt-4"
            latency_ms = 50.0
            confidence = 0.88

        m.record_inference_hook(FakeRecord())  # Should not raise

    def test_record_epoch_opened(self):
        m = ARIAMetrics(system_id="s")
        m.record_epoch_opened("s")

    def test_record_epoch_closed(self):
        m = ARIAMetrics(system_id="s")
        m.record_epoch_closed("s")

    def test_record_drift_detected(self):
        m = ARIAMetrics(system_id="s")
        m.record_drift_detected(test="ks", system_id="s")

    def test_record_compliance_failure(self):
        m = ARIAMetrics(system_id="s")
        m.record_compliance_failure("s")

    def test_record_alert(self):
        m = ARIAMetrics(system_id="s")
        m.record_alert(kind="LATENCY_SPIKE", severity="WARNING", system_id="s")

    def test_record_alert_hook(self):
        m = ARIAMetrics(system_id="s")

        class FakeAlert:
            kind = "STUCK_EPOCH"
            severity = "INFO"

        m.record_alert_hook(FakeAlert())

    def test_set_active_epochs(self):
        m = ARIAMetrics(system_id="s")
        m.set_active_epochs(3, "s")

    def test_system_id_fallback(self):
        m = ARIAMetrics(system_id="default-sys")
        # When no system_id passed, uses default
        m.record_inference("model")  # Should not raise

    def test_prometheus_available_is_bool(self):
        m = ARIAMetrics()
        assert isinstance(m.prometheus_available, bool)


# ---------------------------------------------------------------------------
# Mock prometheus_client to verify correct calls
# ---------------------------------------------------------------------------

class FakeMetric:
    def __init__(self, name):
        self.name = name
        self.calls: list[dict] = []

    def labels(self, **kwargs):
        self._last_labels = kwargs
        return self

    def inc(self, n=1):
        self.calls.append({"op": "inc", "n": n, "labels": self._last_labels})

    def observe(self, v):
        self.calls.append({"op": "observe", "v": v, "labels": self._last_labels})

    def set(self, v):
        self.calls.append({"op": "set", "v": v, "labels": self._last_labels})

    def dec(self, n=1):
        self.calls.append({"op": "dec", "n": n, "labels": self._last_labels})


class TestARIAMetricsWithFakePrometheus:
    """Patch the _prom dict to verify routing logic."""

    def _make_metrics(self):
        m = ARIAMetrics(system_id="test-sys")
        fakes = {
            "inferences":         FakeMetric("inferences"),
            "epochs_opened":      FakeMetric("epochs_opened"),
            "epochs_closed":      FakeMetric("epochs_closed"),
            "drift_detections":   FakeMetric("drift_detections"),
            "compliance_failures":FakeMetric("compliance_failures"),
            "alerts":             FakeMetric("alerts"),
            "latency_ms":         FakeMetric("latency_ms"),
            "confidence":         FakeMetric("confidence"),
            "active_epochs":      FakeMetric("active_epochs"),
        }
        m._prom = fakes
        return m, fakes

    def test_inference_increments_counter(self):
        m, fakes = self._make_metrics()
        m.record_inference("gpt-4", latency_ms=100.0, confidence=0.9, system_id="sys")
        assert any(c["op"] == "inc" for c in fakes["inferences"].calls)

    def test_inference_observes_latency(self):
        m, fakes = self._make_metrics()
        m.record_inference("gpt-4", latency_ms=123.0, system_id="sys")
        assert any(c["op"] == "observe" and c["v"] == 123.0 for c in fakes["latency_ms"].calls)

    def test_inference_skips_zero_latency(self):
        m, fakes = self._make_metrics()
        m.record_inference("gpt-4", latency_ms=0)
        assert fakes["latency_ms"].calls == []

    def test_inference_observes_confidence(self):
        m, fakes = self._make_metrics()
        m.record_inference("gpt-4", confidence=0.77, system_id="sys")
        assert any(c["v"] == 0.77 for c in fakes["confidence"].calls)

    def test_epoch_opened_increments_counter_and_gauge(self):
        m, fakes = self._make_metrics()
        m.record_epoch_opened("sys")
        assert any(c["op"] == "inc" for c in fakes["epochs_opened"].calls)
        assert any(c["op"] == "inc" for c in fakes["active_epochs"].calls)

    def test_epoch_closed_decrements_gauge(self):
        m, fakes = self._make_metrics()
        m.record_epoch_closed("sys")
        assert any(c["op"] == "inc" for c in fakes["epochs_closed"].calls)
        assert any(c["op"] == "dec" for c in fakes["active_epochs"].calls)

    def test_drift_increments_counter_with_test_label(self):
        m, fakes = self._make_metrics()
        m.record_drift_detected(test="ks", system_id="sys")
        labels = fakes["drift_detections"].calls[-1]["labels"]
        assert labels["test"] == "ks"

    def test_alert_includes_kind_and_severity(self):
        m, fakes = self._make_metrics()
        m.record_alert(kind="LATENCY_SPIKE", severity="WARNING", system_id="sys")
        labels = fakes["alerts"].calls[-1]["labels"]
        assert labels["kind"] == "LATENCY_SPIKE"
        assert labels["severity"] == "WARNING"

    def test_set_active_epochs(self):
        m, fakes = self._make_metrics()
        m.set_active_epochs(7, "sys")
        ops = [c["op"] for c in fakes["active_epochs"].calls]
        assert "set" in ops
