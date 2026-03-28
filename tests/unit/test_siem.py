"""Tests for aria.siem — SIEMExporter."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from aria.siem import (
    SIEMEvent,
    SIEMExporter,
    SIEMFormat,
    SIEMSeverity,
    _to_cef,
    _to_json,
    _to_leef,
)


# ---------------------------------------------------------------------------
# SIEMEvent helpers
# ---------------------------------------------------------------------------

def _make_alert(**kwargs):
    defaults = dict(
        alert_id="al-1",
        rule_name="low_confidence",
        epoch_id="ep-1",
        model_id="gpt-4o",
        severity="high",
        message="Confidence dropped below threshold",
        metric="confidence",
        value=0.35,
        threshold=0.5,
    )
    defaults.update(kwargs)
    alert = MagicMock()
    for k, v in defaults.items():
        setattr(alert, k, v)
    return alert


def _make_event(severity=SIEMSeverity.HIGH):
    return SIEMEvent(
        event_type="aria.alert",
        severity=severity,
        source="test-model",
        timestamp="2025-01-01T00:00:00+00:00",
        message="Test event",
        payload={"epoch_id": "ep-1", "value": 0.3},
    )


# ---------------------------------------------------------------------------
# SIEMEvent.from_alert
# ---------------------------------------------------------------------------

class TestSIEMEventFromAlert:
    def test_basic(self):
        alert = _make_alert()
        evt = SIEMEvent.from_alert(alert)
        assert evt.event_type == "aria.alert"
        assert evt.severity == SIEMSeverity.HIGH
        assert evt.source == "gpt-4o"
        assert "Confidence" in evt.message

    def test_severity_mapping(self):
        for raw, expected in [
            ("critical", SIEMSeverity.CRITICAL),
            ("high",     SIEMSeverity.HIGH),
            ("medium",   SIEMSeverity.MEDIUM),
            ("low",      SIEMSeverity.LOW),
        ]:
            evt = SIEMEvent.from_alert(_make_alert(severity=raw))
            assert evt.severity == expected

    def test_unknown_severity_defaults_medium(self):
        evt = SIEMEvent.from_alert(_make_alert(severity="unknown"))
        assert evt.severity == SIEMSeverity.MEDIUM

    def test_payload_fields(self):
        evt = SIEMEvent.from_alert(_make_alert())
        assert evt.payload["epoch_id"] == "ep-1"
        assert evt.payload["value"] == pytest.approx(0.35)
        assert evt.payload["threshold"] == pytest.approx(0.5)

    def test_timestamp_set(self):
        evt = SIEMEvent.from_alert(_make_alert())
        assert "T" in evt.timestamp


# ---------------------------------------------------------------------------
# SIEMEvent.from_anomaly
# ---------------------------------------------------------------------------

class TestSIEMEventFromAnomaly:
    def test_basic(self):
        evt = SIEMEvent.from_anomaly("ep-1", "confidence", 0.3, 0.5)
        assert evt.event_type == "aria.anomaly"
        assert "confidence" in evt.message
        assert "0.3" in evt.message

    def test_payload(self):
        evt = SIEMEvent.from_anomaly("ep-1", "latency_ms", 500.0, 200.0, SIEMSeverity.CRITICAL)
        assert evt.payload["metric"] == "latency_ms"
        assert evt.payload["value"] == pytest.approx(500.0)
        assert evt.severity == SIEMSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_json_valid(self):
        evt = _make_event()
        s = _to_json(evt)
        d = json.loads(s)
        assert d["event_type"] == "aria.alert"
        assert d["severity"] == "high"
        assert "epoch_id" in d

    def test_cef_format(self):
        evt = _make_event()
        s = _to_cef(evt)
        assert s.startswith("CEF:0|ARIA-BSV|")
        assert "aria.alert" in s
        assert "Test event" in s

    def test_cef_severity_numeric(self):
        for sev, expected_num in [
            (SIEMSeverity.CRITICAL, "10"),
            (SIEMSeverity.HIGH,     "7"),
            (SIEMSeverity.MEDIUM,   "5"),
            (SIEMSeverity.LOW,      "3"),
            (SIEMSeverity.INFO,     "0"),
        ]:
            evt = _make_event(severity=sev)
            s = _to_cef(evt)
            assert f"|{expected_num}|" in s

    def test_leef_format(self):
        evt = _make_event()
        s = _to_leef(evt)
        assert s.startswith("LEEF:2.0|ARIA-BSV|")
        assert "aria.alert" in s
        assert "sev=high" in s


# ---------------------------------------------------------------------------
# SIEMExporter
# ---------------------------------------------------------------------------

class TestSIEMExporter:
    def test_emit_alert_json(self):
        exp = SIEMExporter(fmt=SIEMFormat.JSON)
        exp.emit_alert(_make_alert())
        assert len(exp.emitted) == 1
        d = json.loads(exp.emitted[0])
        assert d["event_type"] == "aria.alert"

    def test_emit_alert_cef(self):
        exp = SIEMExporter(fmt=SIEMFormat.CEF)
        exp.emit_alert(_make_alert())
        assert exp.emitted[0].startswith("CEF:")

    def test_emit_alert_leef(self):
        exp = SIEMExporter(fmt=SIEMFormat.LEEF)
        exp.emit_alert(_make_alert())
        assert exp.emitted[0].startswith("LEEF:")

    def test_emit_anomaly(self):
        exp = SIEMExporter()
        exp.emit_anomaly("ep-1", "confidence", 0.3, 0.5)
        assert len(exp.emitted) == 1
        d = json.loads(exp.emitted[0])
        assert d["event_type"] == "aria.anomaly"

    def test_emit_raw(self):
        exp = SIEMExporter()
        evt = _make_event()
        exp.emit_raw(evt)
        assert len(exp.emitted) == 1

    def test_batch_size_buffers(self):
        exp = SIEMExporter(batch_size=3)
        exp.emit_alert(_make_alert())
        assert len(exp.emitted) == 0  # Not flushed yet
        exp.emit_alert(_make_alert())
        assert len(exp.emitted) == 0
        exp.emit_alert(_make_alert())
        assert len(exp.emitted) == 3  # Flushed

    def test_flush_empties_buffer(self):
        exp = SIEMExporter(batch_size=10)
        exp.emit_alert(_make_alert())
        exp.emit_alert(_make_alert())
        exp.flush()
        assert len(exp.emitted) == 2

    def test_file_sink(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            exp = SIEMExporter(file_path=path)
            exp.emit_alert(_make_alert())
            with open(path, "r") as f:
                content = f.read()
            assert "aria.alert" in content
        finally:
            os.unlink(path)

    def test_multiple_emits(self):
        exp = SIEMExporter()
        for _ in range(5):
            exp.emit_alert(_make_alert())
        assert len(exp.emitted) == 5


# ---------------------------------------------------------------------------
# SIEMSeverity enum
# ---------------------------------------------------------------------------

class TestSIEMSeverity:
    def test_values(self):
        assert SIEMSeverity.INFO.value     == "info"
        assert SIEMSeverity.CRITICAL.value == "critical"
