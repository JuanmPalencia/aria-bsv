"""Tests for ARIA OpenTelemetry integration."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aria.telemetry.otel import ARIASpanExporter, ARIAMeterProvider, otel_available
from aria.core.record import AuditRecord


def _make_record(**kwargs) -> AuditRecord:
    defaults = dict(
        epoch_id="ep-otel-1",
        model_id="model-v1",
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=1,
        confidence=0.88,
        latency_ms=120,
        metadata={},
    )
    defaults.update(kwargs)
    return AuditRecord(**defaults)


class TestARIASpanExporter:
    def test_on_record_noop_without_otel(self):
        """on_record is safe to call even when opentelemetry is not installed."""
        exporter = ARIASpanExporter()
        exporter._tracer = None  # Simulate no OTEL
        record = _make_record()
        # Should not raise
        exporter.on_record(record)

    def test_on_record_emits_span_when_otel_available(self):
        """If a tracer is configured, on_record creates a span with correct attrs."""
        mock_span = MagicMock()
        mock_ctx_manager = MagicMock()
        mock_ctx_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx_manager.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_ctx_manager

        exporter = ARIASpanExporter()
        exporter._tracer = mock_tracer

        record = _make_record()
        exporter.on_record(record)

        mock_tracer.start_as_current_span.assert_called_once()
        span_name = mock_tracer.start_as_current_span.call_args[0][0]
        assert "model-v1" in span_name

        mock_span.set_attributes.assert_called()
        attrs = mock_span.set_attributes.call_args[0][0]
        assert attrs["aria.model_id"] == "model-v1"
        assert attrs["aria.epoch_id"] == "ep-otel-1"
        assert attrs["aria.latency_ms"] == 120

    def test_on_record_sets_confidence_attribute(self):
        """Confidence is set as a separate attribute when present."""
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_ctx

        exporter = ARIASpanExporter()
        exporter._tracer = mock_tracer

        record = _make_record(confidence=0.77)
        exporter.on_record(record)

        calls = {k: v for call in mock_span.set_attribute.call_args_list for k, v in [call[0]]}
        assert "aria.confidence" in calls
        assert abs(calls["aria.confidence"] - 0.77) < 1e-6

    def test_on_record_no_confidence_skips_attribute(self):
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_span)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_ctx

        exporter = ARIASpanExporter()
        exporter._tracer = mock_tracer

        record = _make_record(confidence=None)
        exporter.on_record(record)

        attr_names = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        assert "aria.confidence" not in attr_names

    def test_on_record_swallows_span_errors(self):
        """If span creation raises, on_record must not propagate the exception."""
        exporter = ARIASpanExporter()
        exporter._tracer = MagicMock()
        exporter._tracer.start_as_current_span.side_effect = RuntimeError("OTEL down")
        # Must not raise
        exporter.on_record(_make_record())


class TestARIAMeterProvider:
    def test_on_record_noop_without_otel(self):
        provider = ARIAMeterProvider()
        provider._meter = None
        # Should not raise
        provider.on_record(_make_record())

    def test_on_record_increments_counter(self):
        mock_counter = MagicMock()
        mock_lat_hist = MagicMock()
        mock_conf_hist = MagicMock()

        provider = ARIAMeterProvider()
        provider._meter = MagicMock()
        provider._counter = mock_counter
        provider._latency_hist = mock_lat_hist
        provider._confidence_hist = mock_conf_hist

        record = _make_record(latency_ms=200, confidence=0.9)
        provider.on_record(record)

        mock_counter.add.assert_called_once()
        args, kwargs = mock_counter.add.call_args
        assert args[0] == 1  # increment by 1

        mock_lat_hist.record.assert_called_once()
        lat_args = mock_lat_hist.record.call_args[0]
        assert lat_args[0] == 200

        mock_conf_hist.record.assert_called_once()
        conf_args = mock_conf_hist.record.call_args[0]
        assert abs(conf_args[0] - 0.9) < 1e-6

    def test_on_record_no_confidence_skips_conf_hist(self):
        provider = ARIAMeterProvider()
        provider._meter = MagicMock()
        provider._counter = MagicMock()
        provider._latency_hist = MagicMock()
        provider._confidence_hist = MagicMock()

        provider.on_record(_make_record(confidence=None))
        provider._confidence_hist.record.assert_not_called()

    def test_on_record_swallows_metric_errors(self):
        provider = ARIAMeterProvider()
        provider._meter = MagicMock()
        provider._counter = MagicMock()
        provider._counter.add.side_effect = RuntimeError("metrics down")
        provider._latency_hist = MagicMock()
        provider._confidence_hist = MagicMock()
        # Must not raise
        provider.on_record(_make_record())


class TestOtelAvailable:
    def test_returns_bool(self):
        result = otel_available()
        assert isinstance(result, bool)
