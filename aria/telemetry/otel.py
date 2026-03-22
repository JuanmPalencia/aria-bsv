"""
aria.telemetry.otel — OpenTelemetry integration for ARIA.

Provides two integration points:

ARIASpanExporter
    Attaches to the InferenceAuditor's record() pipeline and creates an OTEL
    trace span for every inference.  Attributes exported per span:
        aria.model_id, aria.epoch_id, aria.latency_ms, aria.confidence,
        aria.input_hash, aria.output_hash, aria.record_id, aria.sequence

ARIAMeterProvider
    Registers OTEL metrics instruments:
        aria.inference.count          (Counter)   — total inferences
        aria.inference.latency_ms     (Histogram) — latency distribution
        aria.inference.confidence     (Histogram) — confidence distribution
        aria.epoch.records_count      (Gauge)     — records in last closed epoch

Usage::

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from aria.telemetry.otel import ARIASpanExporter, ARIAMeterProvider

    # Span tracing
    provider = TracerProvider()
    aria_exporter = ARIASpanExporter(tracer_provider=provider)
    # Plug into auditor:
    auditor.add_record_hook(aria_exporter.on_record)

    # Metrics
    meter_prov = ARIAMeterProvider(service_name="my-ai-service")
    auditor.add_record_hook(meter_prov.on_record)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..core.record import AuditRecord

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Span, StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    pass


class ARIASpanExporter:
    """Creates an OTEL span for every ARIA audit record.

    Designed to be registered as a record hook on ``InferenceAuditor``::

        exporter = ARIASpanExporter()
        auditor.add_record_hook(exporter.on_record)

    Or used standalone by calling ``on_record(record)`` directly.

    Args:
        tracer_provider: An OTEL TracerProvider.  If None, uses the global
                         provider from ``opentelemetry.trace.get_tracer_provider()``.
        service_name:    Value of the ``service.name`` resource attribute.
    """

    TRACER_NAME = "aria.auditor"

    def __init__(
        self,
        tracer_provider: Any | None = None,
        service_name: str = "aria",
    ) -> None:
        self._service_name = service_name
        self._tracer: Any = None

        if not _OTEL_AVAILABLE:
            return

        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer(self.TRACER_NAME, schema_url="https://opentelemetry.io/schemas/1.24.0")

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_record(self, record: "AuditRecord") -> None:
        """Create a finished span from an AuditRecord.

        Safe to call even when opentelemetry-sdk is not installed — the call
        is a no-op in that case.
        """
        if self._tracer is None:
            return
        try:
            self._emit_span(record)
        except Exception:
            pass  # Never let telemetry crash the audit pipeline

    def _emit_span(self, record: "AuditRecord") -> None:
        span_name = f"aria.inference/{record.model_id}"
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attributes({
                "aria.record_id": record.record_id,
                "aria.epoch_id": record.epoch_id,
                "aria.model_id": record.model_id,
                "aria.input_hash": record.input_hash,
                "aria.output_hash": record.output_hash,
                "aria.sequence": record.sequence,
                "aria.latency_ms": record.latency_ms,
                "aria.version": record.aria_version,
                "service.name": self._service_name,
            })
            if record.confidence is not None:
                span.set_attribute("aria.confidence", record.confidence)
            span.set_status(StatusCode.OK)


class ARIAMeterProvider:
    """Registers ARIA-specific OTEL metrics instruments.

    Args:
        meter_provider: An OTEL MeterProvider.  If None, uses the global one.
        service_name:   Value of the ``service.name`` resource attribute.
    """

    METER_NAME = "aria.auditor"

    def __init__(
        self,
        meter_provider: Any | None = None,
        service_name: str = "aria",
    ) -> None:
        self._service_name = service_name
        self._meter: Any = None
        self._counter: Any = None
        self._latency_hist: Any = None
        self._confidence_hist: Any = None

        if not _OTEL_AVAILABLE:
            return

        provider = meter_provider or metrics.get_meter_provider()
        self._meter = provider.get_meter(self.METER_NAME)

        self._counter = self._meter.create_counter(
            name="aria.inference.count",
            description="Total number of AI inference records submitted to ARIA.",
            unit="1",
        )
        self._latency_hist = self._meter.create_histogram(
            name="aria.inference.latency_ms",
            description="Inference latency in milliseconds.",
            unit="ms",
        )
        self._confidence_hist = self._meter.create_histogram(
            name="aria.inference.confidence",
            description="Model inference confidence score (0.0–1.0).",
            unit="1",
        )

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_record(self, record: "AuditRecord") -> None:
        """Record metrics for an AuditRecord.  No-op if OTEL not installed."""
        if self._meter is None:
            return
        try:
            attrs = {"aria.model_id": record.model_id, "aria.epoch_id": record.epoch_id}
            self._counter.add(1, attrs)
            self._latency_hist.record(record.latency_ms, attrs)
            if record.confidence is not None:
                self._confidence_hist.record(record.confidence, attrs)
        except Exception:
            pass


def otel_available() -> bool:
    """Return True if opentelemetry-api/sdk packages are importable."""
    return _OTEL_AVAILABLE
