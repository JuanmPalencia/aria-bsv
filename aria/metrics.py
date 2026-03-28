"""
aria.metrics — Prometheus metrics for ARIA monitoring.

Exports operational metrics from the ARIA system that can be scraped by
Prometheus and visualised in Grafana dashboards.

Requires: pip install aria-bsv[prometheus]
Falls back to no-op counters if prometheus_client is not installed.

Metrics exported
-----------------
aria_inferences_total{system_id, model_id}    — Counter
aria_epochs_opened_total{system_id}           — Counter
aria_epochs_closed_total{system_id}           — Counter
aria_drift_detections_total{system_id, test}  — Counter
aria_compliance_failures_total{system_id}     — Counter
aria_alerts_total{system_id, kind, severity}  — Counter
aria_inference_latency_ms{system_id, model_id} — Histogram
aria_inference_confidence{system_id, model_id} — Histogram
aria_active_epochs{system_id}                 — Gauge

Usage::

    from aria.metrics import ARIAMetrics
    from aria.auditor import InferenceAuditor

    metrics = ARIAMetrics(namespace="my_service")
    auditor = InferenceAuditor(config, storage)

    # Wire inference metrics automatically:
    auditor.add_record_hook(metrics.record_inference_hook)

    # Expose for Prometheus scraping (example with prometheus_client):
    from prometheus_client import start_http_server
    start_http_server(8000)
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# No-op shims (used when prometheus_client is not installed)
# ---------------------------------------------------------------------------

class _NoOpCounter:
    def labels(self, **_): return self
    def inc(self, _=1):    pass

class _NoOpHistogram:
    def labels(self, **_): return self
    def observe(self, _):  pass

class _NoOpGauge:
    def labels(self, **_): return self
    def inc(self, _=1):    pass
    def dec(self, _=1):    pass
    def set(self, _):      pass


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
_CONF_BUCKETS    = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def _try_prometheus(namespace: str) -> dict[str, Any]:
    """Attempt to create real prometheus_client metrics."""
    try:
        import prometheus_client as pc
        # Per-instance registry avoids duplicate registration errors in tests
        registry = pc.CollectorRegistry()
        return {
            "inferences": pc.Counter(
                f"{namespace}_inferences_total",
                "Total inference records audited",
                ["system_id", "model_id"],
                registry=registry,
            ),
            "epochs_opened": pc.Counter(
                f"{namespace}_epochs_opened_total",
                "Total epochs opened",
                ["system_id"],
                registry=registry,
            ),
            "epochs_closed": pc.Counter(
                f"{namespace}_epochs_closed_total",
                "Total epochs closed and anchored",
                ["system_id"],
                registry=registry,
            ),
            "drift_detections": pc.Counter(
                f"{namespace}_drift_detections_total",
                "Total drift alerts fired",
                ["system_id", "test"],
                registry=registry,
            ),
            "compliance_failures": pc.Counter(
                f"{namespace}_compliance_failures_total",
                "Total epoch compliance check failures",
                ["system_id"],
                registry=registry,
            ),
            "alerts": pc.Counter(
                f"{namespace}_alerts_total",
                "Total watchdog alerts fired",
                ["system_id", "kind", "severity"],
                registry=registry,
            ),
            "latency_ms": pc.Histogram(
                f"{namespace}_inference_latency_ms",
                "Inference latency in milliseconds",
                ["system_id", "model_id"],
                buckets=_LATENCY_BUCKETS,
                registry=registry,
            ),
            "confidence": pc.Histogram(
                f"{namespace}_inference_confidence",
                "Inference confidence score [0, 1]",
                ["system_id", "model_id"],
                buckets=_CONF_BUCKETS,
                registry=registry,
            ),
            "active_epochs": pc.Gauge(
                f"{namespace}_active_epochs",
                "Number of currently open epochs",
                ["system_id"],
                registry=registry,
            ),
        }
    except ImportError:
        _log.info(
            "ARIAMetrics: prometheus_client not installed — metrics are no-ops. "
            "pip install aria-bsv[prometheus]"
        )
        return {}


# ---------------------------------------------------------------------------
# ARIAMetrics
# ---------------------------------------------------------------------------

class ARIAMetrics:
    """Prometheus metrics adapter for ARIA.

    Args:
        namespace:  Metric name prefix (default ``"aria"``).
        system_id:  Default system_id label (can be overridden per call).
    """

    def __init__(
        self,
        namespace: str = "aria",
        system_id: str = "",
    ) -> None:
        self._system_id = system_id
        self._prom = _try_prometheus(namespace)
        self._available = bool(self._prom)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def system_id(self) -> str:
        return self._system_id

    def _get(self, name: str):
        return self._prom.get(name, _NoOpCounter())

    def _sid(self, system_id: str | None) -> str:
        return system_id or self._system_id

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def record_inference(
        self,
        model_id: str,
        latency_ms: float | None = None,
        confidence: float | None = None,
        system_id: str | None = None,
    ) -> None:
        """Increment inference counter and observe histogram values."""
        sid = self._sid(system_id)
        self._get("inferences").labels(system_id=sid, model_id=model_id).inc()
        if latency_ms is not None and latency_ms > 0:
            self._get("latency_ms").labels(system_id=sid, model_id=model_id).observe(latency_ms)
        if confidence is not None:
            self._get("confidence").labels(system_id=sid, model_id=model_id).observe(confidence)

    def record_inference_hook(self, record: Any) -> None:
        """Hook compatible with ``InferenceAuditor.add_record_hook()``."""
        self.record_inference(
            model_id=record.model_id or "",
            latency_ms=record.latency_ms,
            confidence=record.confidence,
        )

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def record_epoch_opened(self, system_id: str | None = None) -> None:
        sid = self._sid(system_id)
        self._get("epochs_opened").labels(system_id=sid).inc()
        self._get("active_epochs").labels(system_id=sid).inc()

    def record_epoch_closed(self, system_id: str | None = None) -> None:
        sid = self._sid(system_id)
        self._get("epochs_closed").labels(system_id=sid).inc()
        self._get("active_epochs").labels(system_id=sid).dec()

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    def record_drift_detected(
        self,
        test: str = "js",
        system_id: str | None = None,
    ) -> None:
        sid = self._sid(system_id)
        self._get("drift_detections").labels(system_id=sid, test=test).inc()

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------

    def record_compliance_failure(self, system_id: str | None = None) -> None:
        sid = self._sid(system_id)
        self._get("compliance_failures").labels(system_id=sid).inc()

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def record_alert(
        self,
        kind: str,
        severity: str,
        system_id: str | None = None,
    ) -> None:
        sid = self._sid(system_id)
        self._get("alerts").labels(
            system_id=sid, kind=kind, severity=severity
        ).inc()

    def record_alert_hook(self, alert: Any) -> None:
        """Hook compatible with ``WatchdogDaemon.add_alert_handler()``."""
        self.record_alert(
            kind=str(alert.kind),
            severity=str(alert.severity),
        )

    # ------------------------------------------------------------------
    # Convenience: set gauge directly
    # ------------------------------------------------------------------

    def set_active_epochs(self, count: int, system_id: str | None = None) -> None:
        sid = self._sid(system_id)
        self._get("active_epochs").labels(system_id=sid).set(count)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def prometheus_available(self) -> bool:
        """True if prometheus_client was successfully imported."""
        return self._available
