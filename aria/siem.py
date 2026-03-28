"""
aria.siem — SIEM (Security Information and Event Management) export adapter.

Formats ARIA alerts and inference anomalies as structured events suitable
for ingestion by Splunk, Elasticsearch/OpenSearch, Datadog, or any CEF/JSON
SIEM. Supports webhook delivery and file sink.

Usage::

    from aria.siem import SIEMExporter, SIEMFormat

    exporter = SIEMExporter(
        fmt=SIEMFormat.JSON,
        endpoint="https://my-siem.corp/ingest",
        api_key="Bearer sk-...",
    )

    exporter.emit_alert(alert)
    exporter.emit_epoch_anomaly(epoch_id, metric, value, threshold)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


class SIEMFormat(str, Enum):
    JSON     = "json"       # Newline-delimited JSON
    CEF      = "cef"        # Common Event Format (Splunk/ArcSight)
    LEEF     = "leef"       # Log Event Extended Format (IBM QRadar)


class SIEMSeverity(str, Enum):
    INFO     = "info"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SIEMEvent:
    """Normalised SIEM event."""
    event_type:   str
    severity:     SIEMSeverity
    source:       str
    timestamp:    str
    message:      str
    payload:      dict

    @classmethod
    def from_alert(cls, alert: Any) -> "SIEMEvent":
        severity_map = {
            "critical": SIEMSeverity.CRITICAL,
            "high":     SIEMSeverity.HIGH,
            "medium":   SIEMSeverity.MEDIUM,
            "low":      SIEMSeverity.LOW,
        }
        raw_sev = str(getattr(alert, "severity", "medium")).lower()
        sev = severity_map.get(raw_sev, SIEMSeverity.MEDIUM)

        return cls(
            event_type="aria.alert",
            severity=sev,
            source=str(getattr(alert, "model_id", "aria")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=str(getattr(alert, "message", "")),
            payload={
                "alert_id":   str(getattr(alert, "alert_id", "")),
                "rule_name":  str(getattr(alert, "rule_name", "")),
                "epoch_id":   str(getattr(alert, "epoch_id", "")),
                "model_id":   str(getattr(alert, "model_id", "")),
                "metric":     str(getattr(alert, "metric", "")),
                "value":      getattr(alert, "value", None),
                "threshold":  getattr(alert, "threshold", None),
            },
        )

    @classmethod
    def from_anomaly(
        cls,
        epoch_id: str,
        metric: str,
        value: float,
        threshold: float,
        severity: SIEMSeverity = SIEMSeverity.HIGH,
    ) -> "SIEMEvent":
        return cls(
            event_type="aria.anomaly",
            severity=severity,
            source="aria-watchdog",
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Anomaly detected: {metric}={value:.4f} exceeds threshold {threshold:.4f}",
            payload={
                "epoch_id":  epoch_id,
                "metric":    metric,
                "value":     value,
                "threshold": threshold,
            },
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _to_json(event: SIEMEvent) -> str:
    d = {
        "event_type": event.event_type,
        "severity":   event.severity.value,
        "source":     event.source,
        "timestamp":  event.timestamp,
        "message":    event.message,
        **event.payload,
    }
    return json.dumps(d)


def _to_cef(event: SIEMEvent) -> str:
    """Common Event Format: CEF:Version|Device Vendor|Device Product|..."""
    sev_map = {
        SIEMSeverity.INFO:     0,
        SIEMSeverity.LOW:      3,
        SIEMSeverity.MEDIUM:   5,
        SIEMSeverity.HIGH:     7,
        SIEMSeverity.CRITICAL: 10,
    }
    sev_num = sev_map.get(event.severity, 5)
    def _escape_cef(val: str) -> str:
        return val.replace("=", "\\=").replace("|", "\\|")

    extensions = " ".join(
        f"{k}={_escape_cef(str(v))}"
        for k, v in event.payload.items()
        if v is not None
    )
    return (
        f"CEF:0|ARIA-BSV|InferenceAuditor|1.0|{event.event_type}|"
        f"{event.message}|{sev_num}|{extensions}"
    )


def _to_leef(event: SIEMEvent) -> str:
    """Log Event Extended Format."""
    attrs = "\t".join(
        f"{k}={v}"
        for k, v in event.payload.items()
        if v is not None
    )
    return (
        f"LEEF:2.0|ARIA-BSV|InferenceAuditor|1.0|{event.event_type}|"
        f"sev={event.severity.value}\tcat={event.event_type}\t{attrs}"
    )


# ---------------------------------------------------------------------------
# SIEMExporter
# ---------------------------------------------------------------------------

class SIEMExporter:
    """Formats and dispatches ARIA events to a SIEM system.

    Args:
        fmt:      Output format (JSON, CEF, or LEEF).
        endpoint: Optional HTTP endpoint for webhook delivery.
        api_key:  Optional Authorization header value.
        file_path: Optional file path to append events to.
        batch_size: Events to buffer before flushing (default 1 = immediate).
    """

    def __init__(
        self,
        fmt: SIEMFormat = SIEMFormat.JSON,
        endpoint: str | None = None,
        api_key:  str | None = None,
        file_path: str | None = None,
        batch_size: int = 1,
    ) -> None:
        self._fmt       = fmt
        self._endpoint  = endpoint
        self._api_key   = api_key
        self._file_path = file_path
        self._batch_size = batch_size
        self._buffer: list[SIEMEvent] = []
        self._emitted: list[str] = []  # formatted strings, for testing

    def emit_alert(self, alert: Any) -> None:
        """Emit an ARIA alert as a SIEM event."""
        self._emit(SIEMEvent.from_alert(alert))

    def emit_anomaly(
        self,
        epoch_id: str,
        metric: str,
        value: float,
        threshold: float,
        severity: SIEMSeverity = SIEMSeverity.HIGH,
    ) -> None:
        """Emit an anomaly detection event."""
        self._emit(SIEMEvent.from_anomaly(epoch_id, metric, value, threshold, severity))

    def emit_raw(self, event: SIEMEvent) -> None:
        """Emit a pre-built SIEMEvent."""
        self._emit(event)

    def flush(self) -> None:
        """Flush any buffered events immediately."""
        while self._buffer:
            self._dispatch(self._buffer.pop(0))

    @property
    def emitted(self) -> list[str]:
        """All formatted event strings emitted so far (for testing)."""
        return list(self._emitted)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, event: SIEMEvent) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def _format(self, event: SIEMEvent) -> str:
        if self._fmt == SIEMFormat.CEF:
            return _to_cef(event)
        if self._fmt == SIEMFormat.LEEF:
            return _to_leef(event)
        return _to_json(event)

    def _dispatch(self, event: SIEMEvent) -> None:
        formatted = self._format(event)
        self._emitted.append(formatted)

        if self._file_path:
            try:
                with open(self._file_path, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception as exc:
                _log.warning("SIEMExporter: file write error: %s", exc)

        if self._endpoint:
            self._send_http(formatted)

    def _send_http(self, payload: str) -> None:
        try:
            import urllib.request
            req = urllib.request.Request(
                self._endpoint,
                data=payload.encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": self._api_key} if self._api_key else {}),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                _log.debug("SIEM: HTTP %s", resp.status)
        except Exception as exc:
            _log.warning("SIEMExporter: HTTP delivery failed: %s", exc)
