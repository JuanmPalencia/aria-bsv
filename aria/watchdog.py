"""
aria.watchdog — Background monitoring daemon for ARIA deployments.

Continuously polls the storage backend for anomalies and fires configurable
alert handlers.  Designed to run as a sidecar thread or standalone process.

Checks performed every ``interval_secs``:
  - STUCK_EPOCH     — epoch open for > ``max_open_secs`` without closing
  - LATENCY_SPIKE   — mean latency > ``latency_spike_factor`` × rolling baseline
  - CONFIDENCE_DROP — rolling mean confidence < ``min_confidence`` threshold
  - EPOCH_MISMATCH  — records_count in epoch row ≠ actual records in storage

Usage::

    from aria.watchdog import WatchdogDaemon, Alert, AlertSeverity
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    watchdog = WatchdogDaemon(storage=storage, system_id="prod-system")

    def my_handler(alert: Alert) -> None:
        print(f"[{alert.severity}] {alert.message}")

    watchdog.add_alert_handler(my_handler)
    watchdog.start()
    # ... run your application ...
    watchdog.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .storage.base import StorageInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """An anomaly detected by the WatchdogDaemon."""
    kind: str               # One of: STUCK_EPOCH, LATENCY_SPIKE, CONFIDENCE_DROP, EPOCH_MISMATCH
    severity: AlertSeverity
    message: str
    epoch_id: str | None = None
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        ep = f" [epoch={self.epoch_id}]" if self.epoch_id else ""
        return f"[{self.severity.value}] {self.kind}{ep}: {self.message}"


# ---------------------------------------------------------------------------
# WatchdogDaemon
# ---------------------------------------------------------------------------

class WatchdogDaemon:
    """Background monitoring thread that checks ARIA storage for anomalies.

    Args:
        storage:              Any StorageInterface implementation.
        system_id:            Filter epochs by this system_id.  None = all systems.
        interval_secs:        How often to run checks (default 60 s).
        max_open_secs:        Epochs open longer than this trigger STUCK_EPOCH
                              (default 3600 s = 1 hour).
        latency_spike_factor: Alert when latency exceeds this multiple of the
                              rolling baseline (default 3.0×).
        min_confidence:       Alert when rolling mean confidence drops below
                              this value.  None = disabled (default None).
        window_epochs:        Number of recent epochs to use for baselines
                              (default 5).
    """

    def __init__(
        self,
        storage: "StorageInterface",
        system_id: str | None = None,
        interval_secs: float = 60.0,
        max_open_secs: float = 3600.0,
        latency_spike_factor: float = 3.0,
        min_confidence: float | None = None,
        window_epochs: int = 5,
    ) -> None:
        self._storage = storage
        self._system_id = system_id
        self._interval = interval_secs
        self._max_open_secs = max_open_secs
        self._latency_spike_factor = latency_spike_factor
        self._min_confidence = min_confidence
        self._window_epochs = window_epochs

        self._handlers: list[Callable[[Alert], None]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Rolling baselines
        self._latency_baseline: list[float] = []
        self._confidence_baseline: list[float] = []
        # Separate tracking sets to avoid cross-contamination between checks
        self._mismatch_checked: set[str] = set()   # epochs already checked for count mismatch
        self._baseline_processed: set[str] = set() # epochs already added to rolling baselines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback to receive Alert objects on anomaly detection."""
        self._handlers.append(handler)

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="aria-watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info("WatchdogDaemon started (interval=%ss)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the monitoring thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("WatchdogDaemon stopped")

    def run_once(self) -> list[Alert]:
        """Run one check cycle synchronously and return all alerts.

        Useful for testing or one-shot health checks without starting a thread.
        """
        alerts: list[Alert] = []
        try:
            epochs = self._storage.list_epochs(system_id=self._system_id, limit=self._window_epochs * 4)
        except Exception as exc:
            logger.warning("WatchdogDaemon: storage.list_epochs failed: %s", exc)
            return alerts

        now_ms = int(time.time() * 1000)

        for row in epochs:
            alerts.extend(self._check_stuck_epoch(row, now_ms))
            alerts.extend(self._check_epoch_mismatch(row))

        # Update baselines and check trends using recent closed epochs
        closed = [r for r in epochs if r.close_txid]
        self._update_baselines(closed)
        alerts.extend(self._check_latency_spike())
        alerts.extend(self._check_confidence_drop())

        for alert in alerts:
            self._fire(alert)

        return alerts

    # ------------------------------------------------------------------
    # Private: check loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.error("WatchdogDaemon check cycle error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self._interval)

    # ------------------------------------------------------------------
    # Private: individual checks
    # ------------------------------------------------------------------

    def _check_stuck_epoch(self, row: "EpochRow", now_ms: int) -> list[Alert]:  # type: ignore[name-defined]
        alerts: list[Alert] = []
        if row.close_txid:
            return alerts  # Already closed

        age_secs = (now_ms - row.opened_at) / 1000
        if age_secs > self._max_open_secs:
            alerts.append(Alert(
                kind="STUCK_EPOCH",
                severity=AlertSeverity.WARNING,
                message=(
                    f"Epoch {row.epoch_id!r} has been open for "
                    f"{age_secs / 3600:.1f} h (threshold {self._max_open_secs / 3600:.1f} h)"
                ),
                epoch_id=row.epoch_id,
                details={"age_secs": age_secs, "threshold_secs": self._max_open_secs},
            ))
        return alerts

    def _check_epoch_mismatch(self, row: "EpochRow") -> list[Alert]:  # type: ignore[name-defined]
        if row.epoch_id in self._mismatch_checked:
            return []  # Only check each epoch once
        self._mismatch_checked.add(row.epoch_id)

        if not row.close_txid:
            return []  # Skip open epochs — count is still growing

        try:
            records = self._storage.list_records_by_epoch(row.epoch_id)
        except Exception:
            return []

        actual = len(records)
        if actual != row.records_count:
            return [Alert(
                kind="EPOCH_MISMATCH",
                severity=AlertSeverity.CRITICAL,
                message=(
                    f"Epoch {row.epoch_id!r}: records_count={row.records_count} "
                    f"but storage has {actual} records"
                ),
                epoch_id=row.epoch_id,
                details={"stored_count": row.records_count, "actual_count": actual},
            )]
        return []

    def _update_baselines(self, closed_rows: list) -> None:
        """Compute per-epoch mean latency/confidence for the rolling window.

        Epochs are sorted by opened_at ascending so that appending to the
        baseline list preserves chronological order — the spike-detection logic
        uses _latency_baseline[-1] as "current" and _latency_baseline[:-1] as
        the historical baseline.
        """
        # Sort ascending by opened_at (robust against non-deterministic DB ordering
        # when multiple epochs share the same millisecond timestamp)
        new_rows = sorted(
            [r for r in closed_rows if r.epoch_id not in self._baseline_processed],
            key=lambda r: r.opened_at,
        )
        # Only take the most recent window_epochs of those
        for row in new_rows[-self._window_epochs:]:
            try:
                records = self._storage.list_records_by_epoch(row.epoch_id)
            except Exception:
                continue
            self._baseline_processed.add(row.epoch_id)
            if not records:
                continue
            lats = [r.latency_ms for r in records]
            mean_lat = sum(lats) / len(lats)
            self._latency_baseline.append(mean_lat)
            if len(self._latency_baseline) > self._window_epochs:
                self._latency_baseline = self._latency_baseline[-self._window_epochs:]

            confs = [r.confidence for r in records if r.confidence is not None]
            if confs:
                mean_conf = sum(confs) / len(confs)
                self._confidence_baseline.append(mean_conf)
                if len(self._confidence_baseline) > self._window_epochs:
                    self._confidence_baseline = self._confidence_baseline[-self._window_epochs:]

    def _check_latency_spike(self) -> list[Alert]:
        if len(self._latency_baseline) < 2:
            return []
        baseline_mean = sum(self._latency_baseline[:-1]) / len(self._latency_baseline[:-1])
        current = self._latency_baseline[-1]
        if baseline_mean > 0 and current > baseline_mean * self._latency_spike_factor:
            return [Alert(
                kind="LATENCY_SPIKE",
                severity=AlertSeverity.WARNING,
                message=(
                    f"Latency spike: current mean {current:.0f} ms is "
                    f"{current / baseline_mean:.1f}× baseline {baseline_mean:.0f} ms"
                ),
                details={"current_ms": current, "baseline_ms": baseline_mean},
            )]
        return []

    def _check_confidence_drop(self) -> list[Alert]:
        if self._min_confidence is None:
            return []
        if not self._confidence_baseline:
            return []
        current = self._confidence_baseline[-1]
        if current < self._min_confidence:
            return [Alert(
                kind="CONFIDENCE_DROP",
                severity=AlertSeverity.WARNING,
                message=(
                    f"Mean confidence {current:.3f} is below threshold "
                    f"{self._min_confidence:.3f}"
                ),
                details={"current": current, "threshold": self._min_confidence},
            )]
        return []

    def _fire(self, alert: Alert) -> None:
        logger.warning("WatchdogDaemon alert: %s", alert)
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as exc:
                logger.error("Alert handler error: %s", exc)
