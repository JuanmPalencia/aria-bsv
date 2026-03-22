"""Tests for WatchdogDaemon."""

from __future__ import annotations

import time

import pytest

from aria.watchdog import WatchdogDaemon, Alert, AlertSeverity
from aria.storage.sqlite import SQLiteStorage
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> SQLiteStorage:
    return SQLiteStorage("sqlite://")


def _make_record(epoch_id: str, latency_ms: int = 100, confidence: float | None = 0.9, seq: int = 1) -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id="m-1",
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=seq,
        confidence=confidence,
        latency_ms=latency_ms,
        metadata={},
    )


def _open_epoch(storage: SQLiteStorage, epoch_id: str, opened_ago_ms: int = 0) -> None:
    now = int(time.time() * 1000) - opened_ago_ms
    storage.save_epoch_open(epoch_id, "sys", "tx-open", {}, "sh", now)


def _close_epoch(storage: SQLiteStorage, epoch_id: str, count: int = 1) -> None:
    now = int(time.time() * 1000)
    storage.save_epoch_close(epoch_id, "tx-close", "sha256:" + "0" * 64, count, now)


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------

class TestAlertModel:
    def test_str_representation(self):
        alert = Alert(
            kind="STUCK_EPOCH",
            severity=AlertSeverity.WARNING,
            message="Epoch open too long",
            epoch_id="ep-1",
        )
        s = str(alert)
        assert "STUCK_EPOCH" in s
        assert "WARNING" in s
        assert "ep-1" in s

    def test_severity_enum_values(self):
        assert AlertSeverity.INFO == "INFO"
        assert AlertSeverity.WARNING == "WARNING"
        assert AlertSeverity.CRITICAL == "CRITICAL"


# ---------------------------------------------------------------------------
# run_once — stuck epoch
# ---------------------------------------------------------------------------

class TestStuckEpochCheck:
    def test_stuck_epoch_triggers_alert(self):
        storage = _make_storage()
        # Open an epoch 2 hours ago (7200 s)
        _open_epoch(storage, "ep-stuck", opened_ago_ms=7200 * 1000)

        watchdog = WatchdogDaemon(storage=storage, max_open_secs=3600)
        alerts = watchdog.run_once()

        stuck = [a for a in alerts if a.kind == "STUCK_EPOCH"]
        assert len(stuck) == 1
        assert stuck[0].epoch_id == "ep-stuck"
        assert stuck[0].severity == AlertSeverity.WARNING

    def test_fresh_epoch_no_alert(self):
        storage = _make_storage()
        _open_epoch(storage, "ep-fresh", opened_ago_ms=60 * 1000)  # 1 min ago

        watchdog = WatchdogDaemon(storage=storage, max_open_secs=3600)
        alerts = watchdog.run_once()

        stuck = [a for a in alerts if a.kind == "STUCK_EPOCH"]
        assert len(stuck) == 0

    def test_closed_epoch_not_flagged(self):
        storage = _make_storage()
        _open_epoch(storage, "ep-closed", opened_ago_ms=7200 * 1000)
        storage.save_record(_make_record("ep-closed"))
        _close_epoch(storage, "ep-closed", count=1)

        watchdog = WatchdogDaemon(storage=storage, max_open_secs=3600)
        alerts = watchdog.run_once()

        stuck = [a for a in alerts if a.kind == "STUCK_EPOCH"]
        assert len(stuck) == 0


# ---------------------------------------------------------------------------
# run_once — epoch mismatch
# ---------------------------------------------------------------------------

class TestEpochMismatchCheck:
    def test_mismatch_triggers_critical_alert(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open("ep-mismatch", "sys", "tx", {}, "sh", now)
        storage.save_record(_make_record("ep-mismatch", seq=1))
        storage.save_record(_make_record("ep-mismatch", seq=2))
        # Close with count=1 but we stored 2 records
        storage.save_epoch_close("ep-mismatch", "tx-c", "root", 1, now + 1)

        watchdog = WatchdogDaemon(storage=storage)
        alerts = watchdog.run_once()

        mismatch = [a for a in alerts if a.kind == "EPOCH_MISMATCH"]
        assert len(mismatch) == 1
        assert mismatch[0].severity == AlertSeverity.CRITICAL

    def test_correct_count_no_alert(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open("ep-ok", "sys", "tx", {}, "sh", now)
        storage.save_record(_make_record("ep-ok", seq=1))
        storage.save_epoch_close("ep-ok", "tx-c", "root", 1, now + 1)

        watchdog = WatchdogDaemon(storage=storage)
        alerts = watchdog.run_once()

        mismatch = [a for a in alerts if a.kind == "EPOCH_MISMATCH"]
        assert len(mismatch) == 0


# ---------------------------------------------------------------------------
# run_once — latency spike
# ---------------------------------------------------------------------------

class TestLatencySpikeCheck:
    def _make_closed_epoch(self, storage, epoch_id, latency_ms):
        now = int(time.time() * 1000)
        storage.save_epoch_open(epoch_id, "sys", "tx", {}, "sh", now)
        storage.save_record(_make_record(epoch_id, latency_ms=latency_ms))
        storage.save_epoch_close(epoch_id, "tx-c", "root", 1, now + 1)

    def test_latency_spike_alert(self):
        storage = _make_storage()
        # Build baseline of 100ms, then spike to 1000ms
        for i, lat in enumerate([100, 100, 100, 1000]):
            self._make_closed_epoch(storage, f"ep-lat-{i}", lat)

        watchdog = WatchdogDaemon(storage=storage, latency_spike_factor=3.0, window_epochs=10)
        alerts = watchdog.run_once()

        spikes = [a for a in alerts if a.kind == "LATENCY_SPIKE"]
        assert len(spikes) == 1
        assert spikes[0].severity == AlertSeverity.WARNING

    def test_no_spike_no_alert(self):
        storage = _make_storage()
        for i in range(4):
            self._make_closed_epoch(storage, f"ep-nosp-{i}", latency_ms=100)

        watchdog = WatchdogDaemon(storage=storage, latency_spike_factor=3.0, window_epochs=10)
        alerts = watchdog.run_once()

        spikes = [a for a in alerts if a.kind == "LATENCY_SPIKE"]
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# run_once — confidence drop
# ---------------------------------------------------------------------------

class TestConfidenceDropCheck:
    def _make_closed_epoch(self, storage, epoch_id, confidence):
        now = int(time.time() * 1000)
        storage.save_epoch_open(epoch_id, "sys", "tx", {}, "sh", now)
        storage.save_record(_make_record(epoch_id, confidence=confidence))
        storage.save_epoch_close(epoch_id, "tx-c", "root", 1, now + 1)

    def test_confidence_drop_alert(self):
        storage = _make_storage()
        self._make_closed_epoch(storage, "ep-cd1", confidence=0.3)

        watchdog = WatchdogDaemon(storage=storage, min_confidence=0.5)
        alerts = watchdog.run_once()

        drops = [a for a in alerts if a.kind == "CONFIDENCE_DROP"]
        assert len(drops) == 1

    def test_no_drop_when_disabled(self):
        storage = _make_storage()
        self._make_closed_epoch(storage, "ep-nd1", confidence=0.3)

        watchdog = WatchdogDaemon(storage=storage, min_confidence=None)
        alerts = watchdog.run_once()

        drops = [a for a in alerts if a.kind == "CONFIDENCE_DROP"]
        assert len(drops) == 0


# ---------------------------------------------------------------------------
# Alert handler
# ---------------------------------------------------------------------------

class TestAlertHandler:
    def test_handler_receives_alerts(self):
        storage = _make_storage()
        _open_epoch(storage, "ep-handler", opened_ago_ms=7200 * 1000)

        received = []
        watchdog = WatchdogDaemon(storage=storage, max_open_secs=3600)
        watchdog.add_alert_handler(received.append)
        watchdog.run_once()

        stuck = [a for a in received if a.kind == "STUCK_EPOCH"]
        assert len(stuck) == 1

    def test_handler_exception_does_not_crash_watchdog(self):
        storage = _make_storage()
        _open_epoch(storage, "ep-exc", opened_ago_ms=7200 * 1000)

        def bad_handler(alert):
            raise RuntimeError("handler crash")

        watchdog = WatchdogDaemon(storage=storage, max_open_secs=3600)
        watchdog.add_alert_handler(bad_handler)
        # Should not raise
        watchdog.run_once()


# ---------------------------------------------------------------------------
# Thread lifecycle
# ---------------------------------------------------------------------------

class TestWatchdogThreadLifecycle:
    def test_start_and_stop(self):
        storage = _make_storage()
        watchdog = WatchdogDaemon(storage=storage, interval_secs=0.05)
        watchdog.start()
        assert watchdog._thread is not None
        assert watchdog._thread.is_alive()
        watchdog.stop(timeout=2.0)
        assert watchdog._thread is None

    def test_start_idempotent(self):
        storage = _make_storage()
        watchdog = WatchdogDaemon(storage=storage, interval_secs=100)
        watchdog.start()
        thread_id = id(watchdog._thread)
        watchdog.start()  # Second start should be a no-op
        assert id(watchdog._thread) == thread_id
        watchdog.stop()
