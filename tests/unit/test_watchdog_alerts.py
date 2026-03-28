"""Tests for WatchdogDaemon + alert channel integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aria.alerts import LogAlertChannel, MultiAlertChannel
from aria.watchdog import WatchdogDaemon


# ---------------------------------------------------------------------------
# Mock storage
# ---------------------------------------------------------------------------

class FakeEpochRow:
    def __init__(self, epoch_id, system_id, open_txid, close_txid, records_count,
                 merkle_root="", opened_at=None):
        self.epoch_id = epoch_id
        self.system_id = system_id
        self.open_txid = open_txid
        self.close_txid = close_txid
        self.records_count = records_count
        self.merkle_root = merkle_root
        self.opened_at = opened_at or int(time.time() * 1000)
        self.model_hashes = {"m": "h"}
        self.state_hash = ""
        self.closed_at = None


class FakeRecord:
    def __init__(self, epoch_id, latency_ms=100.0, confidence=0.9, model_id="m"):
        self.epoch_id = epoch_id
        self.latency_ms = latency_ms
        self.confidence = confidence
        self.model_id = model_id
        self.confidence = confidence


class MockWatchdogStorage:
    def __init__(self):
        self._epochs: list[FakeEpochRow] = []
        self._records: dict[str, list[FakeRecord]] = {}

    def add_epoch(self, epoch):
        self._epochs.append(epoch)

    def add_records(self, epoch_id, records):
        self._records[epoch_id] = records

    def list_epochs(self, system_id=None, limit=100):
        result = self._epochs
        if system_id:
            result = [e for e in result if e.system_id == system_id]
        return result[:limit]

    def list_records_by_epoch(self, epoch_id):
        return self._records.get(epoch_id, [])


# ---------------------------------------------------------------------------
# Tests: alert handler integration
# ---------------------------------------------------------------------------

class TestWatchdogAlertHandlers:
    def test_add_alert_handler_called_on_alert(self):
        """alert handlers registered on WatchdogDaemon should receive alerts."""
        storage = MockWatchdogStorage()
        # Stuck epoch: opened 2 hours ago, never closed
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="stuck-ep",
            system_id="sys",
            open_txid="tx-open",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        received = []

        def handler(alert):
            received.append(alert)

        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(handler)
        wd.run_once()

        assert len(received) >= 1
        kinds = [a.kind for a in received]
        assert any("STUCK" in k or "EPOCH" in k for k in kinds)

    def test_multiple_handlers_all_receive(self):
        storage = MockWatchdogStorage()
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-1",
            system_id="sys",
            open_txid="tx",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        received_a = []
        received_b = []

        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(lambda a: received_a.append(a))
        wd.add_alert_handler(lambda a: received_b.append(a))
        wd.run_once()

        assert len(received_a) > 0
        assert len(received_b) > 0
        assert len(received_a) == len(received_b)

    def test_log_alert_channel_integration(self, caplog):
        """LogAlertChannel can be wired directly as a watchdog handler."""
        import logging
        storage = MockWatchdogStorage()
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-log",
            system_id="sys",
            open_txid="tx",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        channel = LogAlertChannel(level="WARNING")
        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(channel.send)

        with caplog.at_level(logging.WARNING):
            wd.run_once()

        assert "[ARIA ALERT]" in caplog.text

    def test_broken_handler_does_not_crash_watchdog(self):
        """A handler that raises must not stop the watchdog from running."""
        storage = MockWatchdogStorage()
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-err",
            system_id="sys",
            open_txid="tx",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        good_received = []

        def bad_handler(alert):
            raise RuntimeError("handler crashed")

        def good_handler(alert):
            good_received.append(alert)

        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(bad_handler)
        wd.add_alert_handler(good_handler)

        # Should not raise
        wd.run_once()

        # Good handler still got alerts
        assert len(good_received) > 0

    def test_no_alerts_when_all_healthy(self):
        storage = MockWatchdogStorage()
        # Healthy closed epoch
        recent_ts = int(time.time() * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-healthy",
            system_id="sys",
            open_txid="tx-open",
            close_txid="tx-close",
            records_count=10,
            merkle_root="abc" * 20,
            opened_at=recent_ts,
        ))
        storage.add_records("ep-healthy", [
            FakeRecord("ep-healthy", latency_ms=100.0, confidence=0.9)
            for _ in range(10)
        ])

        received = []
        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(lambda a: received.append(a))
        wd.run_once()

        # Healthy epoch should not fire alerts (except possibly INFO)
        critical_alerts = [a for a in received if str(a.severity) == "CRITICAL"]
        assert len(critical_alerts) == 0


# ---------------------------------------------------------------------------
# Tests: metrics hook integration
# ---------------------------------------------------------------------------

class TestWatchdogMetricsIntegration:
    def test_metrics_alert_hook_called_on_alert(self):
        from aria.metrics import ARIAMetrics

        storage = MockWatchdogStorage()
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-metrics",
            system_id="sys",
            open_txid="tx",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        metrics = ARIAMetrics(namespace="test_wd_metrics", system_id="sys")
        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(metrics.record_alert_hook)

        # Should not raise
        wd.run_once()


# ---------------------------------------------------------------------------
# Tests: MultiAlertChannel + WatchdogDaemon
# ---------------------------------------------------------------------------

class TestMultiChannelWatchdog:
    def test_multi_channel_fanout(self):
        storage = MockWatchdogStorage()
        old_ts = int((time.time() - 7200) * 1000)
        storage.add_epoch(FakeEpochRow(
            epoch_id="ep-multi",
            system_id="sys",
            open_txid="tx",
            close_txid=None,
            records_count=0,
            opened_at=old_ts,
        ))

        received_1 = []
        received_2 = []

        from aria.alerts import AlertChannelBase

        class ListChannel(AlertChannelBase):
            def __init__(self, lst):
                self._lst = lst
            def _deliver(self, alert):
                self._lst.append(alert)

        multi = MultiAlertChannel([ListChannel(received_1), ListChannel(received_2)])

        wd = WatchdogDaemon(storage=storage, system_id="sys")
        wd.add_alert_handler(multi.send)
        wd.run_once()

        assert len(received_1) > 0
        assert len(received_2) > 0
        assert len(received_1) == len(received_2)
