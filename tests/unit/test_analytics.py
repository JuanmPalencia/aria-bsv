"""Tests for CrossEpochAnalytics."""

from __future__ import annotations

import time

import pytest

from aria.analytics import (
    CrossEpochAnalytics,
    LatencyStats,
    ConfidenceStats,
    ModelUsage,
    EpochHealth,
    DriftReport,
    _percentile,
    _compute_latency_stats,
    _compute_confidence_stats,
)
from aria.storage.sqlite import SQLiteStorage
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> SQLiteStorage:
    return SQLiteStorage("sqlite://")


def _make_record(
    epoch_id: str,
    model_id: str = "m-1",
    latency_ms: int = 100,
    confidence: float | None = 0.9,
    seq: int = 1,
) -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id=model_id,
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=seq,
        confidence=confidence,
        latency_ms=latency_ms,
        metadata={},
    )


def _seed_epoch(storage: SQLiteStorage, epoch_id: str, records: list[AuditRecord], close: bool = True) -> None:
    now = int(time.time() * 1000)
    storage.save_epoch_open(epoch_id, "sys", "tx-open", {"m-1": "sha256:" + "c" * 64}, "sh", now)
    for rec in records:
        storage.save_record(rec)
    if close:
        storage.save_epoch_close(epoch_id, "tx-close", "sha256:" + "0" * 64, len(records), now + 5000)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

class TestPercentileHelper:
    def test_p50_of_even_list(self):
        assert _percentile([10, 20, 30, 40], 50) == 25.0

    def test_p100_returns_max(self):
        assert _percentile([1, 2, 3, 4, 5], 100) == 5.0

    def test_p0_returns_min(self):
        assert _percentile([1, 2, 3], 0) == 1.0

    def test_single_element(self):
        assert _percentile([42], 95) == 42.0

    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_basic_stats(self):
        latencies = [100, 200, 300, 400, 500]
        stats = _compute_latency_stats(latencies)
        assert stats.count == 5
        assert stats.mean_ms == 300.0
        assert stats.min_ms == 100
        assert stats.max_ms == 500
        assert stats.p50_ms == 300.0

    def test_empty_returns_zero_stats(self):
        storage = _make_storage()
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.latency_stats([])
        assert stats.count == 0
        assert stats.mean_ms == 0.0

    def test_latency_stats_across_epochs(self):
        storage = _make_storage()
        _seed_epoch(storage, "ep-1", [_make_record("ep-1", latency_ms=100)])
        _seed_epoch(storage, "ep-2", [_make_record("ep-2", latency_ms=200)])
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.latency_stats(["ep-1", "ep-2"])
        assert stats.count == 2
        assert stats.mean_ms == 150.0

    def test_p95_latency(self):
        storage = _make_storage()
        recs = [_make_record("ep-p95", latency_ms=i * 10, seq=i) for i in range(1, 101)]
        _seed_epoch(storage, "ep-p95", recs)
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.latency_stats(["ep-p95"])
        assert stats.p95_ms >= 940  # Should be around 950 ms


# ---------------------------------------------------------------------------
# Confidence stats
# ---------------------------------------------------------------------------

class TestConfidenceStats:
    def test_basic_confidence(self):
        storage = _make_storage()
        recs = [_make_record("ep-conf", confidence=c, seq=i)
                for i, c in enumerate([0.7, 0.8, 0.9, 1.0], 1)]
        _seed_epoch(storage, "ep-conf", recs)
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.confidence_stats(["ep-conf"])
        assert stats.count == 4
        assert abs(stats.mean - 0.85) < 0.01

    def test_records_without_confidence_excluded(self):
        storage = _make_storage()
        recs = [
            _make_record("ep-nc", confidence=None, seq=1),
            _make_record("ep-nc", confidence=0.8, seq=2),
        ]
        _seed_epoch(storage, "ep-nc", recs)
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.confidence_stats(["ep-nc"])
        assert stats.count == 1  # Only the record with confidence

    def test_empty_confidence_returns_empty(self):
        storage = _make_storage()
        analytics = CrossEpochAnalytics(storage)
        stats = analytics.confidence_stats([])
        assert stats.count == 0

    def test_histogram_buckets(self):
        stats = _compute_confidence_stats([0.05, 0.15, 0.55, 0.95])
        assert "0.0-0.1" in stats.histogram
        assert "0.9-1.0" in stats.histogram


# ---------------------------------------------------------------------------
# Model usage
# ---------------------------------------------------------------------------

class TestModelUsage:
    def test_groups_by_model(self):
        storage = _make_storage()
        recs = [
            _make_record("ep-mu", model_id="m-A", latency_ms=100, seq=1),
            _make_record("ep-mu", model_id="m-A", latency_ms=200, seq=2),
            _make_record("ep-mu", model_id="m-B", latency_ms=300, seq=3),
        ]
        _seed_epoch(storage, "ep-mu", recs)
        analytics = CrossEpochAnalytics(storage)
        usage = analytics.model_usage(["ep-mu"])
        usage_by_model = {u.model_id: u for u in usage}
        assert usage_by_model["m-A"].record_count == 2
        assert usage_by_model["m-A"].mean_latency_ms == 150.0
        assert usage_by_model["m-B"].record_count == 1

    def test_empty_epochs_returns_empty(self):
        storage = _make_storage()
        analytics = CrossEpochAnalytics(storage)
        assert analytics.model_usage([]) == []


# ---------------------------------------------------------------------------
# Epoch health
# ---------------------------------------------------------------------------

class TestEpochHealth:
    def test_healthy_closed_epoch(self):
        storage = _make_storage()
        recs = [_make_record("ep-h1")]
        _seed_epoch(storage, "ep-h1", recs, close=True)
        analytics = CrossEpochAnalytics(storage)
        health = analytics.epoch_health("ep-h1")
        assert health.is_closed
        assert health.merkle_root_present
        assert health.record_count == 1
        assert health.healthy

    def test_open_epoch_not_healthy(self):
        storage = _make_storage()
        recs = [_make_record("ep-open")]
        _seed_epoch(storage, "ep-open", recs, close=False)
        analytics = CrossEpochAnalytics(storage)
        health = analytics.epoch_health("ep-open")
        assert not health.is_closed
        assert not health.healthy
        assert any("not yet closed" in w for w in health.warnings)

    def test_missing_epoch_returns_warning(self):
        storage = _make_storage()
        analytics = CrossEpochAnalytics(storage)
        health = analytics.epoch_health("nonexistent-id")
        assert not health.healthy
        assert any("not found" in w for w in health.warnings)

    def test_record_count_mismatch_flagged(self):
        storage = _make_storage()
        # Manually create epoch with wrong records_count
        now = int(time.time() * 1000)
        storage.save_epoch_open("ep-mis", "sys", "tx-open", {}, "sh", now)
        storage.save_record(_make_record("ep-mis", seq=1))
        storage.save_record(_make_record("ep-mis", seq=2))
        # Close with wrong count (claims 1 but we stored 2)
        storage.save_epoch_close("ep-mis", "tx-close", "root", 1, now + 5000)
        analytics = CrossEpochAnalytics(storage)
        health = analytics.epoch_health("ep-mis")
        assert any("mismatch" in w for w in health.warnings)


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

class TestDriftReport:
    def test_confidence_delta_computed(self):
        storage = _make_storage()
        _seed_epoch(storage, "ep-d1", [_make_record("ep-d1", confidence=0.9)])
        _seed_epoch(storage, "ep-d2", [_make_record("ep-d2", confidence=0.7)])
        analytics = CrossEpochAnalytics(storage)
        report = analytics.drift_report("ep-d1", "ep-d2")
        assert report.confidence_delta is not None
        assert abs(report.confidence_delta - (-0.2)) < 1e-6

    def test_latency_delta_computed(self):
        storage = _make_storage()
        _seed_epoch(storage, "ep-ld1", [_make_record("ep-ld1", latency_ms=100)])
        _seed_epoch(storage, "ep-ld2", [_make_record("ep-ld2", latency_ms=300)])
        analytics = CrossEpochAnalytics(storage)
        report = analytics.drift_report("ep-ld1", "ep-ld2")
        assert report.latency_delta_ms == 200.0

    def test_empty_epoch_handled(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open("ep-empty", "sys", "tx", {}, "sh", now)
        storage.save_epoch_close("ep-empty", "tx-c", "root", 0, now + 1)
        _seed_epoch(storage, "ep-full", [_make_record("ep-full")])
        analytics = CrossEpochAnalytics(storage)
        report = analytics.drift_report("ep-empty", "ep-full")
        assert report.record_count_a == 0
        assert report.record_count_b == 1
        assert report.confidence_delta is None  # No data for epoch A
