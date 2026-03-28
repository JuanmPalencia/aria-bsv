"""Tests for aria.canary — CanaryTracker."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest

from aria.canary import (
    CanaryHealth,
    CanaryStatus,
    CanaryTracker,
    MetricComparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(confidence: float | None = None, latency_ms: float | None = None):
    r = MagicMock()
    r.confidence = confidence
    r.latency_ms = latency_ms
    return r


def _storage(stable_records: list, canary_records: list, stable_id="s", canary_id="c"):
    storage = MagicMock()

    def side_effect(eid):
        if eid == stable_id:
            return stable_records
        if eid == canary_id:
            return canary_records
        return []

    storage.list_records_by_epoch.side_effect = side_effect
    return storage


def _tracker(stable: list, canary: list, **kwargs):
    storage = _storage(stable, canary)
    return CanaryTracker(
        storage,
        stable_epoch="s",
        canary_epoch="c",
        **kwargs,
    )


def _confidence_records(values: list[float]):
    return [_record(confidence=v) for v in values]


def _latency_records(values: list[float]):
    return [_record(confidence=0.8, latency_ms=v) for v in values]


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_empty_stable(self):
        t = _tracker(stable=[], canary=_confidence_records([0.8] * 20))
        s = t.status()
        assert s.health == CanaryHealth.UNKNOWN

    def test_empty_canary(self):
        t = _tracker(stable=_confidence_records([0.8] * 20), canary=[])
        s = t.status()
        assert s.health == CanaryHealth.UNKNOWN

    def test_below_min_samples(self):
        t = _tracker(
            stable=_confidence_records([0.8] * 5),
            canary=_confidence_records([0.8] * 5),
            min_samples=10,
        )
        s = t.status()
        assert s.health == CanaryHealth.UNKNOWN
        assert "5" in s.recommendation


# ---------------------------------------------------------------------------
# Healthy canary (confidence)
# ---------------------------------------------------------------------------

class TestHealthyCanary:
    def test_identical_confidence(self):
        vals = [0.8] * 20
        t = _tracker(
            stable=_confidence_records(vals),
            canary=_confidence_records(vals),
            metrics=["confidence"],
        )
        s = t.status()
        assert s.health == CanaryHealth.HEALTHY

    def test_canary_slightly_better_confidence(self):
        stable  = _confidence_records([0.80] * 20)
        canary  = _confidence_records([0.82] * 20)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        assert s.health == CanaryHealth.HEALTHY

    def test_should_promote_healthy(self):
        vals = [0.8] * 20
        t = _tracker(stable=_confidence_records(vals), canary=_confidence_records(vals), metrics=["confidence"])
        assert t.should_promote() is True

    def test_should_not_rollback_healthy(self):
        vals = [0.8] * 20
        t = _tracker(stable=_confidence_records(vals), canary=_confidence_records(vals), metrics=["confidence"])
        assert t.should_rollback() is False


# ---------------------------------------------------------------------------
# Degraded canary
# ---------------------------------------------------------------------------

class TestDegradedCanary:
    def test_degraded_confidence(self):
        # Stable 0.80, canary 0.72 → 10% drop, warn=5%, critical=15%
        stable = _confidence_records([0.80] * 20)
        canary = _confidence_records([0.72] * 20)
        t = _tracker(
            stable=stable,
            canary=canary,
            metrics=["confidence"],
            warn_threshold=5.0,
            critical_threshold=15.0,
        )
        s = t.status()
        assert s.health == CanaryHealth.DEGRADED


# ---------------------------------------------------------------------------
# Critical canary
# ---------------------------------------------------------------------------

class TestCriticalCanary:
    def test_critical_confidence(self):
        # Stable 0.90, canary 0.60 → 33% drop
        stable = _confidence_records([0.90] * 20)
        canary = _confidence_records([0.60] * 20)
        t = _tracker(
            stable=stable,
            canary=canary,
            metrics=["confidence"],
            warn_threshold=5.0,
            critical_threshold=15.0,
        )
        s = t.status()
        assert s.health == CanaryHealth.CRITICAL

    def test_should_rollback_critical(self):
        stable = _confidence_records([0.90] * 20)
        canary = _confidence_records([0.50] * 20)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        assert t.should_rollback() is True

    def test_should_not_promote_critical(self):
        stable = _confidence_records([0.90] * 20)
        canary = _confidence_records([0.50] * 20)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        assert t.should_promote() is False


# ---------------------------------------------------------------------------
# Latency metric (lower is better)
# ---------------------------------------------------------------------------

class TestLatencyMetric:
    def test_canary_faster(self):
        stable = _latency_records([500.0] * 20)
        canary = _latency_records([450.0] * 20)  # 10% faster → healthy
        t = _tracker(stable=stable, canary=canary, metrics=["latency_ms"])
        s = t.status()
        assert s.health == CanaryHealth.HEALTHY

    def test_canary_slower_critical(self):
        stable = _latency_records([200.0] * 20)
        canary = _latency_records([350.0] * 20)  # 75% slower → critical
        t = _tracker(
            stable=stable,
            canary=canary,
            metrics=["latency_ms"],
            warn_threshold=5.0,
            critical_threshold=15.0,
        )
        s = t.status()
        assert s.health == CanaryHealth.CRITICAL


# ---------------------------------------------------------------------------
# Multiple metrics
# ---------------------------------------------------------------------------

class TestMultipleMetrics:
    def test_worst_metric_wins(self):
        # confidence healthy, latency critical
        stable = [_record(confidence=0.8, latency_ms=200.0) for _ in range(20)]
        canary = [_record(confidence=0.81, latency_ms=400.0) for _ in range(20)]
        t = CanaryTracker(
            _storage(stable, canary),
            stable_epoch="s",
            canary_epoch="c",
            metrics=["confidence", "latency_ms"],
            warn_threshold=5.0,
            critical_threshold=15.0,
        )
        s = t.status()
        assert s.health == CanaryHealth.CRITICAL

    def test_comparisons_include_both_metrics(self):
        stable = [_record(confidence=0.8, latency_ms=200.0) for _ in range(20)]
        canary = [_record(confidence=0.8, latency_ms=200.0) for _ in range(20)]
        t = CanaryTracker(
            _storage(stable, canary),
            stable_epoch="s",
            canary_epoch="c",
            metrics=["confidence", "latency_ms"],
        )
        s = t.status()
        metric_names = [c.metric for c in s.comparisons]
        assert "confidence" in metric_names
        assert "latency_ms" in metric_names


# ---------------------------------------------------------------------------
# Status data
# ---------------------------------------------------------------------------

class TestStatusData:
    def test_n_stable_n_canary(self):
        stable = _confidence_records([0.8] * 15)
        canary = _confidence_records([0.8] * 12)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        assert s.n_stable == 15
        assert s.n_canary == 12

    def test_epoch_ids_preserved(self):
        stable = _confidence_records([0.8] * 15)
        canary = _confidence_records([0.8] * 15)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        assert s.stable_epoch == "s"
        assert s.canary_epoch == "c"

    def test_str_representation(self):
        stable = _confidence_records([0.8] * 15)
        canary = _confidence_records([0.85] * 15)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        text = str(s)
        assert "CanaryStatus" in text
        assert "confidence" in text

    def test_delta_positive_when_canary_better_confidence(self):
        stable = _confidence_records([0.70] * 20)
        canary = _confidence_records([0.80] * 20)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        conf_cmp = next(c for c in s.comparisons if c.metric == "confidence")
        assert conf_cmp.delta > 0

    def test_recommendation_contains_action(self):
        stable = _confidence_records([0.8] * 15)
        canary = _confidence_records([0.8] * 15)
        t = _tracker(stable=stable, canary=canary, metrics=["confidence"])
        s = t.status()
        assert s.recommendation != ""


# ---------------------------------------------------------------------------
# MetricComparison
# ---------------------------------------------------------------------------

class TestMetricComparison:
    def test_fields(self):
        c = MetricComparison(
            metric="confidence",
            mean_stable=0.8,
            mean_canary=0.9,
            delta=0.1,
            delta_pct=12.5,
            health=CanaryHealth.HEALTHY,
        )
        assert c.delta_pct == 12.5
        assert c.health == CanaryHealth.HEALTHY
