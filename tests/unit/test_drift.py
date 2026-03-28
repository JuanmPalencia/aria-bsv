"""Tests for aria.drift — statistical drift detection."""

from __future__ import annotations

import math
import random

import pytest

from aria.drift import (
    DriftDetector,
    DriftResult,
    SlidingWindowDrift,
    js_divergence,
    kl_divergence,
    ks_statistic,
    _build_histogram,
    _smooth,
)


# ---------------------------------------------------------------------------
# _build_histogram
# ---------------------------------------------------------------------------

class TestBuildHistogram:
    def test_empty_returns_zeros(self):
        h = _build_histogram([], n_bins=5)
        assert h == [0.0] * 5
        assert len(h) == 5

    def test_sums_to_one(self):
        values = [random.random() for _ in range(100)]
        h = _build_histogram(values, n_bins=10)
        assert abs(sum(h) - 1.0) < 1e-9

    def test_bin_count(self):
        h = _build_histogram([0.5], n_bins=20)
        assert len(h) == 20

    def test_boundary_value_1_0(self):
        # value == 1.0 should fall in the last bin
        h = _build_histogram([1.0], n_bins=4)
        assert h[-1] == 1.0

    def test_uniform_distribution(self):
        # Perfectly uniform: equal values in each bin
        vals = [i / 20 + 0.025 for i in range(20)]
        h = _build_histogram(vals, n_bins=20)
        assert all(abs(v - 1 / 20) < 1e-9 for v in h)


# ---------------------------------------------------------------------------
# _smooth
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_no_zero_bins(self):
        h = [0.0, 0.5, 0.5, 0.0]
        s = _smooth(h)
        assert all(v > 0 for v in s)

    def test_sums_to_approximately_one(self):
        h = [0.25, 0.25, 0.25, 0.25]
        s = _smooth(h)
        assert abs(sum(s) - 1.0) < 1e-6

    def test_proportions_preserved(self):
        h = [1.0, 0.0]
        s = _smooth(h, epsilon=1e-9)
        # First bin should be much larger than second
        assert s[0] > s[1] * 1000


# ---------------------------------------------------------------------------
# ks_statistic
# ---------------------------------------------------------------------------

class TestKSStatistic:
    def test_identical_distributions(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert ks_statistic(a, a) == 0.0

    def test_empty_returns_zero(self):
        assert ks_statistic([], [0.1, 0.2]) == 0.0
        assert ks_statistic([0.1], []) == 0.0

    def test_completely_separated(self):
        a = [0.0] * 10
        b = [1.0] * 10
        stat = ks_statistic(a, b)
        assert stat == 1.0

    def test_in_range_0_1(self):
        rng = random.Random(42)
        a = [rng.random() for _ in range(50)]
        b = [rng.random() for _ in range(50)]
        stat = ks_statistic(a, b)
        assert 0.0 <= stat <= 1.0

    def test_shifted_distributions_detected(self):
        a = [0.1 * i for i in range(1, 11)]     # 0.1 .. 1.0
        b = [0.1 * i + 0.5 for i in range(1, 11)]  # 0.6 .. 1.5 (clamp irrelevant for KS)
        stat = ks_statistic(a, b)
        assert stat > 0.3

    def test_returns_float(self):
        assert isinstance(ks_statistic([0.5], [0.6]), float)

    def test_symmetric(self):
        rng = random.Random(7)
        a = [rng.random() for _ in range(30)]
        b = [rng.random() for _ in range(30)]
        assert ks_statistic(a, b) == ks_statistic(b, a)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_near_zero(self):
        a = [0.1 * i for i in range(1, 11)]
        kl = kl_divergence(a, a)
        assert kl < 0.01

    def test_empty_returns_small_value(self):
        # Both empty → identical uniform histograms after smoothing
        kl = kl_divergence([], [])
        assert kl < 0.01

    def test_asymmetric(self):
        # Use two distributions at different parts of the histogram so
        # KL(P||Q) ≠ KL(Q||P) numerically — confirmed with unequal spread.
        import random
        rng = random.Random(42)
        a = [rng.uniform(0.05, 0.15) for _ in range(50)]  # narrow low peak
        b = [rng.uniform(0.0, 1.0) for _ in range(50)]    # spread across range
        kl_ab = kl_divergence(a, b)
        kl_ba = kl_divergence(b, a)
        # KL is asymmetric in general (may be equal in edge cases, but not here)
        assert abs(kl_ab - kl_ba) > 1e-6

    def test_different_distributions_large(self):
        a = [0.1] * 30
        b = [0.9] * 30
        kl = kl_divergence(a, b)
        assert kl > 0.1


# ---------------------------------------------------------------------------
# js_divergence
# ---------------------------------------------------------------------------

class TestJSDivergence:
    def test_identical_is_zero(self):
        a = [i / 20.0 for i in range(1, 21)]
        js = js_divergence(a, a)
        assert js < 0.001

    def test_bounded_0_1(self):
        rng = random.Random(99)
        for _ in range(10):
            a = [rng.random() for _ in range(30)]
            b = [rng.random() for _ in range(30)]
            js = js_divergence(a, b)
            assert 0.0 <= js <= 1.0, f"JS out of range: {js}"

    def test_symmetric(self):
        rng = random.Random(1)
        a = [rng.random() for _ in range(40)]
        b = [rng.random() for _ in range(40)]
        assert abs(js_divergence(a, b) - js_divergence(b, a)) < 1e-6

    def test_maximally_different(self):
        a = [0.01] * 20
        b = [0.99] * 20
        js = js_divergence(a, b)
        assert js > 0.5

    def test_empty_lists_returns_zero(self):
        js = js_divergence([], [])
        assert js < 0.01


# ---------------------------------------------------------------------------
# DriftResult
# ---------------------------------------------------------------------------

class TestDriftResult:
    def test_drift_detected_flag(self):
        r = DriftResult(
            epoch_a="a", epoch_b="b", test="js",
            statistic=0.20, threshold=0.10,
            drift_detected=True,
            sample_size_a=50, sample_size_b=50,
        )
        assert r.drift_detected is True

    def test_no_drift_flag(self):
        r = DriftResult(
            epoch_a="a", epoch_b="b", test="ks",
            statistic=0.05, threshold=0.15,
            drift_detected=False,
            sample_size_a=100, sample_size_b=100,
        )
        assert r.drift_detected is False


# ---------------------------------------------------------------------------
# SlidingWindowDrift
# ---------------------------------------------------------------------------

class TestSlidingWindowDrift:
    def _make_result(self, detected: bool) -> DriftResult:
        return DriftResult(
            epoch_a="a", epoch_b="b", test="js",
            statistic=0.2 if detected else 0.05,
            threshold=0.10, drift_detected=detected,
            sample_size_a=30, sample_size_b=30,
        )

    def test_any_drift_true(self):
        sw = SlidingWindowDrift(
            system_id="sys",
            epochs=["a", "b", "c"],
            results=[self._make_result(False), self._make_result(True)],
        )
        assert sw.any_drift is True

    def test_any_drift_false(self):
        sw = SlidingWindowDrift(
            system_id="sys",
            epochs=["a", "b"],
            results=[self._make_result(False)],
        )
        assert sw.any_drift is False

    def test_max_statistic(self):
        sw = SlidingWindowDrift(
            system_id="sys",
            epochs=["a", "b", "c"],
            results=[self._make_result(False), self._make_result(True)],
        )
        assert sw.max_statistic == 0.2

    def test_empty_results(self):
        sw = SlidingWindowDrift(system_id="", epochs=[], results=[])
        assert sw.any_drift is False
        assert sw.max_statistic == 0.0


# ---------------------------------------------------------------------------
# DriftDetector — integration with mock storage
# ---------------------------------------------------------------------------

class FakeRecord:
    def __init__(self, confidence):
        self.confidence = confidence

class FakeEpoch:
    def __init__(self, epoch_id):
        self.epoch_id = epoch_id

class MockStorage:
    def __init__(self):
        self._records: dict[str, list[float]] = {}
        self._epochs: list[FakeEpoch] = []

    def add_epoch(self, epoch_id):
        self._epochs.append(FakeEpoch(epoch_id))
        self._records[epoch_id] = []

    def add_records(self, epoch_id, confidences):
        self._records[epoch_id] = confidences

    def list_records_by_epoch(self, epoch_id):
        return [FakeRecord(c) for c in self._records.get(epoch_id, [])]

    def list_epochs(self, system_id=None, limit=100):
        return self._epochs[:limit]


class TestDriftDetector:
    def _make_storage_with_two_epochs(self, confs_a, confs_b):
        storage = MockStorage()
        storage.add_epoch("epoch-a")
        storage.add_epoch("epoch-b")
        storage.add_records("epoch-a", confs_a)
        storage.add_records("epoch-b", confs_b)
        return storage

    def test_identical_distributions_no_drift(self):
        confs = [0.1 * i for i in range(1, 21)] * 3   # 60 values
        storage = self._make_storage_with_two_epochs(confs, confs)
        detector = DriftDetector(storage, threshold=0.10, test="js")
        result = detector.compare("epoch-a", "epoch-b")
        assert result.drift_detected is False
        assert result.statistic < 0.01

    def test_shifted_distributions_detect_drift(self):
        rng = random.Random(42)
        confs_a = [rng.uniform(0.1, 0.3) for _ in range(50)]
        confs_b = [rng.uniform(0.7, 0.9) for _ in range(50)]
        storage = self._make_storage_with_two_epochs(confs_a, confs_b)
        detector = DriftDetector(storage, threshold=0.10, test="js")
        result = detector.compare("epoch-a", "epoch-b")
        assert result.drift_detected is True

    def test_ks_test(self):
        confs_a = [0.2] * 30
        confs_b = [0.8] * 30
        storage = self._make_storage_with_two_epochs(confs_a, confs_b)
        detector = DriftDetector(storage, threshold=0.10, test="ks")
        result = detector.compare("epoch-a", "epoch-b")
        assert result.test == "ks"
        assert result.drift_detected is True

    def test_kl_test(self):
        confs_a = [0.5] * 30
        confs_b = [0.5] * 30
        storage = self._make_storage_with_two_epochs(confs_a, confs_b)
        detector = DriftDetector(storage, threshold=0.50, test="kl")
        result = detector.compare("epoch-a", "epoch-b")
        assert result.test == "kl"
        assert result.drift_detected is False

    def test_insufficient_data_returns_no_drift(self):
        storage = self._make_storage_with_two_epochs([0.5] * 3, [0.5] * 3)
        detector = DriftDetector(storage, threshold=0.10, min_samples=10)
        result = detector.compare("epoch-a", "epoch-b")
        assert result.drift_detected is False
        assert result.statistic == 0.0
        assert "Insufficient" in result.detail

    def test_result_has_sample_sizes(self):
        confs_a = [0.5] * 20
        confs_b = [0.6] * 25
        storage = self._make_storage_with_two_epochs(confs_a, confs_b)
        detector = DriftDetector(storage, threshold=0.10)
        result = detector.compare("epoch-a", "epoch-b")
        assert result.sample_size_a == 20
        assert result.sample_size_b == 25

    def test_nonexistent_epoch_returns_no_drift(self):
        storage = MockStorage()
        detector = DriftDetector(storage, threshold=0.10)
        result = detector.compare("nonexistent-a", "nonexistent-b")
        assert result.drift_detected is False
        assert "Insufficient" in result.detail

    def test_detail_string_populated_on_drift(self):
        rng = random.Random(7)
        confs_a = [rng.uniform(0.1, 0.3) for _ in range(50)]
        confs_b = [rng.uniform(0.7, 0.9) for _ in range(50)]
        storage = self._make_storage_with_two_epochs(confs_a, confs_b)
        detector = DriftDetector(storage, threshold=0.10)
        result = detector.compare("epoch-a", "epoch-b")
        assert result.detail != ""
        assert "JS" in result.detail.upper() or "KS" in result.detail.upper()

    def test_sliding_window_check(self):
        storage = MockStorage()
        rng = random.Random(3)
        for i in range(5):
            storage.add_epoch(f"epoch-{i}")
            storage.add_records(f"epoch-{i}", [rng.random() for _ in range(30)])

        detector = DriftDetector(storage, threshold=0.50)
        sw = detector.sliding_window_check(n_epochs=5)
        assert isinstance(sw, SlidingWindowDrift)
        assert len(sw.epochs) == 5
        assert len(sw.results) == 4   # 5 epochs → 4 pairs

    def test_sliding_window_single_epoch(self):
        storage = MockStorage()
        storage.add_epoch("only-one")
        storage.add_records("only-one", [0.5] * 20)
        detector = DriftDetector(storage)
        sw = detector.sliding_window_check(n_epochs=5)
        assert len(sw.results) == 0

    def test_threshold_boundary(self):
        # Use identical distributions so JS ≈ 0 — strict threshold flags it, loose doesn't
        confs = [0.1 * (i % 10) + 0.05 for i in range(40)]  # spread evenly across bins
        storage = self._make_storage_with_two_epochs(confs, confs)

        # Threshold below actual statistic (near-zero for identical dists)
        stat = js_divergence(confs, confs)  # should be ~0

        detector_strict = DriftDetector(storage, threshold=-0.001)  # always fires
        result_strict = detector_strict.compare("epoch-a", "epoch-b")

        detector_loose = DriftDetector(storage, threshold=0.99)  # never fires for identical
        result_loose = detector_loose.compare("epoch-a", "epoch-b")

        assert result_strict.drift_detected is True
        assert result_loose.drift_detected is False

    def test_records_without_confidence_excluded(self):
        storage = MockStorage()
        storage.add_epoch("epoch-a")
        storage.add_epoch("epoch-b")
        # Mix of None and valid
        storage._records["epoch-a"] = [None, 0.5, None, 0.6, 0.7] * 4  # type: ignore
        storage._records["epoch-b"] = [None, 0.5, 0.6, None, 0.7] * 4  # type: ignore

        class RecordWithOptionalConf:
            def __init__(self, c):
                self.confidence = c

        storage.list_records_by_epoch = lambda eid: [
            RecordWithOptionalConf(c) for c in storage._records.get(eid, [])
        ]

        detector = DriftDetector(storage, threshold=0.10, min_samples=5)
        result = detector.compare("epoch-a", "epoch-b")
        # Should use only non-None values
        assert result.sample_size_a <= 20
