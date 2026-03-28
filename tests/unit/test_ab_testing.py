"""Tests for aria.ab_testing — A/B statistical testing framework."""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

import pytest

from aria.ab_testing import (
    ABMetric,
    ABTestReport,
    ABTestRunner,
    ABVerdict,
    BatchABReport,
    EffectSize,
    _cohens_d,
    _effect_size_label,
    _mann_whitney_u,
    _mean,
    _std,
    _variance,
    _welch_t,
)


# ---------------------------------------------------------------------------
# Pure stats helpers
# ---------------------------------------------------------------------------

class TestMean:
    def test_basic(self):
        assert _mean([1, 2, 3, 4, 5]) == pytest.approx(3.0)

    def test_single(self):
        assert _mean([7.0]) == 7.0

    def test_empty(self):
        assert _mean([]) == 0.0


class TestVariance:
    def test_identical_values(self):
        assert _variance([5, 5, 5, 5]) == pytest.approx(0.0)

    def test_known_variance(self):
        # Variance of [2, 4, 4, 4, 5, 5, 7, 9] == 4.571...
        v = _variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert v == pytest.approx(4.571, abs=0.01)

    def test_single(self):
        assert _variance([3.0]) == 0.0

    def test_empty(self):
        assert _variance([]) == 0.0


class TestStd:
    def test_known(self):
        assert _std([2, 4, 6]) == pytest.approx(2.0, abs=0.01)

    def test_empty(self):
        assert _std([]) == 0.0


class TestCohensD:
    def test_identical(self):
        a = [0.8] * 20
        b = [0.8] * 20
        assert _cohens_d(a, b) == pytest.approx(0.0)

    def test_clear_difference(self):
        rng = random.Random(99)
        a = [rng.gauss(0.5, 0.1) for _ in range(20)]
        b = [rng.gauss(0.9, 0.1) for _ in range(20)]
        d = _cohens_d(a, b)
        assert abs(d) > 1.0

    def test_sign(self):
        # a > b → d > 0
        rng = random.Random(42)
        a = [rng.gauss(0.9, 0.05) for _ in range(20)]
        b = [rng.gauss(0.5, 0.05) for _ in range(20)]
        assert _cohens_d(a, b) > 0

    def test_empty(self):
        assert _cohens_d([], [1, 2, 3]) == 0.0
        assert _cohens_d([1, 2], []) == 0.0

    def test_zero_pooled_std(self):
        a = [1.0] * 20
        b = [1.0] * 20
        assert _cohens_d(a, b) == 0.0


class TestWelchT:
    def test_identical_groups(self):
        a = [0.7] * 15
        b = [0.7] * 15
        t, p = _welch_t(a, b)
        assert t == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_significant_difference(self):
        rng = random.Random(42)
        a = [rng.gauss(0.5, 0.05) for _ in range(50)]
        b = [rng.gauss(0.9, 0.05) for _ in range(50)]
        t, p = _welch_t(a, b)
        assert p < 0.05

    def test_empty(self):
        t, p = _welch_t([], [1.0, 2.0])
        assert t == 0.0
        assert p == 1.0

    def test_zero_se(self):
        t, p = _welch_t([1.0] * 10, [1.0] * 10)
        assert p == 1.0


class TestMannWhitneyU:
    def test_identical(self):
        a = [0.6] * 15
        b = [0.6] * 15
        u, p = _mann_whitney_u(a, b)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_separated(self):
        a = list(range(1, 21))
        b = list(range(21, 41))
        u, p = _mann_whitney_u(a, b)
        assert p < 0.05

    def test_empty(self):
        u, p = _mann_whitney_u([], [1, 2])
        assert p == 1.0


class TestEffectSizeLabel:
    def test_negligible(self):
        assert _effect_size_label(0.1) == EffectSize.NEGLIGIBLE
        assert _effect_size_label(-0.1) == EffectSize.NEGLIGIBLE

    def test_small(self):
        assert _effect_size_label(0.3) == EffectSize.SMALL
        assert _effect_size_label(-0.35) == EffectSize.SMALL

    def test_medium(self):
        assert _effect_size_label(0.65) == EffectSize.MEDIUM

    def test_large(self):
        assert _effect_size_label(1.2) == EffectSize.LARGE
        assert _effect_size_label(-0.9) == EffectSize.LARGE

    def test_boundary_0_2(self):
        assert _effect_size_label(0.2) == EffectSize.SMALL

    def test_boundary_0_5(self):
        assert _effect_size_label(0.5) == EffectSize.MEDIUM

    def test_boundary_0_8(self):
        assert _effect_size_label(0.8) == EffectSize.LARGE


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_ab_metric_values(self):
        assert ABMetric.CONFIDENCE.value == "confidence"
        assert ABMetric.LATENCY_MS.value == "latency_ms"

    def test_ab_verdict_values(self):
        assert ABVerdict.A_WINS.value == "A_WINS"
        assert ABVerdict.B_WINS.value == "B_WINS"
        assert ABVerdict.NO_DIFF.value == "NO_DIFF"
        assert ABVerdict.INSUFFICIENT.value == "INSUFFICIENT_DATA"

    def test_effect_size_values(self):
        assert EffectSize.NEGLIGIBLE.value == "NEGLIGIBLE"
        assert EffectSize.LARGE.value == "LARGE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(metric: str, value: float):
    r = MagicMock()
    setattr(r, metric, value)
    return r


def _make_storage(epoch_records: dict[str, list]):
    """Returns a storage mock with list_records_by_epoch wired."""
    storage = MagicMock()
    storage.list_records_by_epoch.side_effect = lambda eid: epoch_records.get(eid, [])
    return storage


def _epoch_records(values: list[float], metric: str = "confidence"):
    return [_make_record(metric, v) for v in values]


# ---------------------------------------------------------------------------
# ABTestRunner.compare
# ---------------------------------------------------------------------------

class TestABTestRunnerCompare:
    def _runner(self, epoch_data: dict[str, list[float]], metric: str = "confidence"):
        storage = _make_storage({
            k: _epoch_records(v, metric) for k, v in epoch_data.items()
        })
        return ABTestRunner(storage, alpha=0.05, min_samples=10)

    def test_insufficient_a(self):
        runner = self._runner({"ep-a": [0.5] * 5, "ep-b": [0.8] * 20})
        r = runner.compare("ep-a", "ep-b")
        assert r.verdict == ABVerdict.INSUFFICIENT

    def test_insufficient_b(self):
        runner = self._runner({"ep-a": [0.5] * 20, "ep-b": [0.8] * 3})
        r = runner.compare("ep-a", "ep-b")
        assert r.verdict == ABVerdict.INSUFFICIENT

    def test_b_wins_confidence(self):
        rng = random.Random(1)
        vals_a = [rng.gauss(0.5, 0.02) for _ in range(60)]
        vals_b = [rng.gauss(0.85, 0.02) for _ in range(60)]
        runner = self._runner({"ep-a": vals_a, "ep-b": vals_b})
        r = runner.compare("ep-a", "ep-b")
        assert r.verdict == ABVerdict.B_WINS

    def test_a_wins_confidence(self):
        rng = random.Random(2)
        vals_a = [rng.gauss(0.9, 0.02) for _ in range(60)]
        vals_b = [rng.gauss(0.5, 0.02) for _ in range(60)]
        runner = self._runner({"ep-a": vals_a, "ep-b": vals_b})
        r = runner.compare("ep-a", "ep-b")
        assert r.verdict == ABVerdict.A_WINS

    def test_b_wins_latency(self):
        rng = random.Random(3)
        vals_a = [rng.gauss(500, 10) for _ in range(60)]
        vals_b = [rng.gauss(200, 10) for _ in range(60)]  # lower = better
        storage = _make_storage({
            "ep-a": _epoch_records(vals_a, "latency_ms"),
            "ep-b": _epoch_records(vals_b, "latency_ms"),
        })
        runner = ABTestRunner(storage, alpha=0.05, min_samples=10)
        r = runner.compare("ep-a", "ep-b", metric="latency_ms")
        assert r.verdict == ABVerdict.B_WINS

    def test_no_diff_identical(self):
        vals = [0.75] * 40
        runner = self._runner({"ep-a": vals, "ep-b": vals})
        r = runner.compare("ep-a", "ep-b")
        assert r.verdict == ABVerdict.NO_DIFF

    def test_report_fields(self):
        rng = random.Random(7)
        vals_a = [rng.gauss(0.5, 0.05) for _ in range(30)]
        vals_b = [rng.gauss(0.8, 0.05) for _ in range(30)]
        runner = self._runner({"ep-a": vals_a, "ep-b": vals_b})
        r = runner.compare("ep-a", "ep-b")
        assert r.n_a == 30
        assert r.n_b == 30
        assert 0 <= r.mean_a <= 1
        assert 0 <= r.mean_b <= 1
        assert r.std_a >= 0
        assert r.std_b >= 0
        assert isinstance(r.effect_size, EffectSize)
        assert isinstance(r.verdict, ABVerdict)

    def test_significant_property(self):
        rng = random.Random(8)
        vals_a = [rng.gauss(0.5, 0.02) for _ in range(80)]
        vals_b = [rng.gauss(0.9, 0.02) for _ in range(80)]
        runner = self._runner({"ep-a": vals_a, "ep-b": vals_b})
        r = runner.compare("ep-a", "ep-b")
        assert r.significant is True

    def test_insufficient_detail(self):
        runner = self._runner({"ep-a": [0.5] * 3, "ep-b": [0.8] * 30})
        r = runner.compare("ep-a", "ep-b")
        assert "3" in r.detail

    def test_str_representation(self):
        rng = random.Random(9)
        vals_a = [rng.gauss(0.5, 0.05) for _ in range(30)]
        vals_b = [rng.gauss(0.8, 0.05) for _ in range(30)]
        runner = self._runner({"ep-a": vals_a, "ep-b": vals_b})
        r = runner.compare("ep-a", "ep-b")
        s = str(r)
        assert "ABTestReport" in s
        assert "confidence" in s
        assert "mean=" in s


# ---------------------------------------------------------------------------
# ABTestRunner.batch_compare
# ---------------------------------------------------------------------------

class TestABTestRunnerBatchCompare:
    def test_batch_returns_all_reports(self):
        rng = random.Random(11)
        storage = _make_storage({
            "a1": _epoch_records([rng.gauss(0.5, 0.05) for _ in range(20)]),
            "a2": _epoch_records([rng.gauss(0.5, 0.05) for _ in range(20)]),
            "b1": _epoch_records([rng.gauss(0.8, 0.05) for _ in range(20)]),
            "b2": _epoch_records([rng.gauss(0.8, 0.05) for _ in range(20)]),
        })
        runner = ABTestRunner(storage)
        batch = runner.batch_compare(["a1", "a2"], ["b1", "b2"])
        assert len(batch.reports) == 2

    def test_b_wins_rate_all_wins(self):
        rng = random.Random(12)
        epoch_data = {}
        for i in range(3):
            epoch_data[f"a{i}"] = _epoch_records([rng.gauss(0.4, 0.02) for _ in range(40)])
            epoch_data[f"b{i}"] = _epoch_records([rng.gauss(0.9, 0.02) for _ in range(40)])
        storage = _make_storage(epoch_data)
        runner = ABTestRunner(storage)
        batch = runner.batch_compare([f"a{i}" for i in range(3)], [f"b{i}" for i in range(3)])
        assert batch.b_wins_rate == pytest.approx(1.0)

    def test_recommendation_deploy_b(self):
        rng = random.Random(13)
        epoch_data = {}
        for i in range(5):
            epoch_data[f"a{i}"] = _epoch_records([rng.gauss(0.4, 0.02) for _ in range(40)])
            epoch_data[f"b{i}"] = _epoch_records([rng.gauss(0.9, 0.02) for _ in range(40)])
        storage = _make_storage(epoch_data)
        runner = ABTestRunner(storage)
        batch = runner.batch_compare([f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)])
        assert batch.recommendation == "DEPLOY_B"

    def test_recommendation_keep_a(self):
        rng = random.Random(14)
        epoch_data = {}
        for i in range(5):
            epoch_data[f"a{i}"] = _epoch_records([rng.gauss(0.9, 0.02) for _ in range(40)])
            epoch_data[f"b{i}"] = _epoch_records([rng.gauss(0.4, 0.02) for _ in range(40)])
        storage = _make_storage(epoch_data)
        runner = ABTestRunner(storage)
        batch = runner.batch_compare([f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)])
        assert batch.recommendation == "KEEP_A"

    def test_recommendation_empty_returns_keep_a(self):
        batch = BatchABReport(metric="confidence", reports=[])
        assert batch.recommendation == "KEEP_A"
        assert batch.b_wins_rate == 0.0

    def test_metric_preserved(self):
        storage = _make_storage({"a": [], "b": []})
        runner = ABTestRunner(storage)
        batch = runner.batch_compare(["a"], ["b"], metric="latency_ms")
        assert batch.metric == "latency_ms"


# ---------------------------------------------------------------------------
# ABTestReport dataclass
# ---------------------------------------------------------------------------

class TestABTestReport:
    def _report(self, **kwargs):
        defaults = dict(
            epoch_a="a", epoch_b="b", metric="confidence",
            verdict=ABVerdict.B_WINS,
            mean_a=0.5, mean_b=0.8, std_a=0.05, std_b=0.05,
            n_a=30, n_b=30,
            cohens_d=2.0, effect_size=EffectSize.LARGE,
            t_statistic=-8.0, p_value=0.001, mw_p_value=0.002,
            alpha=0.05,
        )
        defaults.update(kwargs)
        return ABTestReport(**defaults)

    def test_significant_true(self):
        r = self._report(p_value=0.01, mw_p_value=0.01)
        assert r.significant is True

    def test_significant_false(self):
        r = self._report(p_value=0.5, mw_p_value=0.5)
        assert r.significant is False

    def test_significant_one_side(self):
        r = self._report(p_value=0.5, mw_p_value=0.01)
        assert r.significant is True

    def test_str_contains_verdict(self):
        r = self._report()
        assert "B_WINS" in str(r)

    def test_str_shows_delta(self):
        r = self._report(mean_a=0.5, mean_b=0.8)
        assert "+" in str(r)  # positive delta

    def test_str_negative_delta(self):
        r = self._report(mean_a=0.8, mean_b=0.5, verdict=ABVerdict.A_WINS)
        s = str(r)
        assert "ABTestReport" in s
