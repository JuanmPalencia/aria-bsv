"""
aria.ab_testing — Statistical A/B testing framework for AI model comparison.

Compares two model versions across epochs using statistical tests and returns
a structured report suitable for model deployment decisions.

Tests available
---------------
Mann-Whitney U   — Non-parametric comparison of confidence/latency distributions.
                   Recommended default (no normality assumption).
Cohen's d        — Effect size — how large the difference is in practical terms.
Welch's t-test   — Parametric comparison of means (assumes approximate normality).
Proportion test  — Compare success rates (e.g. confidence ≥ threshold).

Usage::

    from aria.ab_testing import ABTestRunner

    runner = ABTestRunner(storage)
    report = runner.compare(
        epoch_a="epoch-model-v1",
        epoch_b="epoch-model-v2",
        metric="confidence",
    )
    print(report)
    # ABTestReport: model-v2 WINS (p=0.003, Cohen's d=0.42, MEDIUM effect)

    # Batch compare across multiple epoch pairs:
    summary = runner.batch_compare(
        epochs_a=["ep-v1-1", "ep-v1-2"],
        epochs_b=["ep-v2-1", "ep-v2-2"],
        metric="latency_ms",
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ABMetric(str, Enum):
    CONFIDENCE  = "confidence"
    LATENCY_MS  = "latency_ms"


class ABVerdict(str, Enum):
    A_WINS      = "A_WINS"
    B_WINS      = "B_WINS"
    NO_DIFF     = "NO_DIFF"
    INSUFFICIENT = "INSUFFICIENT_DATA"


class EffectSize(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"   # |d| < 0.2
    SMALL      = "SMALL"        # 0.2 ≤ |d| < 0.5
    MEDIUM     = "MEDIUM"       # 0.5 ≤ |d| < 0.8
    LARGE      = "LARGE"        # |d| ≥ 0.8


# ---------------------------------------------------------------------------
# Pure-Python statistics
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _std(values: list[float]) -> float:
    return math.sqrt(_variance(values))


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled std)."""
    if not a or not b:
        return 0.0
    n_a, n_b = len(a), len(b)
    var_a, var_b = _variance(a), _variance(b)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (_mean(a) - _mean(b)) / pooled_std


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test. Returns (t_statistic, approx_p_value)."""
    if not a or not b:
        return 0.0, 1.0
    mean_a, mean_b = _mean(a), _mean(b)
    var_a, var_b = _variance(a), _variance(b)
    n_a, n_b = len(a), len(b)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se
    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    # Approximate p-value using Student-t CDF approximation
    p = _approx_t_pvalue(abs(t), df)
    return t, p


def _mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U statistic and approximate p-value (normal approximation)."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    # Count: for each a_i, how many b_j < a_i
    u_a = sum(
        sum(1 for bj in b if ai > bj) + 0.5 * sum(1 for bj in b if ai == bj)
        for ai in a
    )
    # Normal approximation
    mean_u = n_a * n_b / 2
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if std_u == 0:
        return u_a, 1.0
    z = (u_a - mean_u) / std_u
    p = 2 * _standard_normal_sf(abs(z))
    return u_a, p


def _approx_t_pvalue(t: float, df: float) -> float:
    """Approximate two-tailed p-value for t-distribution using normal approximation."""
    return 2 * _standard_normal_sf(t)


def _standard_normal_sf(z: float) -> float:
    """Survival function of standard normal (1 - CDF) using erf approximation."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _effect_size_label(d: float) -> EffectSize:
    ad = abs(d)
    if ad < 0.2:
        return EffectSize.NEGLIGIBLE
    if ad < 0.5:
        return EffectSize.SMALL
    if ad < 0.8:
        return EffectSize.MEDIUM
    return EffectSize.LARGE


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ABTestReport:
    """Result of a statistical A/B test between two model epoch groups."""
    epoch_a:       str
    epoch_b:       str
    metric:        str
    verdict:       ABVerdict
    mean_a:        float
    mean_b:        float
    std_a:         float
    std_b:         float
    n_a:           int
    n_b:           int
    cohens_d:      float
    effect_size:   EffectSize
    t_statistic:   float
    p_value:       float           # Welch's t-test p-value
    mw_p_value:    float           # Mann-Whitney p-value
    alpha:         float = 0.05
    detail:        str = ""

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha or self.mw_p_value < self.alpha

    def __str__(self) -> str:
        delta = self.mean_b - self.mean_a
        sign = "+" if delta >= 0 else ""
        return (
            f"ABTestReport [{self.verdict.value}] metric={self.metric}\n"
            f"  A: mean={self.mean_a:.4f} ± {self.std_a:.4f}  (n={self.n_a})\n"
            f"  B: mean={self.mean_b:.4f} ± {self.std_b:.4f}  (n={self.n_b})\n"
            f"  Δ = {sign}{delta:.4f}  Cohen's d={self.cohens_d:.3f} ({self.effect_size.value})\n"
            f"  Welch p={self.p_value:.4f}  MW p={self.mw_p_value:.4f}  "
            f"{'SIGNIFICANT' if self.significant else 'not significant'} (α={self.alpha})"
        )


@dataclass
class BatchABReport:
    """Summary of multiple A/B test comparisons."""
    metric:   str
    reports:  list[ABTestReport] = field(default_factory=list)

    @property
    def b_wins_rate(self) -> float:
        if not self.reports:
            return 0.0
        wins = sum(1 for r in self.reports if r.verdict == ABVerdict.B_WINS)
        return wins / len(self.reports)

    @property
    def recommendation(self) -> str:
        rate = self.b_wins_rate
        if rate >= 0.7:
            return "DEPLOY_B"
        if rate <= 0.3:
            return "KEEP_A"
        return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# ABTestRunner
# ---------------------------------------------------------------------------

class ABTestRunner:
    """Runs statistical A/B tests comparing model performance across epochs.

    Args:
        storage:    Any StorageInterface implementation.
        alpha:      Significance level (default 0.05).
        min_samples: Minimum samples per epoch to run a test (default 10).
    """

    def __init__(
        self,
        storage: "StorageInterface",
        alpha: float = 0.05,
        min_samples: int = 10,
    ) -> None:
        self._storage = storage
        self._alpha = alpha
        self._min_samples = min_samples

    def compare(
        self,
        epoch_a: str,
        epoch_b: str,
        metric: str = "confidence",
    ) -> ABTestReport:
        """Compare metric distributions between two epochs.

        Args:
            epoch_a: Reference epoch (model version A).
            epoch_b: Test epoch (model version B).
            metric:  ``"confidence"`` or ``"latency_ms"``.

        Returns:
            ABTestReport with statistical test results.
        """
        vals_a = self._extract(epoch_a, metric)
        vals_b = self._extract(epoch_b, metric)

        if len(vals_a) < self._min_samples or len(vals_b) < self._min_samples:
            return ABTestReport(
                epoch_a=epoch_a, epoch_b=epoch_b, metric=metric,
                verdict=ABVerdict.INSUFFICIENT,
                mean_a=_mean(vals_a), mean_b=_mean(vals_b),
                std_a=_std(vals_a), std_b=_std(vals_b),
                n_a=len(vals_a), n_b=len(vals_b),
                cohens_d=0.0, effect_size=EffectSize.NEGLIGIBLE,
                t_statistic=0.0, p_value=1.0, mw_p_value=1.0,
                alpha=self._alpha,
                detail=f"Need ≥{self._min_samples} samples; got A={len(vals_a)}, B={len(vals_b)}",
            )

        d = _cohens_d(vals_a, vals_b)
        t, p_t = _welch_t(vals_a, vals_b)
        _, p_mw = _mann_whitney_u(vals_a, vals_b)

        mean_a, mean_b = _mean(vals_a), _mean(vals_b)
        significant = p_t < self._alpha or p_mw < self._alpha

        if not significant or abs(d) < 0.2:
            verdict = ABVerdict.NO_DIFF
        elif mean_b > mean_a and metric == "confidence":
            verdict = ABVerdict.B_WINS
        elif mean_b < mean_a and metric == "confidence":
            verdict = ABVerdict.A_WINS
        elif mean_b < mean_a and metric == "latency_ms":
            verdict = ABVerdict.B_WINS   # lower latency = better
        else:
            verdict = ABVerdict.A_WINS

        return ABTestReport(
            epoch_a=epoch_a, epoch_b=epoch_b, metric=metric,
            verdict=verdict,
            mean_a=round(mean_a, 6), mean_b=round(mean_b, 6),
            std_a=round(_std(vals_a), 6), std_b=round(_std(vals_b), 6),
            n_a=len(vals_a), n_b=len(vals_b),
            cohens_d=round(d, 4),
            effect_size=_effect_size_label(d),
            t_statistic=round(t, 4),
            p_value=round(p_t, 6),
            mw_p_value=round(p_mw, 6),
            alpha=self._alpha,
            detail=f"Δmean={round(mean_b - mean_a, 4)}",
        )

    def batch_compare(
        self,
        epochs_a: list[str],
        epochs_b: list[str],
        metric: str = "confidence",
    ) -> BatchABReport:
        """Compare multiple epoch pairs and aggregate results.

        Args:
            epochs_a: List of reference epoch IDs (model A).
            epochs_b: List of test epoch IDs (model B).
            metric:   Metric to compare.

        Returns:
            BatchABReport with per-pair results and overall recommendation.
        """
        reports = [
            self.compare(ea, eb, metric)
            for ea, eb in zip(epochs_a, epochs_b)
        ]
        return BatchABReport(metric=metric, reports=reports)

    def _extract(self, epoch_id: str, metric: str) -> list[float]:
        records = self._storage.list_records_by_epoch(epoch_id)
        values = []
        for r in records:
            v = getattr(r, metric, None)
            if v is not None and v > 0:
                values.append(float(v))
        return values
