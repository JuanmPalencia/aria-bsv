"""
aria.canary — Canary deployment tracking for gradual model rollouts.

Monitors a small traffic slice (canary) against the baseline (stable) model,
automatically comparing distributions and flagging regressions.

Usage::

    from aria.canary import CanaryTracker

    tracker = CanaryTracker(
        storage,
        stable_epoch="epoch-v1-stable",
        canary_epoch="epoch-v1-canary",
    )

    status = tracker.status()
    print(status)
    # CanaryStatus: canary=HEALTHY  confidence Δ=+0.032  latency Δ=-12ms

    # Promote if canary is healthy across all metrics:
    if tracker.should_promote():
        print("Canary looks good — promote to stable")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)


class CanaryHealth(str, Enum):
    HEALTHY    = "HEALTHY"      # Canary ≥ stable (or within tolerance)
    DEGRADED   = "DEGRADED"     # Canary worse but within warn threshold
    CRITICAL   = "CRITICAL"     # Canary significantly worse — rollback recommended
    UNKNOWN    = "UNKNOWN"      # Insufficient data


@dataclass
class MetricComparison:
    """Comparison of a single metric between stable and canary."""
    metric:      str
    mean_stable: float
    mean_canary: float
    delta:       float          # canary - stable
    delta_pct:   float          # delta / stable * 100
    health:      CanaryHealth


@dataclass
class CanaryStatus:
    """Overall canary deployment status."""
    stable_epoch:   str
    canary_epoch:   str
    health:         CanaryHealth
    comparisons:    list[MetricComparison] = field(default_factory=list)
    n_stable:       int = 0
    n_canary:       int = 0
    recommendation: str = ""

    def __str__(self) -> str:
        parts = [f"CanaryStatus: canary={self.health.value}"]
        for c in self.comparisons:
            sign = "+" if c.delta >= 0 else ""
            parts.append(
                f"  {c.metric}: stable={c.mean_stable:.4f}  "
                f"canary={c.mean_canary:.4f}  Δ={sign}{c.delta:.4f}"
            )
        if self.recommendation:
            parts.append(f"  → {self.recommendation}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CanaryTracker
# ---------------------------------------------------------------------------

class CanaryTracker:
    """Tracks canary vs stable model performance across metrics.

    Args:
        storage:          StorageInterface implementation.
        stable_epoch:     Epoch ID for the stable (baseline) model.
        canary_epoch:     Epoch ID for the canary model.
        metrics:          List of metric names to compare (default: confidence, latency_ms).
        warn_threshold:   Relative degradation (%) that triggers DEGRADED (default 5%).
        critical_threshold: Relative degradation (%) that triggers CRITICAL (default 15%).
        min_samples:      Minimum samples required per epoch (default 10).
    """

    _HIGHER_IS_BETTER = {"confidence"}
    _LOWER_IS_BETTER  = {"latency_ms"}

    def __init__(
        self,
        storage: "StorageInterface",
        stable_epoch: str,
        canary_epoch: str,
        metrics: list[str] | None = None,
        warn_threshold: float = 5.0,
        critical_threshold: float = 15.0,
        min_samples: int = 10,
    ) -> None:
        self._storage = storage
        self._stable = stable_epoch
        self._canary = canary_epoch
        self._metrics = metrics or ["confidence", "latency_ms"]
        self._warn = warn_threshold
        self._critical = critical_threshold
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self) -> CanaryStatus:
        """Compute current canary health status."""
        stable_recs = self._storage.list_records_by_epoch(self._stable)
        canary_recs = self._storage.list_records_by_epoch(self._canary)

        n_stable = len(stable_recs)
        n_canary = len(canary_recs)

        if n_stable < self._min_samples or n_canary < self._min_samples:
            return CanaryStatus(
                stable_epoch=self._stable,
                canary_epoch=self._canary,
                health=CanaryHealth.UNKNOWN,
                n_stable=n_stable,
                n_canary=n_canary,
                recommendation=(
                    f"Need ≥{self._min_samples} samples; "
                    f"stable={n_stable}, canary={n_canary}"
                ),
            )

        comparisons = []
        for metric in self._metrics:
            s_vals = self._extract(stable_recs, metric)
            c_vals = self._extract(canary_recs, metric)
            if not s_vals or not c_vals:
                continue
            cmp = self._compare_metric(metric, s_vals, c_vals)
            comparisons.append(cmp)

        overall = self._aggregate_health(comparisons)
        recommendation = self._recommendation(overall)

        return CanaryStatus(
            stable_epoch=self._stable,
            canary_epoch=self._canary,
            health=overall,
            comparisons=comparisons,
            n_stable=n_stable,
            n_canary=n_canary,
            recommendation=recommendation,
        )

    def should_promote(self) -> bool:
        """Return True if the canary is healthy enough to promote."""
        s = self.status()
        return s.health == CanaryHealth.HEALTHY

    def should_rollback(self) -> bool:
        """Return True if the canary is critical and should be rolled back."""
        s = self.status()
        return s.health == CanaryHealth.CRITICAL

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract(self, records: list[Any], metric: str) -> list[float]:
        values = []
        for r in records:
            v = getattr(r, metric, None)
            if v is not None and float(v) > 0:
                values.append(float(v))
        return values

    def _mean(self, vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _compare_metric(
        self,
        metric: str,
        stable: list[float],
        canary: list[float],
    ) -> MetricComparison:
        ms = self._mean(stable)
        mc = self._mean(canary)
        delta = mc - ms
        delta_pct = (delta / ms * 100) if ms != 0 else 0.0

        # Determine health based on direction of improvement
        if metric in self._HIGHER_IS_BETTER:
            # Positive delta = canary better
            degradation_pct = -delta_pct  # negative delta = degradation
        elif metric in self._LOWER_IS_BETTER:
            # Negative delta = canary better (lower latency)
            degradation_pct = delta_pct   # positive delta = degradation
        else:
            degradation_pct = abs(delta_pct)  # unknown metric — conservative

        if degradation_pct >= self._critical:
            health = CanaryHealth.CRITICAL
        elif degradation_pct >= self._warn:
            health = CanaryHealth.DEGRADED
        else:
            health = CanaryHealth.HEALTHY

        return MetricComparison(
            metric=metric,
            mean_stable=round(ms, 6),
            mean_canary=round(mc, 6),
            delta=round(delta, 6),
            delta_pct=round(delta_pct, 2),
            health=health,
        )

    def _aggregate_health(self, comparisons: list[MetricComparison]) -> CanaryHealth:
        if not comparisons:
            return CanaryHealth.UNKNOWN
        # Worst metric wins
        if any(c.health == CanaryHealth.CRITICAL for c in comparisons):
            return CanaryHealth.CRITICAL
        if any(c.health == CanaryHealth.DEGRADED for c in comparisons):
            return CanaryHealth.DEGRADED
        return CanaryHealth.HEALTHY

    def _recommendation(self, health: CanaryHealth) -> str:
        return {
            CanaryHealth.HEALTHY:  "PROMOTE — canary performing at or above stable",
            CanaryHealth.DEGRADED: "MONITOR — canary shows minor degradation",
            CanaryHealth.CRITICAL: "ROLLBACK — canary significantly underperforming",
            CanaryHealth.UNKNOWN:  "WAIT — insufficient data to decide",
        }[health]
