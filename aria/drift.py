"""
aria.drift — Statistical drift detection for AI model monitoring.

Detects distributional shift in model confidence scores between epochs.
All statistics are computed in pure Python — no numpy or scipy required.

Three tests are available:

  KS (Kolmogorov-Smirnov)   — Most sensitive to location and shape changes.
                               Works well with small samples.
  KL (Kullback-Leibler)     — Measures information gain P→Q.
                               Asymmetric; requires smoothing for zero bins.
  JS (Jensen-Shannon)       — Symmetric, bounded [0, log2].  Recommended default.

Usage::

    from aria.drift import DriftDetector
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    detector = DriftDetector(storage, threshold=0.15)

    result = detector.compare("epoch-v1", "epoch-v2")
    if result.drift_detected:
        print(f"Drift detected! JS={result.statistic:.4f} (threshold {result.threshold})")

    # Or batch-compare the last N epochs:
    summary = detector.sliding_window_check("sys-id", n_epochs=5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage.base import StorageInterface


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Result of a statistical drift test between two epochs.

    Attributes:
        epoch_a:        Reference epoch ID.
        epoch_b:        Current epoch ID.
        test:           Statistical test used (``"ks"``, ``"kl"``, ``"js"``).
        statistic:      Test statistic value.
        threshold:      Drift threshold (configured at detector level).
        drift_detected: ``True`` if ``statistic > threshold``.
        sample_size_a:  Number of confidence values from epoch A.
        sample_size_b:  Number of confidence values from epoch B.
        detail:         Human-readable summary.
    """
    epoch_a: str
    epoch_b: str
    test: str
    statistic: float
    threshold: float
    drift_detected: bool
    sample_size_a: int
    sample_size_b: int
    detail: str = ""


@dataclass
class SlidingWindowDrift:
    """Drift summary across a sliding window of consecutive epochs."""
    system_id: str
    epochs: list[str]
    results: list[DriftResult]

    @property
    def any_drift(self) -> bool:
        return any(r.drift_detected for r in self.results)

    @property
    def max_statistic(self) -> float:
        if not self.results:
            return 0.0
        return max(r.statistic for r in self.results)


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers
# ---------------------------------------------------------------------------

def _build_histogram(values: list[float], n_bins: int = 20) -> list[float]:
    """Build a probability histogram with ``n_bins`` uniform bins over [0, 1]."""
    if not values:
        return [0.0] * n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int(v * n_bins), n_bins - 1)
        counts[idx] += 1
    n = len(values)
    return [c / n for c in counts]


def _smooth(hist: list[float], epsilon: float = 1e-9) -> list[float]:
    """Add tiny epsilon to avoid log(0) in KL/JS calculations."""
    total = sum(hist) + epsilon * len(hist)
    return [(h + epsilon) / total for h in hist]


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic.

    Returns D = max|CDF_a(x) - CDF_b(x)| in [0, 1].
    Returns 0.0 if either sample is empty.
    """
    if not a or not b:
        return 0.0
    all_vals = sorted(set(a + b))
    n_a, n_b = len(a), len(b)
    a_sorted, b_sorted = sorted(a), sorted(b)

    max_diff = 0.0
    i_a = i_b = 0
    for x in all_vals:
        while i_a < n_a and a_sorted[i_a] <= x:
            i_a += 1
        while i_b < n_b and b_sorted[i_b] <= x:
            i_b += 1
        diff = abs(i_a / n_a - i_b / n_b)
        if diff > max_diff:
            max_diff = diff
    return round(max_diff, 6)


def kl_divergence(p: list[float], q: list[float], n_bins: int = 20) -> float:
    """KL divergence KL(P || Q) using histogram binning.

    Returns a value ≥ 0; 0 means identical distributions.
    """
    hist_p = _smooth(_build_histogram(p, n_bins))
    hist_q = _smooth(_build_histogram(q, n_bins))
    return round(sum(pi * math.log(pi / qi) for pi, qi in zip(hist_p, hist_q)), 6)


def js_divergence(p: list[float], q: list[float], n_bins: int = 20) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, log(2)]).

    Normalized to [0, 1] by dividing by log(2).
    Returns 0 for identical distributions, 1 for maximally different.
    """
    hist_p = _smooth(_build_histogram(p, n_bins))
    hist_q = _smooth(_build_histogram(q, n_bins))
    m = [(pi + qi) / 2 for pi, qi in zip(hist_p, hist_q)]
    kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(hist_p, m))
    kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(hist_q, m))
    js = (kl_pm + kl_qm) / 2
    return round(js / math.log(2), 6)  # Normalise to [0, 1]


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detects distributional drift in model confidence scores.

    Args:
        storage:    Any StorageInterface implementation.
        threshold:  Drift threshold.  Alarm fires when statistic > threshold.
                    Typical values: KS 0.1–0.2, JS 0.05–0.15.
        test:       Statistical test — ``"js"`` (default), ``"ks"``, or ``"kl"``.
        n_bins:     Number of histogram bins for KL/JS tests (default 20).
        min_samples: Minimum confidence values required per epoch to run a test.
                    Returns a ``no-data`` result if below this threshold.
    """

    def __init__(
        self,
        storage: "StorageInterface",
        threshold: float = 0.10,
        test: str = "js",
        n_bins: int = 20,
        min_samples: int = 10,
    ) -> None:
        self._storage = storage
        self._threshold = threshold
        self._test = test.lower()
        self._n_bins = n_bins
        self._min_samples = min_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, epoch_a: str, epoch_b: str) -> DriftResult:
        """Compare confidence distributions of two epochs.

        ``epoch_a`` is the **reference** (baseline), ``epoch_b`` is **current**.
        Drift is measured as shift from A to B.

        Returns a DriftResult — never raises even if epochs don't exist.
        """
        conf_a = self._confidences(epoch_a)
        conf_b = self._confidences(epoch_b)

        if len(conf_a) < self._min_samples or len(conf_b) < self._min_samples:
            return DriftResult(
                epoch_a=epoch_a, epoch_b=epoch_b,
                test=self._test, statistic=0.0,
                threshold=self._threshold, drift_detected=False,
                sample_size_a=len(conf_a), sample_size_b=len(conf_b),
                detail=(
                    f"Insufficient data: epoch_a has {len(conf_a)} samples, "
                    f"epoch_b has {len(conf_b)} samples "
                    f"(min required: {self._min_samples})"
                ),
            )

        stat = self._compute(conf_a, conf_b)
        detected = stat > self._threshold
        detail = (
            f"{self._test.upper()} statistic={stat:.4f} "
            f"(threshold={self._threshold}, n_a={len(conf_a)}, n_b={len(conf_b)})"
        )
        return DriftResult(
            epoch_a=epoch_a, epoch_b=epoch_b,
            test=self._test, statistic=stat,
            threshold=self._threshold, drift_detected=detected,
            sample_size_a=len(conf_a), sample_size_b=len(conf_b),
            detail=detail,
        )

    def sliding_window_check(
        self,
        system_id: str | None = None,
        n_epochs: int = 5,
    ) -> SlidingWindowDrift:
        """Compare consecutive epoch pairs in the most recent N epochs.

        Returns a SlidingWindowDrift with one DriftResult per consecutive pair.
        """
        epochs = self._storage.list_epochs(system_id=system_id, limit=n_epochs)
        # list_epochs returns DESC; reverse to get chronological order
        epochs = list(reversed(epochs))
        ids = [e.epoch_id for e in epochs]

        results = []
        for i in range(len(ids) - 1):
            results.append(self.compare(ids[i], ids[i + 1]))

        return SlidingWindowDrift(
            system_id=system_id or "",
            epochs=ids,
            results=results,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _confidences(self, epoch_id: str) -> list[float]:
        records = self._storage.list_records_by_epoch(epoch_id)
        return [r.confidence for r in records if r.confidence is not None]

    def _compute(self, a: list[float], b: list[float]) -> float:
        if self._test == "ks":
            return ks_statistic(a, b)
        elif self._test == "kl":
            return kl_divergence(a, b, self._n_bins)
        else:  # "js" default
            return js_divergence(a, b, self._n_bins)
