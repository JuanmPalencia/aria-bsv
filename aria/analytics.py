"""
aria.analytics — Cross-epoch analytics for ARIA audit data.

Provides aggregated statistics across one or more epochs without requiring
any external dependencies (pure Python + stdlib math).

Usage::

    from aria.analytics import CrossEpochAnalytics
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    analytics = CrossEpochAnalytics(storage)

    stats = analytics.latency_stats(["ep-001", "ep-002"])
    print(stats.p95_ms)

    health = analytics.epoch_health("ep-001")
    print(health.is_closed, health.merkle_root_present)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.record import AuditRecord
    from .storage.base import EpochRow, StorageInterface


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Latency statistics across a set of records."""
    count: int
    mean_ms: float
    min_ms: int
    max_ms: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    stddev_ms: float

    @classmethod
    def empty(cls) -> "LatencyStats":
        return cls(0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class ConfidenceStats:
    """Confidence score statistics across a set of records."""
    count: int                     # records with a confidence value
    mean: float
    min: float
    max: float
    p50: float
    p95: float
    stddev: float
    histogram: dict[str, int]      # bucket label → count

    @classmethod
    def empty(cls) -> "ConfidenceStats":
        return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})


@dataclass
class ModelUsage:
    """Per-model record counts and latency across the queried epochs."""
    model_id: str
    record_count: int
    mean_latency_ms: float
    mean_confidence: float | None


@dataclass
class EpochHealth:
    """Summary health status for a single epoch."""
    epoch_id: str
    is_closed: bool
    merkle_root_present: bool
    record_count: int
    open_txid_present: bool
    close_txid_present: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.is_closed and self.merkle_root_present and not self.warnings


@dataclass
class DriftReport:
    """Drift comparison between two epochs."""
    epoch_a: str
    epoch_b: str
    mean_confidence_a: float | None
    mean_confidence_b: float | None
    confidence_delta: float | None   # b - a; negative = drift down
    mean_latency_a_ms: float
    mean_latency_b_ms: float
    latency_delta_ms: float          # b - a; positive = slower
    record_count_a: int
    record_count_b: int


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class CrossEpochAnalytics:
    """Compute aggregated statistics across ARIA audit epochs.

    Thread-safe: all methods create a new read-only view from storage on each
    call, so they can be called concurrently from multiple threads.

    Args:
        storage: Any ``StorageInterface`` implementation.
    """

    def __init__(self, storage: "StorageInterface") -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Latency analysis
    # ------------------------------------------------------------------

    def latency_stats(self, epoch_ids: list[str]) -> LatencyStats:
        """Compute latency percentiles across the given epochs."""
        latencies = self._collect_latencies(epoch_ids)
        if not latencies:
            return LatencyStats.empty()
        return _compute_latency_stats(latencies)

    # ------------------------------------------------------------------
    # Confidence analysis
    # ------------------------------------------------------------------

    def confidence_stats(self, epoch_ids: list[str]) -> ConfidenceStats:
        """Compute confidence score distribution across the given epochs."""
        records = self._all_records(epoch_ids)
        values = [r.confidence for r in records if r.confidence is not None]
        if not values:
            return ConfidenceStats.empty()
        return _compute_confidence_stats(values)

    # ------------------------------------------------------------------
    # Model usage
    # ------------------------------------------------------------------

    def model_usage(self, epoch_ids: list[str]) -> list[ModelUsage]:
        """Return per-model statistics across the given epochs."""
        records = self._all_records(epoch_ids)
        buckets: dict[str, list["AuditRecord"]] = {}
        for r in records:
            buckets.setdefault(r.model_id, []).append(r)

        result = []
        for model_id, recs in sorted(buckets.items()):
            latencies = [r.latency_ms for r in recs]
            confidences = [r.confidence for r in recs if r.confidence is not None]
            result.append(ModelUsage(
                model_id=model_id,
                record_count=len(recs),
                mean_latency_ms=sum(latencies) / len(latencies),
                mean_confidence=sum(confidences) / len(confidences) if confidences else None,
            ))
        return result

    # ------------------------------------------------------------------
    # Epoch health
    # ------------------------------------------------------------------

    def epoch_health(self, epoch_id: str) -> EpochHealth:
        """Return a health summary for a single epoch."""
        row = self._storage.get_epoch(epoch_id)
        if row is None:
            return EpochHealth(
                epoch_id=epoch_id,
                is_closed=False,
                merkle_root_present=False,
                record_count=0,
                open_txid_present=False,
                close_txid_present=False,
                warnings=[f"Epoch {epoch_id!r} not found in storage"],
            )

        is_closed = bool(row.close_txid)
        merkle_ok = bool(row.merkle_root)
        warnings: list[str] = []

        if not is_closed:
            warnings.append("Epoch is not yet closed (no close_txid)")
        if is_closed and not merkle_ok:
            warnings.append("Epoch is closed but merkle_root is empty")
        if not row.open_txid:
            warnings.append("Epoch has no open_txid (broadcast may have failed)")
        if is_closed and row.records_count == 0:
            warnings.append("Closed epoch contains zero records")

        # Cross-check record count
        actual_records = self._storage.list_records_by_epoch(epoch_id)
        if len(actual_records) != row.records_count:
            warnings.append(
                f"records_count mismatch: epoch says {row.records_count}, "
                f"storage has {len(actual_records)}"
            )

        return EpochHealth(
            epoch_id=epoch_id,
            is_closed=is_closed,
            merkle_root_present=merkle_ok,
            record_count=len(actual_records),
            open_txid_present=bool(row.open_txid),
            close_txid_present=is_closed,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def drift_report(self, epoch_id_a: str, epoch_id_b: str) -> DriftReport:
        """Compare two epochs for latency and confidence drift."""
        records_a = self._storage.list_records_by_epoch(epoch_id_a)
        records_b = self._storage.list_records_by_epoch(epoch_id_b)

        lat_a = [r.latency_ms for r in records_a]
        lat_b = [r.latency_ms for r in records_b]
        conf_a = [r.confidence for r in records_a if r.confidence is not None]
        conf_b = [r.confidence for r in records_b if r.confidence is not None]

        mean_lat_a = sum(lat_a) / len(lat_a) if lat_a else 0.0
        mean_lat_b = sum(lat_b) / len(lat_b) if lat_b else 0.0
        mean_conf_a: float | None = sum(conf_a) / len(conf_a) if conf_a else None
        mean_conf_b: float | None = sum(conf_b) / len(conf_b) if conf_b else None

        conf_delta: float | None = None
        if mean_conf_a is not None and mean_conf_b is not None:
            conf_delta = mean_conf_b - mean_conf_a

        return DriftReport(
            epoch_a=epoch_id_a,
            epoch_b=epoch_id_b,
            mean_confidence_a=mean_conf_a,
            mean_confidence_b=mean_conf_b,
            confidence_delta=conf_delta,
            mean_latency_a_ms=mean_lat_a,
            mean_latency_b_ms=mean_lat_b,
            latency_delta_ms=mean_lat_b - mean_lat_a,
            record_count_a=len(records_a),
            record_count_b=len(records_b),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_records(self, epoch_ids: list[str]) -> list["AuditRecord"]:
        records = []
        for eid in epoch_ids:
            records.extend(self._storage.list_records_by_epoch(eid))
        return records

    def _collect_latencies(self, epoch_ids: list[str]) -> list[int]:
        return [r.latency_ms for r in self._all_records(epoch_ids)]


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float | int], p: float) -> float:
    """Compute the p-th percentile of a pre-sorted list (0 ≤ p ≤ 100)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _compute_latency_stats(latencies: list[int]) -> LatencyStats:
    s = sorted(latencies)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return LatencyStats(
        count=n,
        mean_ms=round(mean, 2),
        min_ms=s[0],
        max_ms=s[-1],
        p50_ms=round(_percentile(s, 50), 2),
        p95_ms=round(_percentile(s, 95), 2),
        p99_ms=round(_percentile(s, 99), 2),
        stddev_ms=round(math.sqrt(variance), 2),
    )


def _compute_confidence_stats(values: list[float]) -> ConfidenceStats:
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n

    # Build a 10-bucket histogram [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    histogram: dict[str, int] = {}
    for v in values:
        bucket = min(int(v * 10), 9)
        label = f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"
        histogram[label] = histogram.get(label, 0) + 1

    return ConfidenceStats(
        count=n,
        mean=round(mean, 4),
        min=round(s[0], 4),
        max=round(s[-1], 4),
        p50=round(_percentile(s, 50), 4),
        p95=round(_percentile(s, 95), 4),
        stddev=round(math.sqrt(variance), 4),
        histogram=histogram,
    )
