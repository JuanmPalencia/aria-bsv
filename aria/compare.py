"""
aria.compare — Model comparison across epochs.

Compare audit metrics (confidence, latency, drift) between two or more
models or between two time periods for the same model.

Usage::

    from aria.compare import ModelComparator

    cmp = ModelComparator(storage)

    # Compare two models in the same epoch
    result = cmp.compare_models("gpt-4", "gpt-3.5", epoch_id="epoch-001")

    # Compare same model across epochs (A/B test scenario)
    result = cmp.compare_epochs("epoch-001", "epoch-002", model_id="gpt-4")

    # Quick summary
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.record import AuditRecord
    from .storage.base import StorageInterface


@dataclass
class ModelStats:
    """Statistics for one model (or one group of records)."""
    label: str
    count: int = 0
    avg_confidence: float | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    stddev_confidence: float | None = None
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "stddev_confidence": self.stddev_confidence,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
        }


@dataclass
class ComparisonResult:
    """Result of comparing two groups of records."""
    group_a: ModelStats
    group_b: ModelStats
    confidence_delta: float | None = None
    latency_delta_ms: float = 0.0
    winner: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_a": self.group_a.to_dict(),
            "group_b": self.group_b.to_dict(),
            "confidence_delta": self.confidence_delta,
            "latency_delta_ms": round(self.latency_delta_ms, 1),
            "winner": self.winner,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Comparison: {self.group_a.label} vs {self.group_b.label}",
            "=" * 50,
            f"  {self.group_a.label}: {self.group_a.count} records, "
            f"confidence={self.group_a.avg_confidence}, "
            f"latency={self.group_a.avg_latency_ms:.0f}ms",
            f"  {self.group_b.label}: {self.group_b.count} records, "
            f"confidence={self.group_b.avg_confidence}, "
            f"latency={self.group_b.avg_latency_ms:.0f}ms",
            "-" * 50,
        ]
        if self.confidence_delta is not None:
            lines.append(f"  Confidence delta: {self.confidence_delta:+.4f}")
        lines.append(f"  Latency delta: {self.latency_delta_ms:+.0f}ms")
        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


def _compute_stats(label: str, records: list["AuditRecord"]) -> ModelStats:
    """Compute stats from a list of records."""
    if not records:
        return ModelStats(label=label)

    confs = [r.confidence for r in records if r.confidence is not None]
    lats = sorted(r.latency_ms for r in records)

    avg_c = sum(confs) / len(confs) if confs else None
    stddev_c = None
    if confs and avg_c is not None and len(confs) > 1:
        variance = sum((c - avg_c) ** 2 for c in confs) / (len(confs) - 1)
        stddev_c = variance ** 0.5

    def percentile(arr: list[int], p: float) -> float:
        if not arr:
            return 0.0
        idx = int(len(arr) * p / 100)
        return float(arr[min(idx, len(arr) - 1)])

    return ModelStats(
        label=label,
        count=len(records),
        avg_confidence=round(avg_c, 6) if avg_c is not None else None,
        min_confidence=min(confs) if confs else None,
        max_confidence=max(confs) if confs else None,
        stddev_confidence=round(stddev_c, 6) if stddev_c is not None else None,
        avg_latency_ms=sum(lats) / len(lats) if lats else 0,
        p50_latency_ms=percentile(lats, 50),
        p95_latency_ms=percentile(lats, 95),
        p99_latency_ms=percentile(lats, 99),
    )


class ModelComparator:
    """Compare models or epochs using audit records.

    Args:
        storage: ARIA StorageInterface for data access.
    """

    def __init__(self, storage: "StorageInterface") -> None:
        self._storage = storage

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        epoch_id: str | None = None,
        system_id: str | None = None,
    ) -> ComparisonResult:
        """Compare two models across one or all epochs.

        Args:
            model_a: First model ID.
            model_b: Second model ID.
            epoch_id: Specific epoch (if None, uses all epochs).
            system_id: Filter epochs by system ID.
        """
        records_a, records_b = self._split_by_model(
            model_a, model_b, epoch_id, system_id,
        )
        return self._compare(model_a, records_a, model_b, records_b)

    def compare_epochs(
        self,
        epoch_a: str,
        epoch_b: str,
        model_id: str | None = None,
    ) -> ComparisonResult:
        """Compare the same model across two epochs.

        Args:
            epoch_a: First epoch ID.
            epoch_b: Second epoch ID.
            model_id: Filter to specific model (if None, uses all).
        """
        recs_a = self._storage.list_records_by_epoch(epoch_a)
        recs_b = self._storage.list_records_by_epoch(epoch_b)

        if model_id:
            recs_a = [r for r in recs_a if r.model_id == model_id]
            recs_b = [r for r in recs_b if r.model_id == model_id]

        label_a = f"{epoch_a}" + (f"/{model_id}" if model_id else "")
        label_b = f"{epoch_b}" + (f"/{model_id}" if model_id else "")

        return self._compare(label_a, recs_a, label_b, recs_b)

    def rank_models(
        self,
        epoch_id: str | None = None,
        system_id: str | None = None,
        by: str = "confidence",
    ) -> list[ModelStats]:
        """Rank all models by a metric.

        Args:
            epoch_id: Specific epoch (if None, uses all).
            system_id: Filter epochs by system ID.
            by: Metric to rank by ('confidence' or 'latency').

        Returns:
            List of ModelStats sorted by metric (best first).
        """
        records = self._get_records(epoch_id, system_id)

        # Group by model
        from collections import defaultdict
        buckets: dict[str, list["AuditRecord"]] = defaultdict(list)
        for r in records:
            buckets[r.model_id].append(r)

        stats = [_compute_stats(mid, recs) for mid, recs in buckets.items()]

        if by == "latency":
            stats.sort(key=lambda s: s.avg_latency_ms)
        else:
            stats.sort(
                key=lambda s: s.avg_confidence if s.avg_confidence is not None else -1,
                reverse=True,
            )

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_records(
        self,
        epoch_id: str | None,
        system_id: str | None,
    ) -> list["AuditRecord"]:
        if epoch_id:
            return self._storage.list_records_by_epoch(epoch_id)
        epochs = self._storage.list_epochs(system_id=system_id, limit=10_000)
        all_recs: list["AuditRecord"] = []
        for e in epochs:
            all_recs.extend(self._storage.list_records_by_epoch(e.epoch_id))
        return all_recs

    def _split_by_model(
        self,
        model_a: str,
        model_b: str,
        epoch_id: str | None,
        system_id: str | None,
    ) -> tuple[list["AuditRecord"], list["AuditRecord"]]:
        records = self._get_records(epoch_id, system_id)
        recs_a = [r for r in records if r.model_id == model_a]
        recs_b = [r for r in records if r.model_id == model_b]
        return recs_a, recs_b

    def _compare(
        self,
        label_a: str,
        records_a: list["AuditRecord"],
        label_b: str,
        records_b: list["AuditRecord"],
    ) -> ComparisonResult:
        stats_a = _compute_stats(label_a, records_a)
        stats_b = _compute_stats(label_b, records_b)

        conf_delta = None
        if stats_a.avg_confidence is not None and stats_b.avg_confidence is not None:
            conf_delta = round(stats_a.avg_confidence - stats_b.avg_confidence, 6)

        lat_delta = stats_a.avg_latency_ms - stats_b.avg_latency_ms

        # Determine winner (higher confidence wins; tie → lower latency wins)
        winner = ""
        notes: list[str] = []
        if conf_delta is not None:
            if conf_delta > 0.01:
                winner = label_a
                notes.append(f"{label_a} has higher confidence")
            elif conf_delta < -0.01:
                winner = label_b
                notes.append(f"{label_b} has higher confidence")
            else:
                notes.append("Confidence is similar")
                if lat_delta < -10:
                    winner = label_a
                    notes.append(f"{label_a} is faster")
                elif lat_delta > 10:
                    winner = label_b
                    notes.append(f"{label_b} is faster")

        if stats_a.count < 30 or stats_b.count < 30:
            notes.append("Warning: small sample size (< 30 records)")

        return ComparisonResult(
            group_a=stats_a,
            group_b=stats_b,
            confidence_delta=conf_delta,
            latency_delta_ms=lat_delta,
            winner=winner,
            notes=notes,
        )
