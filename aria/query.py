"""
aria.query — Fluent query API for audit records.

Search and aggregate audit records without writing SQL.
Works with any StorageInterface (SQLite, Postgres, etc.).

Usage::

    from aria.query import RecordQuery
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    q = RecordQuery(storage)

    # Find low-confidence records in the last 24 hours
    results = (
        q.where(confidence__lt=0.7)
         .since("24h")
         .model("gpt-4")
         .limit(50)
         .execute()
    )

    # Aggregate
    stats = q.where(model_id="gpt-4").since("7d").stats()
    print(stats)  # {"count": 1420, "avg_confidence": 0.89, "avg_latency_ms": 234}

    # Group by model
    groups = q.since("24h").group_by("model_id").execute()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.record import AuditRecord
    from .storage.base import StorageInterface

_DURATION_MAP = {
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def _parse_duration(s: str) -> float:
    """Parse a human duration string like '24h', '7d', '30m' to seconds."""
    s = s.strip().lower()
    for suffix, mult in _DURATION_MAP.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


@dataclass
class QueryStats:
    """Aggregated statistics from a query."""
    count: int = 0
    avg_confidence: float | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    models: list[str] = field(default_factory=list)
    epochs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "models": self.models,
            "epochs": self.epochs,
        }


@dataclass
class GroupResult:
    """Result of a group_by operation."""
    key: str
    value: str
    count: int
    avg_confidence: float | None = None
    avg_latency_ms: float = 0.0


class RecordQuery:
    """Fluent query builder for audit records.

    Chains filter/aggregate operations and executes against storage.

    Args:
        storage: ARIA StorageInterface for data access.
    """

    def __init__(self, storage: "StorageInterface") -> None:
        self._storage = storage
        self._filters: dict[str, Any] = {}
        self._since_sec: float | None = None
        self._until_sec: float | None = None
        self._limit_n: int | None = None
        self._model_filter: str | None = None
        self._system_filter: str | None = None
        self._epoch_filter: str | None = None
        self._group_key: str | None = None
        self._order_by: str | None = None
        self._order_desc: bool = True

    def _clone(self) -> "RecordQuery":
        q = RecordQuery(self._storage)
        q._filters = dict(self._filters)
        q._since_sec = self._since_sec
        q._until_sec = self._until_sec
        q._limit_n = self._limit_n
        q._model_filter = self._model_filter
        q._system_filter = self._system_filter
        q._epoch_filter = self._epoch_filter
        q._group_key = self._group_key
        q._order_by = self._order_by
        q._order_desc = self._order_desc
        return q

    # ------------------------------------------------------------------
    # Filter methods (all return self for chaining)
    # ------------------------------------------------------------------

    def where(self, **kwargs: Any) -> "RecordQuery":
        """Add filter conditions.

        Supports Django-style lookups::

            .where(confidence__lt=0.7)          # confidence < 0.7
            .where(confidence__gt=0.9)          # confidence > 0.9
            .where(confidence__gte=0.5)         # confidence >= 0.5
            .where(latency_ms__gt=1000)         # latency > 1000ms
            .where(model_id="gpt-4")            # exact match
            .where(model_id__contains="gpt")    # substring match
        """
        q = self._clone()
        q._filters.update(kwargs)
        return q

    def since(self, duration: str) -> "RecordQuery":
        """Filter to records created within the last duration.

        Args:
            duration: Human string like '24h', '7d', '30m', '2w'.
        """
        q = self._clone()
        q._since_sec = _parse_duration(duration)
        return q

    def until(self, duration: str) -> "RecordQuery":
        """Filter to records created before the given duration ago."""
        q = self._clone()
        q._until_sec = _parse_duration(duration)
        return q

    def model(self, model_id: str) -> "RecordQuery":
        """Filter by model ID."""
        q = self._clone()
        q._model_filter = model_id
        return q

    def system(self, system_id: str) -> "RecordQuery":
        """Filter by system ID (scopes to epochs of that system)."""
        q = self._clone()
        q._system_filter = system_id
        return q

    def epoch(self, epoch_id: str) -> "RecordQuery":
        """Filter to records in a specific epoch."""
        q = self._clone()
        q._epoch_filter = epoch_id
        return q

    def limit(self, n: int) -> "RecordQuery":
        """Limit the number of results."""
        q = self._clone()
        q._limit_n = n
        return q

    def order_by(self, field: str, desc: bool = True) -> "RecordQuery":
        """Order results by a field."""
        q = self._clone()
        q._order_by = field
        q._order_desc = desc
        return q

    def group_by(self, key: str) -> "RecordQuery":
        """Group results by a field (model_id, epoch_id, etc.)."""
        q = self._clone()
        q._group_key = key
        return q

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self) -> list["AuditRecord"] | list[GroupResult]:
        """Execute the query and return matching records (or groups)."""
        records = self._fetch_and_filter()

        if self._group_key:
            return self._do_group(records)

        if self._order_by:
            records = self._do_sort(records)

        if self._limit_n:
            records = records[:self._limit_n]

        return records

    def count(self) -> int:
        """Count matching records."""
        return len(self._fetch_and_filter())

    def stats(self) -> QueryStats:
        """Compute aggregate statistics for matching records."""
        records = self._fetch_and_filter()
        if not records:
            return QueryStats()

        confs = [r.confidence for r in records if r.confidence is not None]
        lats = sorted(r.latency_ms for r in records)
        models = sorted(set(r.model_id for r in records))
        epochs = sorted(set(r.epoch_id for r in records))

        p95_idx = int(len(lats) * 0.95)
        p95 = lats[min(p95_idx, len(lats) - 1)] if lats else 0.0

        return QueryStats(
            count=len(records),
            avg_confidence=sum(confs) / len(confs) if confs else None,
            min_confidence=min(confs) if confs else None,
            max_confidence=max(confs) if confs else None,
            avg_latency_ms=sum(lats) / len(lats) if lats else 0.0,
            p95_latency_ms=float(p95),
            models=models,
            epochs=epochs,
        )

    def first(self) -> "AuditRecord | None":
        """Return the first matching record, or None."""
        results = self.limit(1).execute()
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_and_filter(self) -> list["AuditRecord"]:
        """Fetch records from storage and apply in-memory filters."""
        # Determine which epochs to scan
        if self._epoch_filter:
            epoch_ids = [self._epoch_filter]
        else:
            epochs = self._storage.list_epochs(
                system_id=self._system_filter, limit=10_000
            )
            epoch_ids = [e.epoch_id for e in epochs]

        # Time filter on epochs
        if self._since_sec is not None:
            cutoff_ms = int((time.time() - self._since_sec) * 1000)
            all_epochs = [self._storage.get_epoch(eid) for eid in epoch_ids]
            epoch_ids = [
                e.epoch_id for e in all_epochs
                if e and e.opened_at and e.opened_at >= cutoff_ms
            ]

        all_records: list["AuditRecord"] = []
        for eid in epoch_ids:
            all_records.extend(self._storage.list_records_by_epoch(eid))

        # Apply filters
        filtered = []
        for r in all_records:
            if self._model_filter and r.model_id != self._model_filter:
                continue
            if not self._match_filters(r):
                continue
            filtered.append(r)

        return filtered

    def _match_filters(self, record: "AuditRecord") -> bool:
        """Check if a record matches all where() filters."""
        for key, value in self._filters.items():
            if "__" in key:
                field_name, op = key.rsplit("__", 1)
            else:
                field_name, op = key, "eq"

            actual = getattr(record, field_name, None)
            if actual is None:
                if op in ("lt", "gt", "lte", "gte"):
                    return False
                continue

            if op == "eq" and actual != value:
                return False
            elif op == "lt" and not (actual < value):
                return False
            elif op == "gt" and not (actual > value):
                return False
            elif op == "lte" and not (actual <= value):
                return False
            elif op == "gte" and not (actual >= value):
                return False
            elif op == "contains" and str(value) not in str(actual):
                return False
            elif op == "in" and actual not in value:
                return False

        return True

    def _do_sort(self, records: list["AuditRecord"]) -> list["AuditRecord"]:
        """Sort records by a field."""
        field = self._order_by
        return sorted(
            records,
            key=lambda r: getattr(r, field, 0) or 0,  # type: ignore[arg-type]
            reverse=self._order_desc,
        )

    def _do_group(self, records: list["AuditRecord"]) -> list[GroupResult]:
        """Group records by a field and compute per-group stats."""
        from collections import defaultdict

        buckets: dict[str, list["AuditRecord"]] = defaultdict(list)
        for r in records:
            key_val = str(getattr(r, self._group_key, "unknown"))  # type: ignore[arg-type]
            buckets[key_val].append(r)

        results = []
        for val, recs in sorted(buckets.items()):
            confs = [r.confidence for r in recs if r.confidence is not None]
            lats = [r.latency_ms for r in recs]
            results.append(GroupResult(
                key=self._group_key or "",
                value=val,
                count=len(recs),
                avg_confidence=sum(confs) / len(confs) if confs else None,
                avg_latency_ms=sum(lats) / len(lats) if lats else 0,
            ))

        return results
