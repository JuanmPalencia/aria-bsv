"""Tests for aria.query — Fluent query API for audit records."""

from __future__ import annotations

import time
import pytest

from aria.core.record import AuditRecord
from aria.storage.sqlite import SQLiteStorage
from aria.query import RecordQuery, QueryStats, GroupResult, _parse_duration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_storage_with_records():
    """Create an in-memory SQLite storage with sample records."""
    storage = SQLiteStorage(dsn="sqlite://")
    epoch_id = "test-epoch"
    now_s = int(time.time())

    storage.save_epoch_open(
        epoch_id=epoch_id,
        system_id="test-system",
        open_txid="tx_open_" + "a" * 56,
        model_hashes={"gpt-4": "sha256:" + "a" * 64},
        state_hash="sha256:" + "b" * 64,
        opened_at=now_s,
    )

    for i in range(10):
        storage.save_record(AuditRecord(
            epoch_id=epoch_id,
            model_id="gpt-4" if i < 7 else "gpt-3.5",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i+100:064x}",
            sequence=i,
            confidence=0.5 + i * 0.05,
            latency_ms=100 + i * 50,
        ))

    return storage, epoch_id


class TestParseDuration:
    def test_minutes(self):
        assert _parse_duration("30m") == 1800

    def test_hours(self):
        assert _parse_duration("24h") == 86400

    def test_days(self):
        assert _parse_duration("7d") == 604800

    def test_weeks(self):
        assert _parse_duration("2w") == 1209600

    def test_raw_seconds(self):
        assert _parse_duration("3600") == 3600


class TestRecordQuery:
    def test_execute_returns_all_records(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).execute()
        assert len(results) == 10

    def test_model_filter(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).model("gpt-4").execute()
        assert len(results) == 7
        assert all(r.model_id == "gpt-4" for r in results)

    def test_where_confidence_lt(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(confidence__lt=0.7).execute()
        assert all(r.confidence < 0.7 for r in results)

    def test_where_confidence_gt(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(confidence__gt=0.8).execute()
        assert all(r.confidence > 0.8 for r in results)

    def test_where_confidence_gte(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(confidence__gte=0.5).execute()
        assert len(results) == 10

    def test_where_latency_gt(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(latency_ms__gt=300).execute()
        assert all(r.latency_ms > 300 for r in results)

    def test_where_exact_match(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(model_id="gpt-3.5").execute()
        assert len(results) == 3

    def test_where_contains(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).where(model_id__contains="gpt").execute()
        assert len(results) == 10

    def test_limit(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).limit(3).execute()
        assert len(results) == 3

    def test_chaining_immutable(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        q1 = q.epoch(epoch_id)
        q2 = q1.model("gpt-4")
        # q1 should not be affected
        assert len(q1.execute()) == 10
        assert len(q2.execute()) == 7

    def test_count(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        assert q.epoch(epoch_id).count() == 10
        assert q.epoch(epoch_id).model("gpt-4").count() == 7

    def test_first(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        result = q.epoch(epoch_id).first()
        assert result is not None
        assert isinstance(result, AuditRecord)

    def test_first_empty(self):
        storage = SQLiteStorage(dsn="sqlite://")
        q = RecordQuery(storage)
        result = q.epoch("nonexistent").first()
        assert result is None

    def test_order_by(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).order_by("latency_ms", desc=True).execute()
        lats = [r.latency_ms for r in results]
        assert lats == sorted(lats, reverse=True)

    def test_order_by_asc(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        results = q.epoch(epoch_id).order_by("confidence", desc=False).execute()
        confs = [r.confidence for r in results]
        assert confs == sorted(confs)


class TestQueryStats:
    def test_stats_basic(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        stats = q.epoch(epoch_id).stats()
        assert isinstance(stats, QueryStats)
        assert stats.count == 10
        assert stats.avg_confidence is not None
        assert stats.min_confidence is not None
        assert stats.max_confidence is not None
        assert stats.avg_latency_ms > 0
        assert stats.p95_latency_ms > 0
        assert len(stats.models) == 2
        assert len(stats.epochs) == 1

    def test_stats_empty(self):
        storage = SQLiteStorage(dsn="sqlite://")
        q = RecordQuery(storage)
        stats = q.epoch("nonexistent").stats()
        assert stats.count == 0
        assert stats.avg_confidence is None

    def test_stats_to_dict(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        d = q.epoch(epoch_id).stats().to_dict()
        assert "count" in d
        assert "avg_confidence" in d
        assert "p95_latency_ms" in d


class TestGroupBy:
    def test_group_by_model(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        groups = q.epoch(epoch_id).group_by("model_id").execute()
        assert isinstance(groups, list)
        assert len(groups) == 2
        assert all(isinstance(g, GroupResult) for g in groups)

    def test_group_counts(self):
        storage, epoch_id = _make_storage_with_records()
        q = RecordQuery(storage)
        groups = q.epoch(epoch_id).group_by("model_id").execute()
        counts = {g.value: g.count for g in groups}
        assert counts["gpt-4"] == 7
        assert counts["gpt-3.5"] == 3
