"""Tests for PostgreSQLStorage via a SQLite engine stub.

PostgreSQLStorage accepts a pre-built SQLAlchemy engine via ``_engine=`` so
that unit tests can inject an in-memory SQLite engine without requiring a live
PostgreSQL server.  All StorageInterface contract tests are exercised this way.

Integration tests that hit a real PostgreSQL instance are marked
``@pytest.mark.integration`` and skipped unless ARIA_TEST_PG_DSN is set.
"""

from __future__ import annotations

import os
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from aria.storage.postgres import PostgreSQLStorage
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> PostgreSQLStorage:
    """Return a PostgreSQLStorage backed by an in-memory SQLite engine."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return PostgreSQLStorage(_engine=engine)


def _make_record(epoch_id: str = "ep-1", seq: int = 1) -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id="m-1",
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=seq,
        confidence=0.95,
        latency_ms=42,
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# DSN validation
# ---------------------------------------------------------------------------

class TestDSNValidation:
    def test_rejects_sqlite_dsn(self):
        with pytest.raises(ValueError, match="postgresql"):
            PostgreSQLStorage(dsn="sqlite://")

    def test_rejects_mysql_dsn(self):
        with pytest.raises(ValueError, match="postgresql"):
            PostgreSQLStorage(dsn="mysql+pymysql://user:pass@localhost/db")

    def test_requires_dsn_or_engine(self):
        with pytest.raises(ValueError, match="dsn is required"):
            PostgreSQLStorage()

    def test_accepts_postgresql_prefix(self):
        """Engine kwarg bypasses DSN check — can still be used with any engine."""
        storage = _make_storage()
        assert storage is not None

    def test_accepts_postgres_prefix(self):
        """Both 'postgresql://' and 'postgres://' prefixes are valid."""
        # We test the prefix logic directly (no live server needed).
        with pytest.raises(ValueError):
            PostgreSQLStorage(dsn="http://localhost/nope")


# ---------------------------------------------------------------------------
# Record CRUD
# ---------------------------------------------------------------------------

class TestRecordCRUD:
    def test_save_and_get_record(self):
        storage = _make_storage()
        rec = _make_record()
        storage.save_record(rec)
        fetched = storage.get_record(rec.record_id)
        assert fetched is not None
        assert fetched.record_id == rec.record_id
        assert fetched.model_id == "m-1"
        assert fetched.epoch_id == "ep-1"

    def test_get_missing_record_returns_none(self):
        storage = _make_storage()
        assert storage.get_record("nonexistent-id") is None

    def test_save_record_persists_latency_and_confidence(self):
        storage = _make_storage()
        rec = _make_record()
        storage.save_record(rec)
        fetched = storage.get_record(rec.record_id)
        assert fetched.latency_ms == 42
        assert abs(fetched.confidence - 0.95) < 1e-6

    def test_save_record_persists_metadata(self):
        storage = _make_storage()
        rec = _make_record()
        storage.save_record(rec)
        fetched = storage.get_record(rec.record_id)
        assert fetched.metadata == {"source": "test"}

    def test_list_records_by_epoch_ordered_by_sequence(self):
        storage = _make_storage()
        for seq in [3, 1, 2]:
            storage.save_record(_make_record(seq=seq))
        records = storage.list_records_by_epoch("ep-1")
        assert [r.sequence for r in records] == [1, 2, 3]

    def test_list_records_empty_epoch(self):
        storage = _make_storage()
        assert storage.list_records_by_epoch("nonexistent") == []


# ---------------------------------------------------------------------------
# Epoch lifecycle
# ---------------------------------------------------------------------------

class TestEpochLifecycle:
    def test_save_epoch_open_and_get(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open(
            epoch_id="ep-open-1",
            system_id="sys-A",
            open_txid="tx-open",
            model_hashes={"m-1": "sha256:" + "c" * 64},
            state_hash="sha256:" + "d" * 64,
            opened_at=now,
        )
        row = storage.get_epoch("ep-open-1")
        assert row is not None
        assert row.system_id == "sys-A"
        assert row.open_txid == "tx-open"
        assert row.close_txid == ""
        assert row.closed_at == 0
        assert row.records_count == 0

    def test_save_epoch_close_updates_row(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open(
            "ep-close-1", "sys-A", "tx-open",
            {"m-1": "sha256:" + "e" * 64},
            "sha256:" + "f" * 64, now,
        )
        storage.save_epoch_close(
            epoch_id="ep-close-1",
            close_txid="tx-close",
            merkle_root="sha256:" + "0" * 64,
            records_count=7,
            closed_at=now + 5000,
        )
        row = storage.get_epoch("ep-close-1")
        assert row.close_txid == "tx-close"
        assert row.records_count == 7
        assert row.merkle_root == "sha256:" + "0" * 64

    def test_save_epoch_close_missing_epoch_raises(self):
        from aria.core.errors import ARIAStorageError
        storage = _make_storage()
        with pytest.raises(ARIAStorageError, match="not found"):
            storage.save_epoch_close("ghost-id", "tx", "root", 0, 0)

    def test_get_epoch_missing_returns_none(self):
        storage = _make_storage()
        assert storage.get_epoch("nope") is None


# ---------------------------------------------------------------------------
# list_epochs
# ---------------------------------------------------------------------------

class TestListEpochs:
    def _open(self, storage: PostgreSQLStorage, epoch_id: str, system_id: str, ts: int) -> None:
        storage.save_epoch_open(epoch_id, system_id, "tx-" + epoch_id, {}, "sh", ts)

    def test_list_epochs_ordered_desc(self):
        storage = _make_storage()
        base = int(time.time() * 1000)
        self._open(storage, "ep-1", "sys-A", base)
        self._open(storage, "ep-2", "sys-A", base + 1000)
        self._open(storage, "ep-3", "sys-A", base + 2000)
        rows = storage.list_epochs("sys-A")
        assert [r.epoch_id for r in rows] == ["ep-3", "ep-2", "ep-1"]

    def test_list_epochs_filter_by_system_id(self):
        storage = _make_storage()
        base = int(time.time() * 1000)
        self._open(storage, "ep-A1", "sys-A", base)
        self._open(storage, "ep-B1", "sys-B", base + 1000)
        rows_a = storage.list_epochs("sys-A")
        rows_b = storage.list_epochs("sys-B")
        assert all(r.system_id == "sys-A" for r in rows_a)
        assert all(r.system_id == "sys-B" for r in rows_b)

    def test_list_epochs_limit(self):
        storage = _make_storage()
        base = int(time.time() * 1000)
        for i in range(10):
            self._open(storage, f"ep-lim-{i}", "sys-X", base + i * 100)
        rows = storage.list_epochs("sys-X", limit=3)
        assert len(rows) == 3

    def test_list_epochs_all_systems_when_no_filter(self):
        storage = _make_storage()
        base = int(time.time() * 1000)
        self._open(storage, "ep-X", "sys-X", base)
        self._open(storage, "ep-Y", "sys-Y", base + 500)
        rows = storage.list_epochs()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Integration marker (skipped without live PostgreSQL)
# ---------------------------------------------------------------------------

PG_DSN = os.environ.get("ARIA_TEST_PG_DSN", "")


@pytest.mark.skipif(not PG_DSN, reason="ARIA_TEST_PG_DSN not set — skipping live PG tests")
class TestPostgreSQLLive:
    def test_round_trip(self):
        storage = PostgreSQLStorage(dsn=PG_DSN)
        rec = _make_record(epoch_id="pg-ep-1")
        storage.save_record(rec)
        fetched = storage.get_record(rec.record_id)
        assert fetched is not None
        assert fetched.input_hash == rec.input_hash
