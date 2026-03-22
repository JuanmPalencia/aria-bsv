"""SQLiteStorage — zero-config local storage backend using SQLAlchemy."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import StaticPool

from ..core.errors import ARIAStorageError
from ..core.record import AuditRecord
from .base import EpochRow, StorageInterface


class _Base(DeclarativeBase):
    pass


class _EpochTable(_Base):
    __tablename__ = "aria_epochs"

    epoch_id = Column(String, primary_key=True)
    system_id = Column(String, nullable=False)
    open_txid = Column(String, nullable=False, default="")
    close_txid = Column(String, nullable=False, default="")
    state_hash = Column(String, nullable=False, default="")
    model_hashes_json = Column(Text, nullable=False, default="{}")
    opened_at = Column(Integer, nullable=False, default=0)
    closed_at = Column(Integer, nullable=False, default=0)
    records_count = Column(Integer, nullable=False, default=0)
    merkle_root = Column(String, nullable=False, default="")


class _RecordTable(_Base):
    __tablename__ = "aria_records"

    record_id = Column(String, primary_key=True)
    epoch_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    input_hash = Column(String, nullable=False)
    output_hash = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=False, default=0)
    sequence = Column(Integer, nullable=False)
    metadata_json = Column(Text, nullable=False, default="{}")
    aria_version = Column(String, nullable=False)
    record_hash = Column(String, nullable=False)
    created_at = Column(Integer, nullable=False)


def _row_to_audit_record(row: _RecordTable) -> AuditRecord:
    meta: dict[str, Any] = json.loads(row.metadata_json or "{}")
    rec = AuditRecord(
        epoch_id=row.epoch_id,
        model_id=row.model_id,
        input_hash=row.input_hash,
        output_hash=row.output_hash,
        sequence=row.sequence,
        confidence=row.confidence,
        latency_ms=row.latency_ms or 0,
        metadata=meta,
    )
    return rec


class SQLiteStorage(StorageInterface):
    """Thread-safe SQLite storage backend.

    Uses a single SQLAlchemy engine with ``check_same_thread=False`` and a
    per-call ``Session`` so that both the main thread (receipts) and the
    background BatchManager thread (writes) can use it concurrently.

    Args:
        dsn: SQLAlchemy DSN.  Examples:
             ``"sqlite:///aria.db"``  — file-based  (relative path)
             ``"sqlite:////abs/path/aria.db"``  — absolute path
             ``"sqlite://"``         — in-memory (for tests)
    """

    def __init__(self, dsn: str = "sqlite:///aria.db") -> None:
        connect_args: dict[str, Any] = {}
        kwargs: dict[str, Any] = {}
        if dsn.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            # In-memory SQLite must share a single connection across threads.
            if dsn in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool

        self._engine = create_engine(dsn, connect_args=connect_args, **kwargs)
        _Base.metadata.create_all(self._engine)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # StorageInterface implementation
    # ------------------------------------------------------------------

    def save_record(self, record: AuditRecord) -> None:
        with self._lock, Session(self._engine) as session:
            try:
                row = _RecordTable(
                    record_id=record.record_id,
                    epoch_id=record.epoch_id,
                    model_id=record.model_id,
                    input_hash=record.input_hash,
                    output_hash=record.output_hash,
                    confidence=record.confidence,
                    latency_ms=record.latency_ms,
                    sequence=record.sequence,
                    metadata_json=json.dumps(record.metadata, sort_keys=True),
                    aria_version=record.aria_version,
                    record_hash=record.hash(),
                    created_at=int(time.time()),
                )
                session.add(row)
                session.commit()
            except Exception as exc:
                session.rollback()
                raise ARIAStorageError(f"save_record failed: {exc}") from exc

    def save_epoch_open(
        self,
        epoch_id: str,
        system_id: str,
        open_txid: str,
        model_hashes: dict[str, str],
        state_hash: str,
        opened_at: int,
    ) -> None:
        with self._lock, Session(self._engine) as session:
            try:
                row = _EpochTable(
                    epoch_id=epoch_id,
                    system_id=system_id,
                    open_txid=open_txid,
                    close_txid="",
                    state_hash=state_hash,
                    model_hashes_json=json.dumps(model_hashes, sort_keys=True),
                    opened_at=opened_at,
                    closed_at=0,
                    records_count=0,
                    merkle_root="",
                )
                session.add(row)
                session.commit()
            except Exception as exc:
                session.rollback()
                raise ARIAStorageError(f"save_epoch_open failed: {exc}") from exc

    def save_epoch_close(
        self,
        epoch_id: str,
        close_txid: str,
        merkle_root: str,
        records_count: int,
        closed_at: int,
    ) -> None:
        with self._lock, Session(self._engine) as session:
            try:
                row = session.get(_EpochTable, epoch_id)
                if row is None:
                    raise ARIAStorageError(f"epoch {epoch_id!r} not found in storage")
                row.close_txid = close_txid
                row.merkle_root = merkle_root
                row.records_count = records_count
                row.closed_at = closed_at
                session.commit()
            except ARIAStorageError:
                raise
            except Exception as exc:
                session.rollback()
                raise ARIAStorageError(f"save_epoch_close failed: {exc}") from exc

    def get_record(self, record_id: str) -> AuditRecord | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(_RecordTable, record_id)
            if row is None:
                return None
            return _row_to_audit_record(row)

    def get_epoch(self, epoch_id: str) -> EpochRow | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(_EpochTable, epoch_id)
            if row is None:
                return None
            return EpochRow(
                epoch_id=row.epoch_id,
                system_id=row.system_id,
                open_txid=row.open_txid,
                close_txid=row.close_txid,
                state_hash=row.state_hash,
                model_hashes=json.loads(row.model_hashes_json or "{}"),
                opened_at=row.opened_at,
                closed_at=row.closed_at,
                records_count=row.records_count,
                merkle_root=row.merkle_root,
            )

    def list_records_by_epoch(self, epoch_id: str) -> list[AuditRecord]:
        with self._lock, Session(self._engine) as session:
            rows = (
                session.query(_RecordTable)
                .filter(_RecordTable.epoch_id == epoch_id)
                .order_by(_RecordTable.sequence)
                .all()
            )
            return [_row_to_audit_record(r) for r in rows]
