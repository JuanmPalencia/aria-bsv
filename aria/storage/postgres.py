"""
aria.storage.postgres — PostgreSQL storage backend for ARIA.

Designed for production deployments where SQLite's single-writer limitation
is a bottleneck.  Uses SQLAlchemy's QueuePool for connection pooling and
relies on PostgreSQL's row-level locking (not a Python threading.Lock) for
safe concurrent writes.

Usage::

    from aria.storage.postgres import PostgreSQLStorage

    storage = PostgreSQLStorage(
        dsn="postgresql+psycopg2://user:pass@localhost:5432/aria",
        pool_size=10,
        max_overflow=20,
    )

Migration::

    # Apply schema migrations with Alembic:
    ARIA_DB_URL=postgresql+psycopg2://... alembic upgrade head
"""

from __future__ import annotations

import json
import time
from typing import Any

from sqlalchemy import Engine, create_engine, select, update
from sqlalchemy.orm import Session

from ..core.errors import ARIAStorageError
from ..core.record import AuditRecord
from .base import EpochRow, StorageInterface
from ._schema import (
    _Base,
    _EpochTable,
    _RecordTable,
    _VKTable,
    _ZKProofTable,
    _row_to_audit_record,
)


class PostgreSQLStorage(StorageInterface):
    """PostgreSQL-backed ARIA storage.

    Args:
        dsn:           SQLAlchemy DSN.  Must start with ``postgresql`` or
                       ``postgres``.  Example::

                           postgresql+psycopg2://user:pass@host:5432/aria_db
                           postgresql+asyncpg://...   (sync interface only)

        pool_size:     Number of persistent connections in the pool (default 5).
        max_overflow:  Extra connections allowed above ``pool_size`` (default 10).
        pool_recycle:  Seconds before a connection is recycled (default 3600).
        _engine:       *Testing only* — supply a pre-built SQLAlchemy engine to
                       bypass DSN validation and pool configuration.  This allows
                       unit tests to substitute a SQLite engine without a live
                       PostgreSQL server.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        _engine: Engine | None = None,
    ) -> None:
        if _engine is not None:
            self._engine = _engine
        else:
            if dsn is None:
                raise ValueError("dsn is required when _engine is not provided")
            if not (dsn.startswith("postgresql") or dsn.startswith("postgres")):
                raise ValueError(
                    f"PostgreSQLStorage requires a postgresql:// DSN, got: {dsn!r}"
                )
            self._engine = create_engine(
                dsn,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
            )
        _Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # StorageInterface implementation
    # ------------------------------------------------------------------

    def save_record(self, record: AuditRecord) -> None:
        with Session(self._engine, expire_on_commit=False) as session:
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
        with Session(self._engine, expire_on_commit=False) as session:
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
        with Session(self._engine, expire_on_commit=False) as session:
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
        with Session(self._engine) as session:
            row = session.get(_RecordTable, record_id)
            if row is None:
                return None
            return _row_to_audit_record(row)

    def get_epoch(self, epoch_id: str) -> EpochRow | None:
        with Session(self._engine) as session:
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

    def list_epochs(
        self,
        system_id: str | None = None,
        limit: int = 100,
    ) -> list[EpochRow]:
        with Session(self._engine) as session:
            q = session.query(_EpochTable)
            if system_id is not None:
                q = q.filter(_EpochTable.system_id == system_id)
            rows = q.order_by(_EpochTable.opened_at.desc()).limit(limit).all()
            return [
                EpochRow(
                    epoch_id=r.epoch_id,
                    system_id=r.system_id,
                    open_txid=r.open_txid,
                    close_txid=r.close_txid,
                    state_hash=r.state_hash,
                    model_hashes=json.loads(r.model_hashes_json or "{}"),
                    opened_at=r.opened_at,
                    closed_at=r.closed_at,
                    records_count=r.records_count,
                    merkle_root=r.merkle_root,
                )
                for r in rows
            ]

    def list_records_by_epoch(self, epoch_id: str) -> list[AuditRecord]:
        with Session(self._engine) as session:
            rows = (
                session.query(_RecordTable)
                .filter(_RecordTable.epoch_id == epoch_id)
                .order_by(_RecordTable.sequence)
                .all()
            )
            return [_row_to_audit_record(r) for r in rows]

    # ------------------------------------------------------------------
    # ZK extension methods
    # ------------------------------------------------------------------

    def save_proof(self, proof: "Any") -> None:
        with Session(self._engine, expire_on_commit=False) as session:
            try:
                row = _ZKProofTable(
                    record_id=proof.record_id or proof.epoch_id,
                    epoch_id=proof.epoch_id,
                    proof_hex=proof.proof_bytes.hex(),
                    public_inputs_json=json.dumps(proof.public_inputs),
                    proving_system=proof.proving_system,
                    tier=proof.tier,
                    model_hash=proof.model_hash,
                    prover_version=proof.prover_version,
                    proof_digest=proof.digest(),
                )
                session.merge(row)
                session.commit()
            except Exception as exc:
                session.rollback()
                raise ARIAStorageError(f"save_proof failed: {exc}") from exc

    def get_proof(self, record_id: str) -> "Any":
        from ..zk.base import ZKProof
        with Session(self._engine) as session:
            row = session.get(_ZKProofTable, record_id)
            if row is None:
                return None
            return ZKProof(
                proof_bytes=bytes.fromhex(row.proof_hex),
                public_inputs=json.loads(row.public_inputs_json),
                proving_system=row.proving_system,
                tier=row.tier,
                model_hash=row.model_hash,
                prover_version=row.prover_version,
                epoch_id=row.epoch_id,
                record_id=record_id if record_id != row.epoch_id else None,
            )

    def list_proofs_by_epoch(self, epoch_id: str) -> "list[Any]":
        from ..zk.base import ZKProof
        with Session(self._engine) as session:
            rows = (
                session.query(_ZKProofTable)
                .filter(_ZKProofTable.epoch_id == epoch_id)
                .all()
            )
            return [
                ZKProof(
                    proof_bytes=bytes.fromhex(r.proof_hex),
                    public_inputs=json.loads(r.public_inputs_json),
                    proving_system=r.proving_system,
                    tier=r.tier,
                    model_hash=r.model_hash,
                    prover_version=r.prover_version,
                    epoch_id=r.epoch_id,
                    record_id=r.record_id if r.record_id != r.epoch_id else None,
                )
                for r in rows
            ]

    def save_vk(self, vk: "Any") -> None:
        with Session(self._engine, expire_on_commit=False) as session:
            try:
                row = _VKTable(
                    model_hash=vk.model_hash,
                    vk_bytes=vk.vk_bytes,
                    proving_system=vk.proving_system,
                    vk_digest=vk.digest(),
                )
                session.merge(row)
                session.commit()
            except Exception as exc:
                session.rollback()
                raise ARIAStorageError(f"save_vk failed: {exc}") from exc

    def get_vk(self, model_hash: str) -> "Any":
        from ..zk.base import VerifyingKey
        with Session(self._engine) as session:
            row = session.get(_VKTable, model_hash)
            if row is None:
                return None
            return VerifyingKey(
                vk_bytes=row.vk_bytes,
                model_hash=row.model_hash,
                proving_system=row.proving_system,
            )
