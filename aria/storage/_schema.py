"""aria.storage._schema — shared SQLAlchemy ORM models for all ARIA storage backends.

Extracting models here lets both SQLiteStorage and PostgreSQLStorage share the same
DeclarativeBase and metadata, which is required for Alembic auto-generation to work.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase

from ..core.record import AuditRecord


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
    epoch_id = Column(String, nullable=False, index=True)
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


class _ZKProofTable(_Base):
    __tablename__ = "aria_zk_proofs"

    record_id = Column(String, primary_key=True)
    epoch_id = Column(String, nullable=False, index=True)
    proof_hex = Column(Text, nullable=False)
    public_inputs_json = Column(Text, nullable=False, default="[]")
    proving_system = Column(String, nullable=False)
    tier = Column(String, nullable=False)
    model_hash = Column(String, nullable=False)
    prover_version = Column(String, nullable=False)
    proof_digest = Column(String, nullable=False)


class _VKTable(_Base):
    __tablename__ = "aria_vk_keys"

    model_hash = Column(String, primary_key=True)
    vk_bytes = Column(LargeBinary, nullable=False)
    proving_system = Column(String, nullable=False)
    vk_digest = Column(String, nullable=False)


def _row_to_audit_record(row: _RecordTable) -> AuditRecord:
    meta: dict[str, Any] = json.loads(row.metadata_json or "{}")
    return AuditRecord(
        epoch_id=row.epoch_id,
        model_id=row.model_id,
        input_hash=row.input_hash,
        output_hash=row.output_hash,
        sequence=row.sequence,
        confidence=row.confidence,
        latency_ms=row.latency_ms or 0,
        metadata=meta,
    )
