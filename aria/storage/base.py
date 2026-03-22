"""Abstract storage interface for ARIA audit records and epochs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.record import AuditRecord

if TYPE_CHECKING:
    from ..zk.base import VerifyingKey, ZKProof


@dataclass
class EpochRow:
    """Snapshot of a stored epoch.

    Attributes:
        epoch_id:          Unique epoch identifier.
        system_id:         Registered system identifier.
        open_txid:         BSV txid of the EPOCH_OPEN transaction.
        close_txid:        BSV txid of the EPOCH_CLOSE transaction (None if still open).
        state_hash:        SHA-256 of the system state committed in EPOCH_OPEN.
        model_hashes:      Mapping of model_id → sha256 committed in EPOCH_OPEN.
        opened_at:         Unix timestamp (seconds) when the epoch was opened.
        closed_at:         Unix timestamp (seconds) when the epoch was closed (0 if open).
        records_count:     Number of AuditRecords in this epoch.
        merkle_root:       Merkle root of all record hashes (empty string if open).
    """

    epoch_id: str
    system_id: str
    open_txid: str
    close_txid: str
    state_hash: str
    model_hashes: dict[str, str]
    opened_at: int
    closed_at: int
    records_count: int
    merkle_root: str


class StorageInterface(ABC):
    """Abstract base class for ARIA local storage backends.

    Implementations must be thread-safe — both SQLite and PostgreSQL backends
    may be called from the BatchManager background thread while the main thread
    reads receipts concurrently.

    The guarantee: records are persisted via ``save_record`` BEFORE any BSV
    broadcast attempt.  If broadcast fails, the record is not lost.
    """

    @abstractmethod
    def save_record(self, record: AuditRecord) -> None:
        """Persist an AuditRecord.  Called before BSV broadcast."""

    @abstractmethod
    def save_epoch_open(
        self,
        epoch_id: str,
        system_id: str,
        open_txid: str,
        model_hashes: dict[str, str],
        state_hash: str,
        opened_at: int,
    ) -> None:
        """Persist the result of a confirmed EPOCH_OPEN broadcast."""

    @abstractmethod
    def save_epoch_close(
        self,
        epoch_id: str,
        close_txid: str,
        merkle_root: str,
        records_count: int,
        closed_at: int,
    ) -> None:
        """Persist the result of a confirmed EPOCH_CLOSE broadcast."""

    @abstractmethod
    def get_record(self, record_id: str) -> AuditRecord | None:
        """Return the AuditRecord with *record_id*, or None if not found."""

    @abstractmethod
    def get_epoch(self, epoch_id: str) -> EpochRow | None:
        """Return the epoch snapshot for *epoch_id*, or None if not found."""

    @abstractmethod
    def list_records_by_epoch(self, epoch_id: str) -> list[AuditRecord]:
        """Return all AuditRecords belonging to *epoch_id*, ordered by sequence."""

    def list_epochs(
        self,
        system_id: str | None = None,
        limit: int = 100,
    ) -> list["EpochRow"]:
        """Return epochs ordered by opened_at descending.  Default: no-op (returns [])."""
        return []

    # ------------------------------------------------------------------
    # ZK extension methods — optional (default: no-op / return None)
    # ------------------------------------------------------------------

    def save_proof(self, proof: "ZKProof") -> None:
        """Persist a ZKProof.  Default: no-op (ZK not configured)."""

    def get_proof(self, record_id: str) -> "ZKProof | None":
        """Return the ZKProof for *record_id*, or None if not stored."""
        return None

    def list_proofs_by_epoch(self, epoch_id: str) -> "list[ZKProof]":
        """Return all ZKProofs for *epoch_id*."""
        return []

    def save_vk(self, vk: "VerifyingKey") -> None:
        """Persist a VerifyingKey indexed by vk.model_hash.  Default: no-op."""

    def get_vk(self, model_hash: str) -> "VerifyingKey | None":
        """Return the VerifyingKey for *model_hash*, or None if not stored."""
        return None
