"""
aria.offline — Offline mode for ARIA.

Records inferences locally without attempting BSV broadcasts.
Later, ``aria sync`` publishes all pending epochs to the blockchain.

Usage::

    from aria.offline import OfflineAuditor

    auditor = OfflineAuditor("my-system", db="aria_dev.db")
    auditor.record("gpt-4", {"prompt": "hi"}, {"text": "hello"}, confidence=0.95)
    auditor.close_epoch()

    # Later, when network is available:
    from aria.offline import sync
    sync(auditor.storage, broadcaster, wallet)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .core.hasher import hash_object
from .core.merkle import ARIAMerkleTree
from .core.record import AuditRecord
from .storage.base import StorageInterface
from .storage.sqlite import SQLiteStorage

_log = logging.getLogger(__name__)


@dataclass
class OfflineEpochResult:
    """Result of closing an offline epoch."""

    epoch_id: str
    records_count: int
    merkle_root: str
    synced: bool = False
    open_txid: str = ""
    close_txid: str = ""


class OfflineAuditor:
    """Records inferences locally without any BSV broadcast.

    Epochs are stored with ``open_txid = "offline:pending"`` and can be
    synced later via the ``sync()`` function or ``aria sync`` CLI command.

    Args:
        system_id:   Unique system identifier.
        db:          SQLite DSN or file path.
        model_hashes: Model hash mapping (optional).
    """

    def __init__(
        self,
        system_id: str,
        db: str = "sqlite:///aria_offline.db",
        model_hashes: dict[str, str] | None = None,
    ) -> None:
        self._system_id = system_id
        self._model_hashes = model_hashes or {}
        dsn = db if db.startswith("sqlite") else f"sqlite:///{db}"
        self._storage = SQLiteStorage(dsn=dsn)
        self._current_epoch_id: str | None = None
        self._sequence = 0
        self._tree = ARIAMerkleTree()
        self._records: list[AuditRecord] = []
        self._open_epoch()

    @property
    def storage(self) -> StorageInterface:
        return self._storage

    @property
    def current_epoch_id(self) -> str | None:
        return self._current_epoch_id

    @property
    def record_count(self) -> int:
        return self._sequence

    def _open_epoch(self) -> str:
        epoch_id = f"offline-{uuid.uuid4().hex[:12]}"
        self._current_epoch_id = epoch_id
        self._sequence = 0
        self._tree = ARIAMerkleTree()
        self._records = []

        state_hash = hash_object({"system_id": self._system_id, "mode": "offline"})
        self._storage.save_epoch_open(
            epoch_id=epoch_id,
            system_id=self._system_id,
            open_txid="offline:pending",
            model_hashes=self._model_hashes,
            state_hash=state_hash,
            opened_at=int(time.time()),
        )
        _log.info("Opened offline epoch: %s", epoch_id)
        return epoch_id

    def record(
        self,
        model_id: str,
        input_data: Any,
        output_data: Any,
        confidence: float | None = None,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an inference locally.

        Returns:
            The record_id.
        """
        if self._current_epoch_id is None:
            self._open_epoch()

        input_hash = hash_object(input_data)
        output_hash = hash_object(output_data)

        rec = AuditRecord(
            epoch_id=self._current_epoch_id,
            model_id=model_id,
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=self._sequence,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._storage.save_record(rec)
        self._tree.add(rec.hash())
        self._records.append(rec)
        self._sequence += 1

        return rec.record_id

    def close_epoch(self) -> OfflineEpochResult:
        """Close the current epoch and compute the Merkle root.

        The epoch is stored with ``close_txid = "offline:pending"``
        and awaits sync.
        """
        if self._current_epoch_id is None:
            raise RuntimeError("No open epoch to close")

        merkle_root = self._tree.root() if self._sequence > 0 else ""

        self._storage.save_epoch_close(
            epoch_id=self._current_epoch_id,
            close_txid="offline:pending",
            merkle_root=merkle_root,
            records_count=self._sequence,
            closed_at=int(time.time()),
        )

        result = OfflineEpochResult(
            epoch_id=self._current_epoch_id,
            records_count=self._sequence,
            merkle_root=merkle_root,
        )

        _log.info(
            "Closed offline epoch %s: %d records, root=%s",
            self._current_epoch_id, self._sequence, merkle_root[:16] + "...",
        )

        self._current_epoch_id = None
        self._open_epoch()
        return result


def list_pending(storage: StorageInterface) -> list[str]:
    """List epoch IDs that are pending sync (offline:pending)."""
    all_epochs = storage.list_epochs(limit=10000)
    return [
        e.epoch_id
        for e in all_epochs
        if e.open_txid == "offline:pending" or e.close_txid == "offline:pending"
    ]


async def sync_epoch(
    epoch_id: str,
    storage: StorageInterface,
    wallet: Any,
    broadcaster: Any,
) -> dict[str, str]:
    """Sync a single offline epoch to BSV.

    Broadcasts EPOCH_OPEN and EPOCH_CLOSE transactions.

    Returns:
        Dict with open_txid and close_txid.
    """
    from .core.epoch import EpochConfig, EpochManager

    epoch = storage.get_epoch(epoch_id)
    if epoch is None:
        raise ValueError(f"Epoch {epoch_id} not found")

    records = storage.list_records_by_epoch(epoch_id)
    _log.info("Syncing epoch %s (%d records) to BSV...", epoch_id, len(records))

    manager = EpochManager(
        config=EpochConfig(system_id=epoch.system_id),
        wallet=wallet,
        broadcaster=broadcaster,
        storage=storage,
    )

    open_result = await manager.open_epoch(
        model_hashes=epoch.model_hashes,
        state_hash=epoch.state_hash,
    )

    tree = ARIAMerkleTree()
    for rec in records:
        tree.add(rec.hash())

    close_result = await manager.close_epoch(
        epoch_id=open_result.epoch_id,
        records=records,
    )

    result = {
        "epoch_id": epoch_id,
        "new_epoch_id": open_result.epoch_id,
        "open_txid": open_result.txid,
        "close_txid": close_result.txid,
        "records_synced": len(records),
    }
    _log.info("Epoch %s synced: open=%s close=%s", epoch_id, open_result.txid[:16], close_result.txid[:16])
    return result


async def sync_all(
    storage: StorageInterface,
    wallet: Any,
    broadcaster: Any,
) -> list[dict[str, str]]:
    """Sync all pending offline epochs to BSV."""
    pending = list_pending(storage)
    results = []
    for epoch_id in pending:
        try:
            result = await sync_epoch(epoch_id, storage, wallet, broadcaster)
            results.append(result)
        except Exception as exc:
            _log.error("Failed to sync epoch %s: %s", epoch_id, exc)
            results.append({"epoch_id": epoch_id, "error": str(exc)})
    return results
