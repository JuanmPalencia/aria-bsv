"""EpochManager — Pre-Commitment Protocol lifecycle (BRC-121)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import ARIAError
from .hasher import hash_object
from .merkle import ARIAMerkleTree
from .record import AuditRecord

from typing import Any

if TYPE_CHECKING:
    from ..wallet.base import WalletInterface
    from ..broadcaster.base import BroadcasterInterface

ARIA_VERSION = "1.0"

# SHA-256 of the empty string — used as Merkle root for zero-record epochs.
_EMPTY_ROOT = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@dataclass
class EpochConfig:
    """Minimal configuration required by EpochManager.

    In Phase 2 this dataclass will be superseded by the full ``AuditConfig``
    defined in ``aria.auditor``.  For Phase 1, EpochConfig is the primary
    entry point for configuring epoch lifecycle.

    Attributes:
        system_id: Unique identifier for the AI system registered in ARIA.
        network:   ``"mainnet"`` or ``"testnet"``.
    """

    system_id: str
    network: str = "mainnet"


@dataclass
class EpochOpenResult:
    """Confirmed result of an EPOCH_OPEN broadcast.

    Attributes:
        epoch_id:      Unique identifier for this epoch.
        txid:          BSV transaction ID of the EPOCH_OPEN transaction.
        timestamp:     Unix timestamp (seconds) at which open_epoch was called.
        model_hashes:  Mapping of model_id → sha256 hash committed to BSV.
        state_hash:    SHA-256 of the canonical system_state dict.
        nonce:         16-byte random hex string included in the payload.
    """

    epoch_id: str
    txid: str
    timestamp: int
    model_hashes: dict[str, str]
    state_hash: str
    nonce: str


@dataclass
class EpochCloseResult:
    """Confirmed result of an EPOCH_CLOSE broadcast.

    Attributes:
        epoch_id:      Identifier of the epoch that was closed.
        txid:          BSV transaction ID of the EPOCH_CLOSE transaction.
        prev_txid:     txid of the corresponding EPOCH_OPEN (the link).
        merkle_root:   SHA-256 Merkle root of all record hashes in the epoch.
        records_count: Number of AuditRecords included.
        duration_ms:   Epoch duration in milliseconds (close_ts - open_ts).
    """

    epoch_id: str
    txid: str
    prev_txid: str
    merkle_root: str
    records_count: int
    duration_ms: int


class EpochManager:
    """Manages the OPEN / CLOSE lifecycle of ARIA pre-commitment epochs.

    Each epoch follows the BRC-121 Pre-Commitment Protocol:

    1. ``open_epoch(model_hashes, system_state)`` — builds and broadcasts an
       ``EPOCH_OPEN`` transaction to BSV **before** any model executes.
       The txid is stored internally.

    2. ``close_epoch(epoch_id, records)`` — builds and broadcasts an
       ``EPOCH_CLOSE`` transaction whose ``prev_txid`` links back to the
       ``EPOCH_OPEN``.  The Merkle root of all record hashes is included.

    This chain guarantees that no record can be backdated: the committed
    model hashes and state hash predate all inference records in the epoch.

    Args:
        config:      System-level configuration (system_id, network).
        wallet:      Wallet implementation for signing transactions.
        broadcaster: Broadcaster implementation for network delivery.
    """

    def __init__(
        self,
        config: EpochConfig,
        wallet: "WalletInterface",
        broadcaster: "BroadcasterInterface",
    ) -> None:
        self._config = config
        self._wallet = wallet
        self._broadcaster = broadcaster
        self._counter: int = 0
        # epoch_id -> EpochOpenResult for all currently open (not yet closed) epochs.
        self._open_epochs: dict[str, EpochOpenResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def open_epoch(
        self,
        model_hashes: dict[str, str],
        system_state: dict,  # type: ignore[type-arg]
    ) -> EpochOpenResult:
        """Publish an EPOCH_OPEN transaction committing model hashes and system state.

        Must be called **before** any model executes in the epoch.
        After this call returns, the commitment is immutable in BSV.

        Args:
            model_hashes:  ``{model_id: "sha256:<hex>"}`` for every model that
                           will run during this epoch.  All keys must appear in
                           subsequent ``AuditRecord.model_id`` fields.
            system_state:  Operational context of the system at epoch start.
                           Must be JSON-serialisable.  Do NOT include user PII.

        Returns:
            ``EpochOpenResult`` with the epoch_id and confirmed txid.

        Raises:
            ARIAWalletError:    If transaction signing fails.
            ARIABroadcastError: If the BSV network rejects the transaction.
        """
        ts_ms = int(time.time() * 1000)
        epoch_id = self._next_epoch_id(ts_ms)
        state_hash = hash_object(system_state)
        nonce = os.urandom(16).hex()

        payload = {
            "aria_version": ARIA_VERSION,
            "type": "EPOCH_OPEN",
            "epoch_id": epoch_id,
            "system_id": self._config.system_id,
            "model_hashes": model_hashes,
            "state_hash": state_hash,
            "timestamp": ts_ms // 1000,
            "nonce": nonce,
        }

        txid = await self._wallet.sign_and_broadcast(payload)

        result = EpochOpenResult(
            epoch_id=epoch_id,
            txid=txid,
            timestamp=ts_ms // 1000,
            model_hashes=model_hashes,
            state_hash=state_hash,
            nonce=nonce,
        )
        self._open_epochs[epoch_id] = result
        return result

    async def close_epoch(
        self,
        epoch_id: str,
        records: list[AuditRecord],
        statement: Any | None = None,
    ) -> EpochCloseResult:
        """Publish an EPOCH_CLOSE transaction linking back to the EPOCH_OPEN.

        The Merkle root is computed from the record hashes sorted by their
        ``sequence`` field.  An epoch with zero records uses the SHA-256 of
        the empty string as its Merkle root.

        Args:
            epoch_id: Must match an epoch previously opened via ``open_epoch``.
            records:  All ``AuditRecord`` objects collected during the epoch.
                      May be empty (zero-record epoch is valid).

        Returns:
            ``EpochCloseResult`` with the confirmed txid and Merkle root.

        Raises:
            ARIAError:          If *epoch_id* was never opened (or already closed).
            ARIAWalletError:    If transaction signing fails.
            ARIABroadcastError: If the BSV network rejects the transaction.
        """
        if epoch_id not in self._open_epochs:
            raise ARIAError(f"epoch {epoch_id!r} was never opened or has already been closed")

        open_result = self._open_epochs[epoch_id]

        merkle_root = self._compute_merkle_root(records)

        now_ms = int(time.time() * 1000)
        duration_ms = now_ms - (open_result.timestamp * 1000)

        payload: dict[str, Any] = {
            "aria_version": ARIA_VERSION,
            "type": "EPOCH_CLOSE",
            "epoch_id": epoch_id,
            "prev_txid": open_result.txid,
            "records_merkle_root": merkle_root,
            "records_count": len(records),
            "duration_ms": duration_ms,
        }

        # ZK extension: embed statement commitment in the on-chain payload.
        if statement is not None:
            payload["zk"] = statement.to_bsv_payload()

        txid = await self._wallet.sign_and_broadcast(payload)

        # Remove from open epochs — this epoch is now immutably closed.
        del self._open_epochs[epoch_id]

        return EpochCloseResult(
            epoch_id=epoch_id,
            txid=txid,
            prev_txid=open_result.txid,
            merkle_root=merkle_root,
            records_count=len(records),
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_epoch_id(self, timestamp_ms: int) -> str:
        """Generate the next monotonically increasing epoch_id."""
        self._counter += 1
        return f"ep_{timestamp_ms}_{self._counter:04d}"

    @staticmethod
    def _compute_merkle_root(records: list[AuditRecord]) -> str:
        """Return the Merkle root of record hashes sorted by sequence."""
        if not records:
            return _EMPTY_ROOT
        tree = ARIAMerkleTree()
        for rec in sorted(records, key=lambda r: r.sequence):
            tree.add(rec.hash())
        return tree.root()
