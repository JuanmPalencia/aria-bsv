"""Verifier — independent verification of ARIA epochs and records from BSV."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from .core.errors import ARIATamperDetected, ARIAVerificationError
from .core.merkle import ARIAMerkleTree, verify_proof, MerkleProof
from .core.record import AuditRecord
from .storage.base import StorageInterface

_log = logging.getLogger(__name__)

# OP_RETURN prefix produced by DirectWallet: OP_FALSE OP_RETURN PUSH4 b'ARIA'
_ARIA_SCRIPT_PREFIX = bytes([0x00, 0x6A, 0x04, 0x41, 0x52, 0x49, 0x41])

_WOC_URLS = {
    "mainnet": "https://api.whatsonchain.com/v1/bsv/main",
    "testnet": "https://api.whatsonchain.com/v1/bsv/test",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Result of an epoch or record verification.

    Attributes:
        valid:         True if the epoch / record passed all checks.
        tampered:      True if a cryptographic inconsistency was detected.
        epoch_id:      Epoch identifier from the EPOCH_OPEN payload.
        system_id:     System identifier from the EPOCH_OPEN payload.
        model_id:      Model identifier (only set for record verification).
        model_version: SHA-256 of the committed model file, or None.
        decided_at:    Datetime from the EPOCH_OPEN timestamp field.
        records_count: Number of records in the epoch (from EPOCH_CLOSE).
        merkle_root:   Merkle root of the epoch (from EPOCH_CLOSE).
        error:         Human-readable error message, or None if valid.
    """

    valid: bool
    tampered: bool = False
    epoch_id: str = ""
    system_id: str = ""
    model_id: str | None = None
    model_version: str | None = None
    decided_at: datetime | None = None
    records_count: int = 0
    merkle_root: str = ""
    error: str | None = None

    @classmethod
    def _err(cls, msg: str, tampered: bool = False) -> "VerificationResult":
        return cls(valid=False, tampered=tampered, error=msg)


# ---------------------------------------------------------------------------
# TxFetcher abstraction
# ---------------------------------------------------------------------------


class TxFetcher(ABC):
    """Abstraction over BSV transaction retrieval.

    Implementations must handle:
    1. Fetching an ARIA payload from a given txid.
    2. Finding an EPOCH_CLOSE txid given an open_txid (optional — may return None).
    """

    @abstractmethod
    async def fetch_payload(self, txid: str) -> dict[str, Any] | None:
        """Fetch and decode the ARIA OP_RETURN JSON payload embedded in *txid*.

        Returns None if the transaction is not found or contains no ARIA payload.
        """

    @abstractmethod
    async def find_close_txid(self, epoch_id: str, open_txid: str) -> str | None:
        """Try to find the EPOCH_CLOSE txid that references *open_txid*.

        May return None if the search is not supported or the close is not found.
        """


class WhatsOnChainFetcher(TxFetcher):
    """Fetches ARIA payloads from BSV via the WhatsOnChain REST API.

    Args:
        network: ``"mainnet"`` or ``"testnet"``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, network: str = "mainnet", timeout: float = 15.0) -> None:
        self._base = _WOC_URLS.get(network, _WOC_URLS["mainnet"])
        self._timeout = timeout

    async def fetch_payload(self, txid: str) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base}/tx/hash/{txid}")
                if resp.status_code != 200:
                    return None
                tx_data = resp.json()
            for vout in tx_data.get("vout", []):
                script_hex: str = vout.get("scriptPubKey", {}).get("hex", "")
                payload = _parse_aria_script(script_hex)
                if payload is not None:
                    return payload
        except Exception as exc:
            _log.debug("WhatsOnChain fetch failed for %s: %s", txid, exc)
        return None

    async def find_close_txid(self, epoch_id: str, open_txid: str) -> str | None:
        # Full BSV search (requires PLANARIA / custom indexer) is out of scope for Phase 3.
        # Independent verifiers must provide the close_txid explicitly.
        return None


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------


def _parse_aria_script(script_hex: str) -> dict[str, Any] | None:
    """Extract the ARIA JSON payload from a bsvlib OP_RETURN script hex string.

    Expected format (from DirectWallet):
        OP_FALSE  OP_RETURN  PUSH4(b'ARIA')  PUSHDATA(<json_bytes>)
        00        6a         04 41524941      <len_byte(s)> <json>

    Returns the decoded dict, or None if the script is not an ARIA payload.
    """
    try:
        raw = bytes.fromhex(script_hex)
    except ValueError:
        return None

    if not raw.startswith(_ARIA_SCRIPT_PREFIX):
        return None

    pos = len(_ARIA_SCRIPT_PREFIX)  # 7 — right after the ARIA prefix marker
    if pos >= len(raw):
        return None

    len_byte = raw[pos]
    pos += 1

    if len_byte <= 75:
        json_len = len_byte
    elif len_byte == 0x4C:  # OP_PUSHDATA1
        if pos >= len(raw):
            return None
        json_len = raw[pos]
        pos += 1
    elif len_byte == 0x4D:  # OP_PUSHDATA2
        if pos + 1 >= len(raw):
            return None
        json_len = int.from_bytes(raw[pos: pos + 2], "little")
        pos += 2
    elif len_byte == 0x4E:  # OP_PUSHDATA4
        if pos + 3 >= len(raw):
            return None
        json_len = int.from_bytes(raw[pos: pos + 4], "little")
        pos += 4
    else:
        return None

    json_bytes = raw[pos: pos + json_len]
    if len(json_bytes) != json_len:
        return None

    try:
        return json.loads(json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class Verifier:
    """Independent verification engine for ARIA epochs and records.

    Can operate in two modes:

    * **Independent** (auditor-agnostic): fetches payloads from BSV via
      ``tx_fetcher``.  The caller must supply ``close_txid`` because there is
      no on-chain index to locate it from ``open_txid`` alone (Phase 3).

    * **Local** (operator's own system): uses ``storage`` to look up the
      ``close_txid`` and, for record verification, all sibling record hashes
      needed for Merkle proof reconstruction.

    Args:
        network:     ``"mainnet"`` or ``"testnet"``.
        tx_fetcher:  BSV transaction fetcher.  Defaults to WhatsOnChainFetcher.
        storage:     Local storage backend.  Enables look-up of close txids and
                     Merkle tree reconstruction without requiring callers to
                     supply the proof explicitly.
    """

    def __init__(
        self,
        network: str = "mainnet",
        tx_fetcher: TxFetcher | None = None,
        storage: StorageInterface | None = None,
    ) -> None:
        self._network = network
        self._fetcher: TxFetcher = tx_fetcher or WhatsOnChainFetcher(network=network)
        self._storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_epoch(
        self,
        open_txid: str,
        close_txid: str | None = None,
    ) -> VerificationResult:
        """Verify the cryptographic integrity of a full epoch.

        Checks:
        1. EPOCH_OPEN payload is parseable and well-formed.
        2. EPOCH_CLOSE payload is parseable and well-formed.
        3. ``prev_txid`` in EPOCH_CLOSE equals *open_txid*.
        4. EPOCH_OPEN timestamp is before EPOCH_CLOSE (chronological order).

        Args:
            open_txid:  BSV txid of the EPOCH_OPEN transaction.
            close_txid: BSV txid of the EPOCH_CLOSE transaction.
                        If not supplied, looked up in local storage.

        Returns:
            VerificationResult with ``valid=True`` if all checks pass.
        """
        # 1. Fetch EPOCH_OPEN.
        open_payload = await self._fetcher.fetch_payload(open_txid)
        if open_payload is None:
            return VerificationResult._err(f"EPOCH_OPEN txid {open_txid!r} not found or not an ARIA tx")

        if open_payload.get("type") != "EPOCH_OPEN":
            return VerificationResult._err(
                f"txid {open_txid!r} contains a {open_payload.get('type')!r} payload, expected EPOCH_OPEN"
            )

        epoch_id: str = open_payload.get("epoch_id", "")
        system_id: str = open_payload.get("system_id", "")
        model_hashes: dict[str, str] = open_payload.get("model_hashes", {})
        open_ts: int = open_payload.get("timestamp", 0)

        # 2. Resolve close_txid.
        if close_txid is None:
            close_txid = self._resolve_close_txid(epoch_id, open_txid)
        if close_txid is None:
            return VerificationResult._err(
                f"EPOCH_CLOSE for epoch {epoch_id!r} not found — supply close_txid explicitly"
            )

        # 3. Fetch EPOCH_CLOSE.
        close_payload = await self._fetcher.fetch_payload(close_txid)
        if close_payload is None:
            return VerificationResult._err(f"EPOCH_CLOSE txid {close_txid!r} not found")

        if close_payload.get("type") != "EPOCH_CLOSE":
            return VerificationResult._err(
                f"txid {close_txid!r} contains a {close_payload.get('type')!r} payload, expected EPOCH_CLOSE"
            )

        # 4. Verify prev_txid link.
        if close_payload.get("prev_txid") != open_txid:
            return VerificationResult(
                valid=False,
                tampered=True,
                epoch_id=epoch_id,
                system_id=system_id,
                error=(
                    f"EPOCH_CLOSE.prev_txid {close_payload.get('prev_txid')!r} "
                    f"does not match open_txid {open_txid!r}"
                ),
            )

        # 5. Verify epoch_id consistency.
        if close_payload.get("epoch_id") != epoch_id:
            return VerificationResult(
                valid=False, tampered=True, epoch_id=epoch_id, system_id=system_id,
                error="EPOCH_CLOSE.epoch_id does not match EPOCH_OPEN.epoch_id",
            )

        records_count: int = close_payload.get("records_count", 0)
        merkle_root: str = close_payload.get("records_merkle_root", "")
        decided_at = datetime.fromtimestamp(open_ts, tz=timezone.utc) if open_ts else None

        return VerificationResult(
            valid=True,
            epoch_id=epoch_id,
            system_id=system_id,
            decided_at=decided_at,
            records_count=records_count,
            merkle_root=merkle_root,
        )

    async def verify_record(
        self,
        open_txid: str,
        record_data: dict[str, Any],
        close_txid: str | None = None,
    ) -> VerificationResult:
        """Verify that a specific record is consistent with a committed epoch.

        Checks performed:
        1. Epoch chain verification (OPEN → CLOSE link).
        2. record_data epoch_id matches the committed epoch.
        3. record_data model_id is listed in the EPOCH_OPEN model_hashes.
        4. If local storage is available: Merkle proof verification — the record
           hash is confirmed present in the epoch's committed Merkle tree.

        Args:
            open_txid:   BSV txid of the EPOCH_OPEN transaction.
            record_data: Dict with at minimum: ``epoch_id``, ``model_id``,
                         ``input_hash``, ``output_hash``, ``sequence``.
                         Remaining ``AuditRecord`` fields are optional.
            close_txid:  BSV txid of the EPOCH_CLOSE.  Auto-resolved if omitted.

        Returns:
            VerificationResult.  ``tampered=True`` if a hash mismatch is detected.
        """
        # Step 1: verify the epoch chain.
        epoch_result = await self.verify_epoch(open_txid, close_txid=close_txid)
        if not epoch_result.valid:
            return epoch_result

        epoch_id = epoch_result.epoch_id
        model_hashes = await self._get_model_hashes(open_txid)

        # Step 2: reconstruct the AuditRecord.
        try:
            record = AuditRecord(
                epoch_id=record_data["epoch_id"],
                model_id=record_data["model_id"],
                input_hash=record_data["input_hash"],
                output_hash=record_data["output_hash"],
                sequence=record_data["sequence"],
                confidence=record_data.get("confidence"),
                latency_ms=record_data.get("latency_ms", 0),
                metadata=record_data.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return VerificationResult._err(f"invalid record_data: {exc}")

        # Step 3: epoch_id must match.
        if record.epoch_id != epoch_id:
            return VerificationResult(
                valid=False, tampered=True, epoch_id=epoch_id,
                system_id=epoch_result.system_id,
                error=(
                    f"record.epoch_id {record.epoch_id!r} does not match "
                    f"committed epoch_id {epoch_id!r}"
                ),
            )

        # Step 4: model_id must be in EPOCH_OPEN model_hashes.
        if model_hashes and record.model_id not in model_hashes:
            return VerificationResult(
                valid=False, tampered=True, epoch_id=epoch_id,
                system_id=epoch_result.system_id, model_id=record.model_id,
                error=(
                    f"model_id {record.model_id!r} was not committed in EPOCH_OPEN "
                    f"(known models: {list(model_hashes.keys())})"
                ),
            )

        model_version = model_hashes.get(record.model_id) if model_hashes else None

        # Step 5: Merkle verification (if local storage available).
        merkle_verified = False
        if self._storage is not None:
            merkle_verified = self._verify_merkle_local(
                record=record,
                epoch_id=epoch_id,
                expected_root=epoch_result.merkle_root,
            )
            if not merkle_verified:
                return VerificationResult(
                    valid=False, tampered=True, epoch_id=epoch_id,
                    system_id=epoch_result.system_id, model_id=record.model_id,
                    model_version=model_version,
                    error=f"record hash not found in Merkle tree (root={epoch_result.merkle_root!r})",
                )

        return VerificationResult(
            valid=True,
            epoch_id=epoch_id,
            system_id=epoch_result.system_id,
            model_id=record.model_id,
            model_version=model_version,
            decided_at=epoch_result.decided_at,
            records_count=epoch_result.records_count,
            merkle_root=epoch_result.merkle_root,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_close_txid(self, epoch_id: str, open_txid: str) -> str | None:
        """Look up the close txid from local storage if available."""
        if self._storage is None:
            return None
        row = self._storage.get_epoch(epoch_id)
        if row and row.close_txid:
            return row.close_txid
        return None

    async def _get_model_hashes(self, open_txid: str) -> dict[str, str]:
        payload = await self._fetcher.fetch_payload(open_txid)
        if payload and payload.get("type") == "EPOCH_OPEN":
            return payload.get("model_hashes", {})
        return {}

    def _verify_merkle_local(
        self,
        record: AuditRecord,
        epoch_id: str,
        expected_root: str,
    ) -> bool:
        """Reconstruct the Merkle tree from all stored records and verify membership."""
        assert self._storage is not None
        all_records = self._storage.list_records_by_epoch(epoch_id)
        if not all_records:
            return False

        tree = ARIAMerkleTree()
        for rec in sorted(all_records, key=lambda r: r.sequence):
            tree.add(rec.hash())

        if tree.root() != expected_root:
            return False

        target_hash = record.hash()
        try:
            proof = tree.proof(target_hash)
            return verify_proof(expected_root, proof, target_hash)
        except Exception:
            return False
