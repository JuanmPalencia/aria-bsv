"""
aria.anchoring — Multi-chain OP_RETURN anchoring for ARIA epochs.

Anchors the ARIA epoch close commitment to an additional chain (Bitcoin BTC)
alongside the primary BSV broadcast.  Both chains record the same canonical
payload via OP_RETURN, creating independent proof-of-existence at the same
block height (approximately).

Protocol payload (identical on both chains)::

    OP_RETURN  ARIA  <version:1>  <epoch_id:16>  <merkle_root:32>  <records_count:4>

Where:
  - ``ARIA``        — 4-byte magic prefix
  - ``<version>``   — 0x01 = BRC-121 v1
  - ``<epoch_id>``  — first 16 bytes of epoch_id UTF-8 (zero-padded)
  - ``<merkle_root>`` — 32 raw bytes of the Merkle root hash
  - ``<records_count>`` — 4-byte big-endian record count

Usage::

    from aria.anchoring import AnchorPayload, build_op_return_payload
    from aria.anchoring import MultiChainAnchor, BTCBroadcaster

    payload = AnchorPayload.from_epoch_close(epoch_id, merkle_root, records_count)
    raw = build_op_return_payload(payload)

    anchor = MultiChainAnchor(
        btc_broadcaster=BTCBroadcaster(api_url="https://blockstream.info/api"),
        secondary_chains=["btc"],
    )
    results = await anchor.broadcast_all(payload, btc_funding_utxo=utxo)
"""

from __future__ import annotations

import asyncio
import hashlib
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------

_MAGIC = b"ARIA"
_VERSION = 0x01


@dataclass
class AnchorPayload:
    """The canonical data committed to each chain via OP_RETURN.

    Attributes:
        epoch_id:      Full epoch UUID string (stored; first 16 bytes go on-chain).
        merkle_root:   Hex-encoded Merkle root (64 hex chars = 32 bytes).
        records_count: Number of records in the epoch.
        system_id:     System identifier (metadata only, not on-chain).
    """
    epoch_id: str
    merkle_root: str          # hex string, 64 chars
    records_count: int
    system_id: str = ""

    @classmethod
    def from_epoch_close(
        cls,
        epoch_id: str,
        merkle_root: str,
        records_count: int,
        system_id: str = "",
    ) -> "AnchorPayload":
        return cls(
            epoch_id=epoch_id,
            merkle_root=merkle_root,
            records_count=records_count,
            system_id=system_id,
        )

    def canonical_bytes(self) -> bytes:
        """Return the full OP_RETURN data bytes (72 bytes total)."""
        return build_op_return_payload(self)

    def digest(self) -> str:
        """SHA-256 of canonical_bytes() — used for cross-chain verification."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_op_return_payload(payload: AnchorPayload) -> bytes:
    """Encode the AnchorPayload into the on-chain OP_RETURN byte string.

    Layout (72 bytes):
        4  bytes — magic "ARIA"
        1  byte  — version 0x01
        16 bytes — epoch_id prefix (UTF-8, zero-padded)
        32 bytes — merkle_root raw bytes
        4  bytes — records_count (big-endian uint32)
        15 bytes — zero padding (reserved for future use)

    Returns:
        72-byte bytes object ready to embed as OP_RETURN data.
    """
    epoch_bytes = payload.epoch_id.encode()[:16].ljust(16, b"\x00")

    # Merkle root: accept either hex string or raw bytes
    if isinstance(payload.merkle_root, str):
        mr = payload.merkle_root.lstrip("sha256:").lstrip("0x")
        # Pad to 64 hex chars if shorter (e.g., empty string)
        mr_bytes = bytes.fromhex(mr.zfill(64))[:32]
    else:
        mr_bytes = bytes(payload.merkle_root)[:32]

    count_bytes = struct.pack(">I", payload.records_count)
    padding = b"\x00" * 15

    return _MAGIC + bytes([_VERSION]) + epoch_bytes + mr_bytes + count_bytes + padding


def decode_op_return_payload(data: bytes) -> AnchorPayload | None:
    """Attempt to decode an OP_RETURN payload.  Returns None if not ARIA-formatted."""
    if len(data) < 53 or data[:4] != _MAGIC:
        return None
    if data[4] != _VERSION:
        return None
    epoch_id_bytes = data[5:21].rstrip(b"\x00")
    mr_bytes = data[21:53]
    count_bytes = data[53:57]

    epoch_id = epoch_id_bytes.decode(errors="replace")
    merkle_root = mr_bytes.hex()
    records_count = struct.unpack(">I", count_bytes)[0]
    return AnchorPayload(epoch_id=epoch_id, merkle_root=merkle_root, records_count=records_count)


# ---------------------------------------------------------------------------
# BTC broadcaster (Blockstream.info / Mempool.space compatible)
# ---------------------------------------------------------------------------

@dataclass
class AnchorResult:
    """Result of an OP_RETURN broadcast attempt."""
    chain: str                  # "btc", "bsv", etc.
    txid: str | None = None
    success: bool = False
    error: str | None = None


class BTCBroadcaster:
    """Broadcast OP_RETURN transactions to Bitcoin via HTTP APIs.

    Supports Blockstream.info and Mempool.space API formats.

    NOTE: This class builds a *minimal* raw transaction carrying a single
    OP_RETURN output.  It requires a funded UTXO passed at broadcast time.
    For production use, integrate with a proper BTC wallet library.

    Args:
        api_url:    Base URL of the broadcast API.  Must support
                    ``POST /tx`` accepting a raw hex transaction.
        network:    ``"mainnet"`` or ``"testnet"`` (used for logging only).
    """

    def __init__(
        self,
        api_url: str = "https://blockstream.info/api",
        network: str = "mainnet",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._network = network

    async def broadcast(self, op_return_data: bytes) -> AnchorResult:
        """Broadcast *op_return_data* as an OP_RETURN transaction.

        This is a stub implementation that constructs the OP_RETURN script and
        posts it to the API.  A full implementation requires a funded UTXO and
        signing infrastructure.

        Returns an AnchorResult with ``success=False`` and a descriptive error
        when no signing key is configured (e.g., in testing contexts).
        """
        script = _build_op_return_script(op_return_data)
        # In a full implementation, we would sign the tx with a funded UTXO here.
        # Returning a stub result to keep this dependency-free in tests.
        return AnchorResult(
            chain="btc",
            txid=None,
            success=False,
            error=(
                "BTCBroadcaster.broadcast() is a stub — "
                "provide a signing key and funded UTXO for production use"
            ),
        )

    async def broadcast_signed(
        self,
        raw_tx_hex: str,
    ) -> AnchorResult:
        """Broadcast a fully-signed raw transaction hex string.

        Args:
            raw_tx_hex: Signed raw transaction as a hex string.

        Returns:
            AnchorResult with txid on success.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._api_url}/tx", content=raw_tx_hex)
                if resp.status_code == 200:
                    txid = resp.text.strip()
                    return AnchorResult(chain="btc", txid=txid, success=True)
                return AnchorResult(
                    chain="btc",
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as exc:
            return AnchorResult(chain="btc", success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Multi-chain anchor coordinator
# ---------------------------------------------------------------------------

class MultiChainAnchor:
    """Coordinates OP_RETURN broadcasts across multiple chains.

    Fires all broadcasts concurrently using asyncio.gather so that latency on
    one chain does not block others.

    Args:
        btc_broadcaster:    A BTCBroadcaster instance for Bitcoin anchoring.
        secondary_chains:   List of chain identifiers to anchor.  Currently
                            only ``"btc"`` is supported.
    """

    def __init__(
        self,
        btc_broadcaster: BTCBroadcaster | None = None,
        secondary_chains: list[str] | None = None,
    ) -> None:
        self._btc = btc_broadcaster or BTCBroadcaster()
        self._chains = set(secondary_chains or ["btc"])

    async def broadcast_all(self, payload: AnchorPayload) -> list[AnchorResult]:
        """Broadcast to all configured secondary chains concurrently.

        Returns a list of AnchorResult objects, one per chain.

        Example::

            results = await anchor.broadcast_all(payload)
            for r in results:
                if r.success:
                    print(f"{r.chain}: txid={r.txid}")
                else:
                    print(f"{r.chain}: FAILED — {r.error}")
        """
        op_return_data = payload.canonical_bytes()
        tasks = []
        for chain in self._chains:
            if chain == "btc":
                tasks.append(self._btc.broadcast(op_return_data))
            else:
                tasks.append(_unsupported_chain(chain))

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)


async def _unsupported_chain(chain: str) -> AnchorResult:
    return AnchorResult(chain=chain, success=False, error=f"Chain {chain!r} not yet supported")


# ---------------------------------------------------------------------------
# OP_RETURN script builder
# ---------------------------------------------------------------------------

def _build_op_return_script(data: bytes) -> bytes:
    """Build a Bitcoin OP_RETURN script for *data*.

    Format: OP_RETURN <PUSHDATA> <data>
    OP_RETURN = 0x6a, PUSHDATA1 = 0x4c for data > 75 bytes.
    """
    OP_RETURN = 0x6A
    if len(data) <= 75:
        return bytes([OP_RETURN, len(data)]) + data
    else:
        OP_PUSHDATA1 = 0x4C
        return bytes([OP_RETURN, OP_PUSHDATA1, len(data)]) + data
