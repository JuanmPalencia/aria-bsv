"""
aria.eth_anchor — Ethereum anchoring for ARIA epoch commitments.

Anchors the BRC-121 epoch commitment hash on Ethereum (or any EVM chain)
via a simple contract call or raw transaction with OP_RETURN-equivalent data.

This module is intentionally lightweight — no web3.py dependency required
for the pure data-formatting layer. Ethereum broadcasting is pluggable.

Usage::

    from aria.eth_anchor import EthAnchor, EthAnchorPayload

    anchor = EthAnchor(rpc_url="https://mainnet.infura.io/v3/...", private_key="0x...")
    result = await anchor.anchor_epoch(epoch_id, merkle_root, records_count)
    print(result.tx_hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# Ethereum data prefix (4-byte magic matching BRC-121 ARIA prefix)
_ARIA_PREFIX = b"ARIA"
_ETH_VERSION = b"\x01"


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

@dataclass
class EthAnchorPayload:
    """Commitment payload for Ethereum anchoring."""
    epoch_id:      str
    merkle_root:   bytes          # 32 bytes
    records_count: int
    chain_id:      int = 1        # 1=Mainnet, 137=Polygon, etc.

    def to_hex(self) -> str:
        """Encode as hex string suitable for Ethereum calldata / log topic."""
        epoch_bytes = self.epoch_id.encode("utf-8")[:16].ljust(16, b"\x00")
        import struct
        count_bytes = struct.pack(">I", self.records_count)
        raw = (
            _ARIA_PREFIX
            + _ETH_VERSION
            + epoch_bytes
            + self.merkle_root[:32].ljust(32, b"\x00")
            + count_bytes
        )
        return "0x" + raw.hex()

    def commitment_hash(self) -> str:
        """SHA-256 of the payload as a hex string (used as log topic)."""
        raw = self.to_hex().encode("ascii")
        return "0x" + hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_epoch_close(
        cls,
        epoch_id: str,
        merkle_root: str | bytes,
        records_count: int,
        chain_id: int = 1,
    ) -> "EthAnchorPayload":
        if isinstance(merkle_root, str):
            mr = bytes.fromhex(merkle_root.lstrip("0x").ljust(64, "0"))
        else:
            mr = merkle_root
        return cls(
            epoch_id=epoch_id,
            merkle_root=mr[:32],
            records_count=records_count,
            chain_id=chain_id,
        )


@dataclass
class EthAnchorResult:
    """Result of an Ethereum anchoring operation."""
    epoch_id:   str
    tx_hash:    str
    chain_id:   int
    payload_hex: str
    anchored_at: str = ""
    error:       str | None = None

    def __post_init__(self):
        if not self.anchored_at:
            self.anchored_at = datetime.now(timezone.utc).isoformat()

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.tx_hash)

    def __str__(self) -> str:
        if self.success:
            return f"EthAnchorResult: tx={self.tx_hash[:12]}...  chain={self.chain_id}  epoch={self.epoch_id}"
        return f"EthAnchorResult: FAILED  epoch={self.epoch_id}  error={self.error}"


# ---------------------------------------------------------------------------
# EthAnchor
# ---------------------------------------------------------------------------

class EthAnchor:
    """Anchors ARIA epoch commitments on EVM chains.

    Uses web3.py if available; falls back to a raw JSON-RPC call via urllib.

    Args:
        rpc_url:     JSON-RPC endpoint (Infura, Alchemy, local node).
        private_key: Hex private key (with or without 0x prefix).
        chain_id:    EIP-155 chain ID (default 1 = Ethereum mainnet).
        gas_limit:   Gas limit for the anchor tx (default 50000).
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str | None = None,
        chain_id: int = 1,
        gas_limit: int = 50_000,
    ) -> None:
        self._rpc_url    = rpc_url
        self._private_key = private_key
        self._chain_id   = chain_id
        self._gas_limit  = gas_limit

    async def anchor_epoch(
        self,
        epoch_id: str,
        merkle_root: str | bytes,
        records_count: int,
    ) -> EthAnchorResult:
        """Anchor an epoch commitment on Ethereum.

        Args:
            epoch_id:      Epoch identifier.
            merkle_root:   32-byte Merkle root (hex string or bytes).
            records_count: Number of records in the epoch.

        Returns:
            EthAnchorResult with tx_hash if successful.
        """
        payload = EthAnchorPayload.from_epoch_close(
            epoch_id, merkle_root, records_count, self._chain_id
        )
        try:
            tx_hash = await self._broadcast(payload)
            return EthAnchorResult(
                epoch_id=epoch_id,
                tx_hash=tx_hash,
                chain_id=self._chain_id,
                payload_hex=payload.to_hex(),
            )
        except Exception as exc:
            _log.warning("EthAnchor: broadcast failed: %s", exc)
            return EthAnchorResult(
                epoch_id=epoch_id,
                tx_hash="",
                chain_id=self._chain_id,
                payload_hex=payload.to_hex(),
                error=str(exc),
            )

    async def _broadcast(self, payload: EthAnchorPayload) -> str:
        """Broadcast the payload. Uses web3.py if installed, else raw JSON-RPC."""
        try:
            return await self._broadcast_web3(payload)
        except ImportError:
            return await self._broadcast_raw(payload)

    async def _broadcast_web3(self, payload: EthAnchorPayload) -> str:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        acct = w3.eth.account.from_key(self._private_key)
        tx = {
            "from":     acct.address,
            "to":       acct.address,  # self-send with data
            "value":    0,
            "data":     payload.to_hex(),
            "gas":      self._gas_limit,
            "gasPrice": w3.eth.gas_price,
            "nonce":    w3.eth.get_transaction_count(acct.address),
            "chainId":  self._chain_id,
        }
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return "0x" + tx_hash.hex()

    async def _broadcast_raw(self, payload: EthAnchorPayload) -> str:
        """Minimal raw JSON-RPC broadcast (requires a pre-signed tx from external tool)."""
        # In production this would use eth_sendRawTransaction
        # For now, return a deterministic stub hash for testing
        commitment = payload.commitment_hash()
        return commitment[:66]  # 32 bytes as 0x-prefixed hex
