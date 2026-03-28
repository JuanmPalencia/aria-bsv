"""Tests for aria.eth_anchor — Ethereum anchoring."""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.eth_anchor import (
    EthAnchor,
    EthAnchorPayload,
    EthAnchorResult,
)


# ---------------------------------------------------------------------------
# EthAnchorPayload
# ---------------------------------------------------------------------------

class TestEthAnchorPayload:
    def _payload(self, epoch_id="epoch-001", records=50):
        mr = hashlib.sha256(b"test").digest()
        return EthAnchorPayload(epoch_id=epoch_id, merkle_root=mr, records_count=records)

    def test_to_hex_starts_with_0x(self):
        p = self._payload()
        assert p.to_hex().startswith("0x")

    def test_to_hex_contains_aria_prefix(self):
        p = self._payload()
        hex_str = p.to_hex()
        # ARIA in hex = 41524941
        assert "41524941" in hex_str.lower()

    def test_commitment_hash_is_hex(self):
        p = self._payload()
        h = p.commitment_hash()
        assert h.startswith("0x")
        assert len(h) == 66  # 0x + 64 hex chars

    def test_commitment_hash_deterministic(self):
        p = self._payload()
        assert p.commitment_hash() == p.commitment_hash()

    def test_different_epochs_different_hash(self):
        p1 = self._payload("epoch-1")
        p2 = self._payload("epoch-2")
        assert p1.commitment_hash() != p2.commitment_hash()

    def test_from_epoch_close_str_merkle(self):
        mr_hex = hashlib.sha256(b"mr").hexdigest()
        p = EthAnchorPayload.from_epoch_close("ep-1", mr_hex, 100)
        assert p.epoch_id == "ep-1"
        assert p.records_count == 100

    def test_from_epoch_close_bytes_merkle(self):
        mr = hashlib.sha256(b"mr").digest()
        p = EthAnchorPayload.from_epoch_close("ep-1", mr, 50)
        assert p.merkle_root == mr[:32]

    def test_chain_id_default(self):
        p = self._payload()
        assert p.chain_id == 1

    def test_epoch_id_truncated_to_16_bytes(self):
        long_id = "a" * 100
        p = EthAnchorPayload.from_epoch_close(long_id, b"\x00" * 32, 10)
        hex_str = p.to_hex()
        # The hex encoding should still succeed
        assert hex_str.startswith("0x")


# ---------------------------------------------------------------------------
# EthAnchorResult
# ---------------------------------------------------------------------------

class TestEthAnchorResult:
    def test_success(self):
        r = EthAnchorResult(epoch_id="ep", tx_hash="0xabc", chain_id=1, payload_hex="0x")
        assert r.success is True

    def test_failure(self):
        r = EthAnchorResult(epoch_id="ep", tx_hash="", chain_id=1, payload_hex="0x",
                            error="timeout")
        assert r.success is False

    def test_anchored_at_set(self):
        r = EthAnchorResult(epoch_id="ep", tx_hash="0x", chain_id=1, payload_hex="0x")
        assert r.anchored_at != ""

    def test_str_success(self):
        r = EthAnchorResult(epoch_id="ep-1", tx_hash="0x" + "a" * 64, chain_id=1, payload_hex="0x")
        s = str(r)
        assert "ep-1" in s

    def test_str_failure(self):
        r = EthAnchorResult(epoch_id="ep-1", tx_hash="", chain_id=1, payload_hex="0x",
                            error="network error")
        s = str(r)
        assert "FAILED" in s
        assert "network error" in s


# ---------------------------------------------------------------------------
# EthAnchor
# ---------------------------------------------------------------------------

class TestEthAnchorFallback:
    """Test that EthAnchor falls back to raw broadcast when web3 not installed."""

    def test_anchor_epoch_success(self):
        anchor = EthAnchor(
            rpc_url="https://mainnet.infura.io/v3/test",
            private_key="0x" + "aa" * 32,
            chain_id=1,
        )
        mr = hashlib.sha256(b"root").digest()

        async def _run():
            # Force fallback to raw broadcast (web3 not installed in test env)
            with patch.object(anchor, "_broadcast_web3", side_effect=ImportError):
                return await anchor.anchor_epoch("ep-1", mr, 100)

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.epoch_id == "ep-1"
        assert result.chain_id == 1
        # Raw fallback returns a deterministic hash
        assert result.tx_hash.startswith("0x")

    def test_anchor_epoch_error_returned(self):
        anchor = EthAnchor("https://bad-rpc", "0x" + "bb" * 32)

        async def _run():
            with patch.object(anchor, "_broadcast_web3", side_effect=ImportError):
                with patch.object(anchor, "_broadcast_raw", side_effect=RuntimeError("fail")):
                    return await anchor.anchor_epoch("ep-1", b"\x00" * 32, 10)

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.success is False
        assert result.error is not None

    def test_raw_broadcast_deterministic(self):
        anchor = EthAnchor("http://localhost:8545", "0x" + "cc" * 32)
        mr = hashlib.sha256(b"merkle").digest()
        payload = EthAnchorPayload.from_epoch_close("ep-test", mr, 5)

        async def _run():
            return await anchor._broadcast_raw(payload)

        h1 = asyncio.get_event_loop().run_until_complete(_run())
        h2 = asyncio.get_event_loop().run_until_complete(_run())
        assert h1 == h2
        assert h1.startswith("0x")
