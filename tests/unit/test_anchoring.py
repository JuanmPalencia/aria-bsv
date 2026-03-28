"""Tests for ARIA multi-chain OP_RETURN anchoring."""

from __future__ import annotations

import struct

import pytest

from aria.anchoring import (
    AnchorPayload,
    AnchorResult,
    BTCBroadcaster,
    MultiChainAnchor,
    build_op_return_payload,
    decode_op_return_payload,
    _build_op_return_script,
)


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

class TestAnchorPayload:
    def test_from_epoch_close(self):
        p = AnchorPayload.from_epoch_close("epoch-abc", "a" * 64, 42)
        assert p.epoch_id == "epoch-abc"
        assert p.merkle_root == "a" * 64
        assert p.records_count == 42

    def test_canonical_bytes_length_is_72(self):
        p = AnchorPayload.from_epoch_close("epoch-xyz", "b" * 64, 7)
        data = p.canonical_bytes()
        assert len(data) == 72

    def test_canonical_bytes_starts_with_magic(self):
        p = AnchorPayload.from_epoch_close("ep", "0" * 64, 1)
        data = p.canonical_bytes()
        assert data[:4] == b"ARIA"

    def test_canonical_bytes_version_byte(self):
        p = AnchorPayload.from_epoch_close("ep", "0" * 64, 1)
        data = p.canonical_bytes()
        assert data[4] == 0x01

    def test_records_count_encoded_big_endian(self):
        p = AnchorPayload.from_epoch_close("ep", "0" * 64, 12345)
        data = p.canonical_bytes()
        count = struct.unpack(">I", data[53:57])[0]
        assert count == 12345

    def test_digest_is_sha256_of_canonical_bytes(self):
        import hashlib
        p = AnchorPayload.from_epoch_close("ep-digest", "c" * 64, 3)
        expected = hashlib.sha256(p.canonical_bytes()).hexdigest()
        assert p.digest() == expected

    def test_epoch_id_encoded_in_bytes(self):
        epoch_id = "my-epoch-id"
        p = AnchorPayload.from_epoch_close(epoch_id, "0" * 64, 1)
        data = p.canonical_bytes()
        epoch_bytes = data[5:21].rstrip(b"\x00")
        assert epoch_bytes.decode() == epoch_id[:16]

    def test_long_epoch_id_truncated(self):
        long_id = "a" * 100
        p = AnchorPayload.from_epoch_close(long_id, "0" * 64, 1)
        data = p.canonical_bytes()
        # Should not raise; epoch field is exactly 16 bytes
        assert len(data[5:21]) == 16

    def test_merkle_root_with_sha256_prefix(self):
        p = AnchorPayload.from_epoch_close("ep", "sha256:" + "e" * 64, 1)
        # Should not raise — the prefix is stripped
        data = p.canonical_bytes()
        assert len(data) == 72


# ---------------------------------------------------------------------------
# build_op_return_payload
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def test_output_is_72_bytes(self):
        p = AnchorPayload.from_epoch_close("ep", "f" * 64, 100)
        assert len(build_op_return_payload(p)) == 72

    def test_deterministic(self):
        p = AnchorPayload.from_epoch_close("ep-det", "1" * 64, 5)
        assert build_op_return_payload(p) == build_op_return_payload(p)


# ---------------------------------------------------------------------------
# decode_op_return_payload
# ---------------------------------------------------------------------------

class TestDecodePayload:
    def test_round_trip(self):
        p = AnchorPayload.from_epoch_close("ep-rt", "a" * 64, 99)
        raw = p.canonical_bytes()
        decoded = decode_op_return_payload(raw)
        assert decoded is not None
        assert decoded.epoch_id == "ep-rt"
        assert decoded.records_count == 99

    def test_wrong_magic_returns_none(self):
        data = b"BADM" + b"\x00" * 68
        assert decode_op_return_payload(data) is None

    def test_short_data_returns_none(self):
        assert decode_op_return_payload(b"ARIA") is None

    def test_wrong_version_returns_none(self):
        p = AnchorPayload.from_epoch_close("ep", "0" * 64, 1)
        raw = bytearray(p.canonical_bytes())
        raw[4] = 0x02  # Wrong version
        assert decode_op_return_payload(bytes(raw)) is None


# ---------------------------------------------------------------------------
# OP_RETURN script builder
# ---------------------------------------------------------------------------

class TestBuildOpReturnScript:
    def test_short_data_no_pushdata1(self):
        data = b"\x01" * 20
        script = _build_op_return_script(data)
        assert script[0] == 0x6A  # OP_RETURN
        assert script[1] == 20    # length byte
        assert script[2:] == data

    def test_long_data_uses_pushdata1(self):
        data = b"\x02" * 76
        script = _build_op_return_script(data)
        assert script[0] == 0x6A  # OP_RETURN
        assert script[1] == 0x4C  # OP_PUSHDATA1
        assert script[2] == 76
        assert script[3:] == data

    def test_72_byte_payload_uses_direct_length(self):
        """Standard ARIA payload (72 bytes ≤ 75) uses direct length byte, not PUSHDATA1."""
        p = AnchorPayload.from_epoch_close("ep", "0" * 64, 1)
        script = _build_op_return_script(p.canonical_bytes())
        assert script[0] == 0x6A  # OP_RETURN
        assert script[1] == 72    # Direct length byte (no PUSHDATA1 needed for ≤75 bytes)


# ---------------------------------------------------------------------------
# BTCBroadcaster stub
# ---------------------------------------------------------------------------

class TestBTCBroadcaster:
    @pytest.mark.asyncio
    async def test_broadcast_without_wif_returns_error(self):
        broadcaster = BTCBroadcaster()
        payload = AnchorPayload.from_epoch_close("ep-btc", "0" * 64, 5)
        result = await broadcaster.broadcast(payload.canonical_bytes())

        assert isinstance(result, AnchorResult)
        assert result.chain == "btc"
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# MultiChainAnchor
# ---------------------------------------------------------------------------

class TestMultiChainAnchor:
    @pytest.mark.asyncio
    async def test_broadcast_all_returns_one_result_per_chain(self):
        anchor = MultiChainAnchor(secondary_chains=["btc"])
        payload = AnchorPayload.from_epoch_close("ep-mc", "0" * 64, 3)
        results = await anchor.broadcast_all(payload)

        assert len(results) == 1
        assert results[0].chain == "btc"

    @pytest.mark.asyncio
    async def test_unsupported_chain_returns_error(self):
        anchor = MultiChainAnchor(secondary_chains=["solana"])
        payload = AnchorPayload.from_epoch_close("ep-sol", "0" * 64, 1)
        results = await anchor.broadcast_all(payload)

        assert len(results) == 1
        assert results[0].success is False
        assert "not yet supported" in (results[0].error or "")

    @pytest.mark.asyncio
    async def test_multiple_chains_concurrent(self):
        anchor = MultiChainAnchor(secondary_chains=["btc", "solana"])
        payload = AnchorPayload.from_epoch_close("ep-multi", "0" * 64, 2)
        results = await anchor.broadcast_all(payload)

        assert len(results) == 2
        chains = {r.chain for r in results}
        assert "btc" in chains
        assert "solana" in chains


# ---------------------------------------------------------------------------
# BTCBroadcaster — full implementation tests
# ---------------------------------------------------------------------------

from aria.anchoring import (
    _varint, _hash256, _hash160, _p2pkh_script,
    _base58_decode, _decode_wif, _privkey_to_pubkey, _build_btc_tx,
)


class TestBTCInternalHelpers:
    def test_varint_single_byte(self):
        assert _varint(0) == b"\x00"
        assert _varint(252) == b"\xfc"

    def test_varint_fd_prefix(self):
        enc = _varint(253)
        assert enc[0:1] == b"\xfd"
        assert len(enc) == 3

    def test_hash256_known_value(self):
        import hashlib
        result = _hash256(b"")
        expected = hashlib.sha256(hashlib.sha256(b"").digest()).digest()
        assert result == expected

    def test_hash160_known_value(self):
        import hashlib
        result = _hash160(b"\x00")
        h1 = hashlib.sha256(b"\x00").digest()
        h2 = hashlib.new("ripemd160", h1).digest()
        assert result == h2

    def test_p2pkh_script_is_25_bytes(self):
        fake_pubkey = b"\x02" + b"\x11" * 32
        script = _p2pkh_script(fake_pubkey)
        assert len(script) == 25
        assert script[0] == 0x76   # OP_DUP
        assert script[-1] == 0xAC  # OP_CHECKSIG

    def test_decode_wif_compressed(self):
        # Known mainnet compressed WIF (test vector)
        # Private key: 0x01 (trivial, for testing)
        # We use a WIF derived from a known key
        import base64
        import hashlib
        # Build a valid WIF manually
        privkey = b"\x01" * 32
        payload = b"\x80" + privkey + b"\x01"  # compressed
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        raw = payload + checksum
        # Base58 encode
        ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        n = int.from_bytes(raw, "big")
        result = ""
        while n:
            n, r = divmod(n, 58)
            result = ALPHABET[r] + result
        wif = result
        decoded_key, compressed = _decode_wif(wif)
        assert decoded_key == privkey
        assert compressed is True

    def test_privkey_to_pubkey_returns_bytes(self):
        privkey = b"\x01" * 32
        pub = _privkey_to_pubkey(privkey, compressed=True)
        assert isinstance(pub, bytes)
        assert len(pub) == 33  # compressed secp256k1 pubkey

    def test_privkey_to_pubkey_uncompressed(self):
        privkey = b"\x01" * 32
        pub = _privkey_to_pubkey(privkey, compressed=False)
        assert isinstance(pub, bytes)
        assert len(pub) == 65  # uncompressed secp256k1 pubkey
        assert pub[0] == 0x04


class TestBTCBroadcasterNoKey:
    @pytest.mark.asyncio
    async def test_broadcast_without_wif_returns_error(self):
        broadcaster = BTCBroadcaster()
        payload = AnchorPayload.from_epoch_close("ep-btc", "0" * 64, 5)
        result = await broadcaster.broadcast(payload.canonical_bytes())
        assert result.success is False
        assert result.error is not None
        assert "wif" in result.error.lower() or "broadcast_signed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_broadcast_signed_network_error(self):
        broadcaster = BTCBroadcaster(api_url="http://localhost:19999")
        result = await broadcaster.broadcast_signed("deadbeef")
        assert result.success is False
        assert result.chain == "btc"

    def test_invalid_wif_raises(self):
        import pytest
        with pytest.raises((ValueError, Exception)):
            BTCBroadcaster(wif="not-a-valid-wif")


class TestBTCBroadcasterWithKey:
    def _make_wif(self, privkey: bytes) -> str:
        import hashlib
        payload = b"\x80" + privkey + b"\x01"
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        raw = payload + checksum
        ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        n = int.from_bytes(raw, "big")
        result = ""
        while n:
            n, r = divmod(n, 58)
            result = ALPHABET[r] + result
        return result

    def test_broadcaster_initializes_with_valid_wif(self):
        wif = self._make_wif(b"\x11" * 32)
        b = BTCBroadcaster(wif=wif)
        assert b._privkey == b"\x11" * 32

    def test_derive_address_returns_string(self):
        wif = self._make_wif(b"\x11" * 32)
        b = BTCBroadcaster(wif=wif)
        addr = b._derive_address()
        assert isinstance(addr, str)
        assert addr.startswith("1")  # mainnet P2PKH

    @pytest.mark.asyncio
    async def test_broadcast_no_utxos_returns_error(self, respx_mock=None):
        pytest.importorskip("respx")
        import respx, httpx
        wif = self._make_wif(b"\x11" * 32)
        broadcaster = BTCBroadcaster(wif=wif, api_url="https://blockstream.info/api")
        addr = broadcaster._derive_address()
        with respx.mock:
            respx.get(f"https://blockstream.info/api/address/{addr}/utxo").mock(
                return_value=httpx.Response(200, json=[])
            )
            payload = AnchorPayload.from_epoch_close("ep", "0" * 64, 1)
            result = await broadcaster.broadcast(payload.canonical_bytes())
        assert result.success is False
        assert "utxo" in (result.error or "").lower()

    def test_build_btc_tx_produces_valid_hex(self):
        privkey = b"\x11" * 32
        data = b"ARIA" + b"\x00" * 68
        raw_hex = _build_btc_tx(
            utxo_txid="a" * 64,
            utxo_vout=0,
            utxo_satoshis=100_000,
            op_return_data=data[:72],
            privkey=privkey,
            compressed=True,
            fee_sat=2000,
        )
        assert isinstance(raw_hex, str)
        assert len(raw_hex) > 0
        tx_bytes = bytes.fromhex(raw_hex)
        # Check version (first 4 bytes, LE) = 1
        import struct
        version = struct.unpack("<I", tx_bytes[:4])[0]
        assert version == 1
