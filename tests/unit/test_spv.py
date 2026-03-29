"""Tests for aria.spv — BlockHeader, MerkleBranch, SPV verification."""

from __future__ import annotations

import hashlib
import struct

import pytest

from aria.spv import (
    BlockHeader,
    MerkleBranch,
    SPVError,
    SPVProof,
    SPVVerificationResult,
    _reverse_hex,
    _sha256d,
    verify_header_chain,
    verify_merkle_branch,
    verify_spv_proof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _make_header_bytes(
    version: int = 1,
    prev_block: bytes = b"\x00" * 32,
    merkle_root: bytes = b"\x00" * 32,
    time: int = 1_600_000_000,
    bits: int = 0x207fffff,  # very easy target (regtest-style)
    nonce: int = 0,
) -> bytes:
    """Build a raw 80-byte block header."""
    header = bytearray(80)
    struct.pack_into("<I", header, 0, version)
    header[4:36] = prev_block
    header[36:68] = merkle_root
    struct.pack_into("<I", header, 68, time)
    struct.pack_into("<I", header, 72, bits)
    struct.pack_into("<I", header, 76, nonce)
    return bytes(header)


def _mine_header(
    version: int = 1,
    prev_block_hex: str = "00" * 32,
    merkle_root_hex: str = "00" * 32,
    time: int = 1_600_000_000,
    bits: int = 0x207fffff,
) -> BlockHeader:
    """Find a nonce so the header meets the easy target."""
    prev = bytes.fromhex(prev_block_hex)[::-1]
    root = bytes.fromhex(merkle_root_hex)[::-1]
    for nonce in range(2 ** 32):
        raw = _make_header_bytes(
            version=version,
            prev_block=prev,
            merkle_root=root,
            time=time,
            bits=bits,
            nonce=nonce,
        )
        h = _sha256d(raw)
        # Extract target from bits
        exp = (bits >> 24) & 0xFF
        man = bits & 0x007FFFFF
        target = man * (256 ** (exp - 3))
        if int(h[::-1].hex(), 16) <= target:
            return BlockHeader.from_hex(raw.hex())
    raise RuntimeError("Could not mine header")


def _build_merkle_root(txids_internal: list[bytes]) -> bytes:
    """Compute the Merkle root of a list of txids in internal byte order."""
    layer = list(txids_internal)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])  # duplicate last
        layer = [
            _sha256d(layer[i] + layer[i + 1])
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


def _build_merkle_branch(txids_internal: list[bytes], index: int) -> MerkleBranch:
    """Compute the Merkle branch for the txid at *index*."""
    layer = list(txids_internal)
    hashes: list[str] = []
    idx = index
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        sibling_idx = idx ^ 1  # XOR with 1 flips least significant bit
        hashes.append(layer[sibling_idx].hex())
        layer = [
            _sha256d(layer[i] + layer[i + 1])
            for i in range(0, len(layer), 2)
        ]
        idx //= 2
    return MerkleBranch(tx_index=index, hashes=hashes)


# ---------------------------------------------------------------------------
# _reverse_hex
# ---------------------------------------------------------------------------


class TestReverseHex:
    def test_reverses_byte_order(self):
        assert _reverse_hex("0102") == "0201"

    def test_roundtrip(self):
        h = "deadbeef"
        assert _reverse_hex(_reverse_hex(h)) == h

    def test_empty(self):
        assert _reverse_hex("") == ""


# ---------------------------------------------------------------------------
# BlockHeader
# ---------------------------------------------------------------------------


class TestBlockHeader:
    def _header(self, **kwargs) -> BlockHeader:
        raw = _make_header_bytes(**kwargs)
        return BlockHeader.from_hex(raw.hex())

    def test_from_hex_round_trip(self):
        raw = _make_header_bytes()
        h = BlockHeader.from_hex(raw.hex())
        assert h.to_bytes() == raw

    def test_from_hex_wrong_length_raises(self):
        with pytest.raises(SPVError):
            BlockHeader.from_hex("00" * 79)

    def test_version_parsed(self):
        h = self._header(version=2)
        assert h.version == 2

    def test_time_parsed(self):
        h = self._header(time=1_700_000_000)
        assert h.time == 1_700_000_000

    def test_nonce_parsed(self):
        raw = _make_header_bytes(nonce=42)
        h = BlockHeader.from_hex(raw.hex())
        assert h.nonce == 42

    def test_block_hash_is_64_hex(self):
        h = self._header()
        assert len(h.block_hash()) == 64
        assert all(c in "0123456789abcdef" for c in h.block_hash())

    def test_different_nonces_different_hashes(self):
        h1 = self._header(nonce=0)
        h2 = self._header(nonce=1)
        assert h1.block_hash() != h2.block_hash()

    def test_target_decoded_from_bits(self):
        # bits = 0x207fffff → exponent=0x20=32, mantissa=0x7fffff
        bits = 0x207fffff
        exp = (bits >> 24) & 0xFF  # 32
        man = bits & 0x007FFFFF    # 0x7fffff
        expected_target = man * (256 ** (exp - 3))
        h = self._header(bits=bits)
        assert h.target() == expected_target

    def test_meets_target_easy(self):
        # Mine a header that actually meets the easy target
        h = _mine_header(bits=0x207fffff)
        assert h.meets_target() is True

    def test_meets_target_impossible(self):
        # bits = 0x01000000 → target = 0 (impossible)
        h = self._header(bits=0x01000000)
        assert h.meets_target() is False

    def test_merkle_root_round_trip(self):
        mr = "ab" * 32
        raw = _make_header_bytes(merkle_root=bytes.fromhex(mr)[::-1])
        h = BlockHeader.from_hex(raw.hex())
        # The header stores it reversed; from_hex reverses again for display
        assert h.merkle_root == mr


# ---------------------------------------------------------------------------
# verify_merkle_branch
# ---------------------------------------------------------------------------


class TestVerifyMerkleBranch:
    def _setup(self, txids_display: list[str], target_index: int):
        """Return (txid_display, branch, merkle_root_display)."""
        internal = [bytes.fromhex(_reverse_hex(t)) for t in txids_display]
        root_internal = _build_merkle_root(internal)
        merkle_root_display = root_internal[::-1].hex()
        branch = _build_merkle_branch(internal, target_index)
        return txids_display[target_index], branch, merkle_root_display

    def test_single_tx(self):
        txids = ["aa" * 32]
        txid, branch, root = self._setup(txids, 0)
        assert verify_merkle_branch(txid, branch, root) is True

    def test_two_txs_index_0(self):
        txids = ["aa" * 32, "bb" * 32]
        txid, branch, root = self._setup(txids, 0)
        assert verify_merkle_branch(txid, branch, root) is True

    def test_two_txs_index_1(self):
        txids = ["aa" * 32, "bb" * 32]
        txid, branch, root = self._setup(txids, 1)
        assert verify_merkle_branch(txid, branch, root) is True

    def test_four_txs_all_indices(self):
        txids = ["aa" * 32, "bb" * 32, "cc" * 32, "dd" * 32]
        for i in range(4):
            txid, branch, root = self._setup(txids, i)
            assert verify_merkle_branch(txid, branch, root) is True

    def test_five_txs_odd_tree(self):
        txids = ["aa" * 32, "bb" * 32, "cc" * 32, "dd" * 32, "ee" * 32]
        for i in range(5):
            txid, branch, root = self._setup(txids, i)
            assert verify_merkle_branch(txid, branch, root) is True

    def test_tampered_txid_fails(self):
        txids = ["aa" * 32, "bb" * 32]
        _, branch, root = self._setup(txids, 0)
        # Provide a different txid
        assert verify_merkle_branch("cc" * 32, branch, root) is False

    def test_wrong_root_fails(self):
        txids = ["aa" * 32, "bb" * 32]
        txid, branch, _ = self._setup(txids, 0)
        assert verify_merkle_branch(txid, branch, "ff" * 32) is False


# ---------------------------------------------------------------------------
# verify_header_chain
# ---------------------------------------------------------------------------


class TestVerifyHeaderChain:
    def test_empty_chain_is_valid(self):
        result = verify_header_chain([])
        assert result.valid is True
        assert result.chain_length == 0

    def test_single_header_no_pow_check(self):
        raw = _make_header_bytes()
        h = BlockHeader.from_hex(raw.hex())
        result = verify_header_chain([h], check_pow=False)
        assert result.valid is True
        assert result.chain_length == 1

    def test_chain_linkage_correct(self):
        h1 = _mine_header()
        h2 = _mine_header(prev_block_hex=h1.block_hash())
        result = verify_header_chain([h1, h2], check_pow=True)
        assert result.valid is True
        assert result.chain_length == 2

    def test_broken_link_detected(self):
        h1 = _mine_header()
        # h2 has wrong prev_block
        h2 = _mine_header(prev_block_hex="ff" * 32)
        result = verify_header_chain([h1, h2], check_pow=False)
        assert result.valid is False
        assert any("prev_block" in e for e in result.errors)

    def test_pow_failure_detected(self):
        # A header with impossible bits should fail PoW
        raw = _make_header_bytes(bits=0x01000000)  # target = 0
        h = BlockHeader.from_hex(raw.hex())
        result = verify_header_chain([h], check_pow=True)
        assert result.valid is False

    def test_errors_list_populated_on_failure(self):
        h1 = _mine_header()
        h2 = _mine_header(prev_block_hex="ff" * 32)
        result = verify_header_chain([h1, h2], check_pow=False)
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# verify_spv_proof (end-to-end)
# ---------------------------------------------------------------------------


class TestVerifySpvProof:
    def _make_proof(self, txids_display: list[str], target_index: int) -> SPVProof:
        internal = [bytes.fromhex(_reverse_hex(t)) for t in txids_display]
        root_internal = _build_merkle_root(internal)
        merkle_root_display = root_internal[::-1].hex()
        branch = _build_merkle_branch(internal, target_index)
        # Build a mined header with this Merkle root
        header = _mine_header(merkle_root_hex=merkle_root_display)
        return SPVProof(
            txid=txids_display[target_index],
            branch=branch,
            header=header,
        )

    def test_valid_proof_returns_true(self):
        proof = self._make_proof(["aa" * 32, "bb" * 32], 0)
        assert verify_spv_proof(proof, check_pow=True) is True

    def test_tampered_txid_fails(self):
        proof = self._make_proof(["aa" * 32, "bb" * 32], 0)
        tampered = SPVProof(
            txid="cc" * 32,  # wrong txid
            branch=proof.branch,
            header=proof.header,
        )
        assert verify_spv_proof(tampered, check_pow=False) is False

    def test_all_txs_in_block_verify(self):
        txids = ["aa" * 32, "bb" * 32, "cc" * 32, "dd" * 32]
        for i in range(4):
            proof = self._make_proof(txids, i)
            assert verify_spv_proof(proof, check_pow=True) is True
