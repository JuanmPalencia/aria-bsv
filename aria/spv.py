"""aria.spv — Simplified Payment Verification for BSV (offline).

Provides offline SPV verification:
    - Block header parsing and validation (hash ≤ target, PoW).
    - Merkle branch verification: proves a txid is in a block without a full
      node or API call.
    - Chain of headers verification: each header correctly links to the prior.

All inputs are hex strings; all serialisation follows BSV (little-endian txid,
big-endian target comparison).

Reference: BRC-121 §7 / Bitcoin SPV chapter.

No external dependencies — uses only Python stdlib (hashlib, struct).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Sequence

from aria.core.errors import ARIAError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SPVError(ARIAError):
    """Raised when SPV verification fails."""


# ---------------------------------------------------------------------------
# Double SHA-256
# ---------------------------------------------------------------------------


def _sha256d(data: bytes) -> bytes:
    """SHA-256(SHA-256(data)) — standard Bitcoin double hash."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _reverse_hex(h: str) -> str:
    """Reverse the byte order of a hex string (little-endian ↔ big-endian)."""
    return bytes.fromhex(h)[::-1].hex()


# ---------------------------------------------------------------------------
# Block header
# ---------------------------------------------------------------------------

_HEADER_SIZE = 80  # bytes


@dataclass
class BlockHeader:
    """Parsed BSV block header (80 bytes).

    Fields match the Bitcoin wire format:
        version (4) | prev_block (32) | merkle_root (32) | time (4) |
        bits (4) | nonce (4)

    All hex values are in **internal byte order** (little-endian as stored in
    the serialised header), matching what nodes and explorers return.
    """

    version: int
    prev_block: str      # 64 hex chars, internal byte order
    merkle_root: str     # 64 hex chars, internal byte order
    time: int            # Unix timestamp
    bits: int            # Compact target representation
    nonce: int

    @classmethod
    def from_hex(cls, hex_header: str) -> "BlockHeader":
        """Parse a hex-encoded 80-byte block header.

        Args:
            hex_header: 160 hex characters (80 bytes).

        Returns:
            A :class:`BlockHeader` instance.

        Raises:
            SPVError: If the input is not exactly 80 bytes.
        """
        raw = bytes.fromhex(hex_header)
        if len(raw) != _HEADER_SIZE:
            raise SPVError(
                f"Block header must be exactly 80 bytes, got {len(raw)}"
            )
        version, = struct.unpack_from("<I", raw, 0)
        prev_block = raw[4:36][::-1].hex()      # display as big-endian
        merkle_root = raw[36:68][::-1].hex()    # display as big-endian
        ts, bits, nonce = struct.unpack_from("<III", raw, 68)
        return cls(
            version=version,
            prev_block=prev_block,
            merkle_root=merkle_root,
            time=ts,
            bits=bits,
            nonce=nonce,
        )

    def to_bytes(self) -> bytes:
        """Serialise header back to 80 raw bytes (wire format)."""
        out = bytearray(80)
        struct.pack_into("<I", out, 0, self.version)
        out[4:36] = bytes.fromhex(self.prev_block)[::-1]
        out[36:68] = bytes.fromhex(self.merkle_root)[::-1]
        struct.pack_into("<I", out, 68, self.time)
        struct.pack_into("<I", out, 72, self.bits)
        struct.pack_into("<I", out, 76, self.nonce)
        return bytes(out)

    def block_hash(self) -> str:
        """Return the block hash as a 64-hex big-endian string.

        Block hash = SHA-256d(header_bytes) reversed for display.
        """
        raw = _sha256d(self.to_bytes())
        return raw[::-1].hex()

    def target(self) -> int:
        """Decode *bits* (compact representation) to the full 256-bit target."""
        exponent = (self.bits >> 24) & 0xFF
        mantissa = self.bits & 0x007FFFFF
        return mantissa * (256 ** (exponent - 3))

    def meets_target(self) -> bool:
        """Return True if block_hash ≤ target (valid proof-of-work)."""
        hash_int = int(self.block_hash(), 16)
        return hash_int <= self.target()


# ---------------------------------------------------------------------------
# Merkle branch verification
# ---------------------------------------------------------------------------


@dataclass
class MerkleBranch:
    """A Merkle inclusion proof for a single transaction.

    Attributes:
        tx_index:  Position of the transaction within the block (0-based).
        hashes:    Sibling hashes at each level, in wire (internal) byte order.
        flags:     Not used in this simplified implementation.
    """

    tx_index: int
    hashes: list[str]  # each entry: 64 hex chars


def verify_merkle_branch(
    txid: str,
    branch: MerkleBranch,
    expected_merkle_root: str,
) -> bool:
    """Verify that *txid* is included in a block whose Merkle root is known.

    Uses the standard Bitcoin Merkle tree (double SHA-256, no second-preimage
    prefix — the BSV Merkle tree uses raw SHA-256d, not RFC 6962).

    Note: this is the *transaction Merkle tree* used in block headers, which
    differs from the ARIA audit Merkle tree (which uses SHA-256 with RFC 6962
    prefixes).  They serve different purposes and must not be confused.

    Args:
        txid:                  64-char hex transaction ID (big-endian display).
        branch:                :class:`MerkleBranch` containing sibling hashes.
        expected_merkle_root:  Merkle root from the block header (big-endian).

    Returns:
        ``True`` if the proof is valid.
    """
    # Work in internal (little-endian) byte order
    current = bytes.fromhex(_reverse_hex(txid))
    index = branch.tx_index

    for sibling_hex in branch.hashes:
        sibling = bytes.fromhex(sibling_hex)
        if index % 2 == 0:
            combined = current + sibling
        else:
            combined = sibling + current
        current = _sha256d(combined)
        index //= 2

    computed_root = current[::-1].hex()  # convert to display order
    return computed_root == expected_merkle_root


# ---------------------------------------------------------------------------
# Header chain verifier
# ---------------------------------------------------------------------------


@dataclass
class SPVVerificationResult:
    """Result of a chain-of-headers SPV verification."""

    valid: bool
    chain_length: int
    errors: list[str] = field(default_factory=list)


def verify_header_chain(
    headers: Sequence[BlockHeader],
    *,
    check_pow: bool = True,
) -> SPVVerificationResult:
    """Verify a sequence of block headers form a valid chain.

    Checks:
        1. Each header's ``prev_block`` matches the hash of the prior header.
        2. (Optional) Each header's hash is ≤ its declared target (PoW check).

    Args:
        headers:    Ordered list of :class:`BlockHeader` objects (oldest first).
        check_pow:  If ``True`` (default), also verify proof-of-work for each
                    header.  Set to ``False`` for test networks or regtest.

    Returns:
        :class:`SPVVerificationResult` with ``valid=True`` if all checks pass.
    """
    errors: list[str] = []

    if not headers:
        return SPVVerificationResult(valid=True, chain_length=0)

    if check_pow and not headers[0].meets_target():
        errors.append(f"Header 0 does not meet PoW target")

    for i in range(1, len(headers)):
        prev_hash = headers[i - 1].block_hash()
        declared_prev = headers[i].prev_block
        if prev_hash != declared_prev:
            errors.append(
                f"Header {i} prev_block mismatch: "
                f"expected {prev_hash}, got {declared_prev}"
            )
        if check_pow and not headers[i].meets_target():
            errors.append(f"Header {i} does not meet PoW target")

    return SPVVerificationResult(
        valid=len(errors) == 0,
        chain_length=len(headers),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Full SPV proof
# ---------------------------------------------------------------------------


@dataclass
class SPVProof:
    """A complete SPV proof: a transaction + Merkle branch + block header.

    Attributes:
        txid:    Transaction ID (big-endian display, 64 hex chars).
        branch:  Merkle inclusion proof.
        header:  The block header that includes the transaction.
    """

    txid: str
    branch: MerkleBranch
    header: BlockHeader


def verify_spv_proof(proof: SPVProof, *, check_pow: bool = True) -> bool:
    """Verify a complete SPV proof.

    Checks:
        1. The Merkle branch links *txid* to the block's Merkle root.
        2. (Optional) The block header meets its declared PoW target.

    Args:
        proof:      :class:`SPVProof` containing txid, branch, and header.
        check_pow:  Also verify proof-of-work (default: ``True``).

    Returns:
        ``True`` if both checks pass.
    """
    if check_pow and not proof.header.meets_target():
        return False
    return verify_merkle_branch(
        proof.txid, proof.branch, proof.header.merkle_root
    )
