"""
aria/core/merkle.py

ARIA Merkle tree with second-preimage attack protection.

Node hashing rules (BRC-121 §4.5 / RFC 6962):
  Leaf node:     SHA-256( 0x00 || leaf_hash_bytes )
  Internal node: SHA-256( 0x01 || left_bytes || right_bytes )

The 0x00 / 0x01 domain-separation prefixes prevent an attacker from
crafting a valid inclusion proof by presenting an internal node hash as
if it were a leaf.

If the number of leaves at any level is odd, the last node is duplicated
to complete the level before computing the parent. This follows the same
convention used by Bitcoin's Merkle tree.

All hashes flowing through the public API use the "sha256:{hex}" string
format defined in aria/core/hasher.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from aria.core.errors import ARIASerializationError

# Domain-separation prefixes (RFC 6962 §2.1)
_LEAF_PREFIX     = b"\x00"
_INTERNAL_PREFIX = b"\x01"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_hash(h: str, context: str = "") -> bytes:
    """
    Extract the raw 32-byte value from a "sha256:{hex}" string.

    Raises:
        ARIASerializationError: if *h* is not in the expected format.
    """
    prefix = "sha256:"
    if not isinstance(h, str) or not h.startswith(prefix):
        raise ARIASerializationError(
            f"Hash must start with 'sha256:', got {h!r}"
            + (f" ({context})" if context else "")
        )
    hex_part = h[len(prefix):]
    if len(hex_part) != 64:
        raise ARIASerializationError(
            f"Hash must be 'sha256:' + 64 hex chars, got {len(hex_part)} chars"
            + (f" ({context})" if context else "")
        )
    try:
        return bytes.fromhex(hex_part)
    except ValueError as exc:
        raise ARIASerializationError(
            f"Hash contains invalid hex: {exc}"
            + (f" ({context})" if context else "")
        ) from exc


def _fmt(b: bytes) -> str:
    """Format 32 raw bytes as 'sha256:{hex}'."""
    return f"sha256:{b.hex()}"


def _hash_leaf(leaf_hash: str) -> bytes:
    """SHA-256( 0x00 || leaf_bytes ) — leaf-domain hash."""
    return hashlib.sha256(_LEAF_PREFIX + _parse_hash(leaf_hash, "leaf")).digest()


def _hash_internal(left: bytes, right: bytes) -> bytes:
    """SHA-256( 0x01 || left || right ) — internal-node hash."""
    return hashlib.sha256(_INTERNAL_PREFIX + left + right).digest()


def _build_tree(leaf_nodes: list[bytes]) -> list[list[bytes]]:
    """
    Build the full Merkle tree bottom-up and return all levels.

    levels[0]  = leaf nodes (after 0x00 prefix hashing)
    levels[-1] = [root]
    """
    levels: list[list[bytes]] = [list(leaf_nodes)]
    current = list(leaf_nodes)

    while len(current) > 1:
        if len(current) % 2 == 1:
            current.append(current[-1])  # duplicate last node if odd
        next_level = [
            _hash_internal(current[i], current[i + 1])
            for i in range(0, len(current), 2)
        ]
        levels.append(next_level)
        current = next_level

    return levels


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class MerkleProof:
    """
    Cryptographic inclusion proof for one leaf in an ARIA Merkle tree.

    Attributes:
        target_hash: "sha256:{hex}" of the record being proved.
        proof_path:  Ordered list of (sibling_hash, position) pairs.
                     *position* is "right" if the sibling is the right child,
                     "left" if it is the left child. Apply them in order from
                     leaf to root to reconstruct the root hash.
        root:        The Merkle root this proof is valid against.
    """
    target_hash: str
    proof_path:  list[tuple[str, Literal["left", "right"]]]
    root:        str


# ---------------------------------------------------------------------------
# ARIAMerkleTree
# ---------------------------------------------------------------------------

class ARIAMerkleTree:
    """
    Incremental Merkle tree for ARIA epoch records.

    Records are added one at a time via *add()*. Once all records for an
    epoch have been added, call *root()* to obtain the value that goes into
    the EPOCH_CLOSE transaction, and *proof()* to generate inclusion proofs.

    The tree is append-only: records cannot be removed or reordered after
    being added.

    Example::

        tree = ARIAMerkleTree()
        for record in epoch_records:
            tree.add(record.hash())
        root = tree.root()
        proof = tree.proof(some_record.hash())
        assert verify_proof(root, proof, some_record.hash())
    """

    def __init__(self) -> None:
        # Original "sha256:{hex}" strings, in insertion order.
        self._leaves: list[str] = []
        # Leaf nodes after applying the 0x00-prefix hash.
        self._leaf_nodes: list[bytes] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, record_hash: str) -> None:
        """
        Append *record_hash* as the next leaf.

        Args:
            record_hash: "sha256:{64 hex chars}" — typically AuditRecord.hash().

        Raises:
            ARIASerializationError: if *record_hash* is not a valid hash string.
        """
        # Validate format (raises if invalid).
        _parse_hash(record_hash, "record_hash")
        self._leaves.append(record_hash)
        self._leaf_nodes.append(_hash_leaf(record_hash))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._leaves)

    def root(self) -> str:
        """
        Compute and return the Merkle root.

        Returns:
            "sha256:{hex}" of the root node.

        Raises:
            ARIASerializationError: if the tree has no leaves.
        """
        if not self._leaf_nodes:
            raise ARIASerializationError(
                "Cannot compute Merkle root of an empty tree. "
                "Add at least one record before calling root()."
            )
        levels = _build_tree(self._leaf_nodes)
        return _fmt(levels[-1][0])

    def proof(self, record_hash: str) -> MerkleProof:
        """
        Generate an inclusion proof for *record_hash*.

        Args:
            record_hash: "sha256:{hex}" of the record to prove.

        Returns:
            MerkleProof that can be verified with *verify_proof()*.

        Raises:
            ARIASerializationError: if *record_hash* is not in the tree,
                or the tree is empty.
        """
        if not self._leaves:
            raise ARIASerializationError(
                "Cannot generate a proof from an empty tree."
            )
        if record_hash not in self._leaves:
            raise ARIASerializationError(
                f"Record hash not found in tree: {record_hash[:20]}..."
            )

        root_str = self.root()
        proof_path: list[tuple[str, Literal["left", "right"]]] = []

        idx = self._leaves.index(record_hash)
        current_level = list(self._leaf_nodes)

        while len(current_level) > 1:
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            sibling_idx: int = idx ^ 1  # toggle the last bit
            sibling_bytes = current_level[sibling_idx]
            # If idx is even, we are the left child → sibling is on the right.
            position: Literal["left", "right"] = "right" if idx % 2 == 0 else "left"
            proof_path.append((_fmt(sibling_bytes), position))

            # Move up one level.
            next_level = [
                _hash_internal(current_level[i], current_level[i + 1])
                for i in range(0, len(current_level), 2)
            ]
            idx = idx // 2
            current_level = next_level

        return MerkleProof(
            target_hash=record_hash,
            proof_path=proof_path,
            root=root_str,
        )


# ---------------------------------------------------------------------------
# verify_proof — standalone, no tree instance required
# ---------------------------------------------------------------------------

def verify_proof(root: str, proof: MerkleProof, record_hash: str) -> bool:
    """
    Verify a Merkle inclusion proof.

    This function is self-contained: it does not require the original
    ARIAMerkleTree instance. Anyone with the *root* (from the EPOCH_CLOSE
    transaction) can verify whether a record was included in the epoch.

    Args:
        root:        Expected Merkle root ("sha256:{hex}"), from EPOCH_CLOSE.
        proof:       The MerkleProof to verify.
        record_hash: The record hash claimed to be in the tree.

    Returns:
        True if the proof is cryptographically valid, False otherwise.
        Never raises — invalid inputs result in False.
    """
    try:
        # The proof must be for the record we are verifying.
        if proof.target_hash != record_hash:
            return False

        # Start from the leaf node (0x00-prefix hash of the record).
        current = _hash_leaf(record_hash)

        for sibling_hash, position in proof.proof_path:
            sibling_bytes = _parse_hash(sibling_hash, "proof sibling")
            if position == "right":
                current = _hash_internal(current, sibling_bytes)
            else:
                current = _hash_internal(sibling_bytes, current)

        return _fmt(current) == root

    except (ARIASerializationError, ValueError):
        return False
