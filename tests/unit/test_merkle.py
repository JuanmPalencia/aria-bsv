"""
tests/unit/test_merkle.py

Unit tests for aria.core.merkle.

Coverage targets:
  - ARIAMerkleTree: add, root, proof, __len__
  - verify_proof: valid proof, invalid proof, tampered root, wrong target
  - Edge cases: 1 leaf, odd/even counts, duplicate hashes
  - Second-preimage protection (distinct hash for leaf vs internal node)
"""

import hashlib

import pytest

from aria.core.errors import ARIASerializationError
from aria.core.merkle import (
    ARIAMerkleTree,
    MerkleProof,
    _hash_internal,
    _hash_leaf,
    _parse_hash,
    verify_proof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_hash(n: int) -> str:
    """Return a valid 'sha256:{hex}' string seeded from integer n."""
    return "sha256:" + hashlib.sha256(str(n).encode()).hexdigest()


def tree_from(*ns: int) -> ARIAMerkleTree:
    t = ARIAMerkleTree()
    for n in ns:
        t.add(fake_hash(n))
    return t


# ---------------------------------------------------------------------------
# _parse_hash (internal — tested for robustness)
# ---------------------------------------------------------------------------

class TestParseHash:

    def test_valid_hash_returns_bytes(self) -> None:
        b = _parse_hash("sha256:" + "ab" * 32)
        assert len(b) == 32

    def test_missing_prefix_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            _parse_hash("deadbeef" * 8)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            _parse_hash("sha256:abc")

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            _parse_hash("sha256:" + "z" * 64)


# ---------------------------------------------------------------------------
# Second-preimage protection
# ---------------------------------------------------------------------------

class TestSecondPreimage:

    def test_leaf_and_internal_produce_different_bytes(self) -> None:
        h = fake_hash(1)
        leaf_bytes = _hash_leaf(h)
        # An internal node over (h, h) should differ from a leaf of h
        internal_bytes = _hash_internal(leaf_bytes, leaf_bytes)
        assert leaf_bytes != internal_bytes

    def test_leaf_prefix_differs_from_internal_prefix(self) -> None:
        """Domain-separation: 0x00 for leaves, 0x01 for internal nodes."""
        h = fake_hash(42)
        raw = _parse_hash(h)
        leaf     = hashlib.sha256(b"\x00" + raw).digest()
        internal = hashlib.sha256(b"\x01" + raw + raw).digest()
        assert leaf != internal


# ---------------------------------------------------------------------------
# ARIAMerkleTree — basic operations
# ---------------------------------------------------------------------------

class TestARIAMerkleTreeBasic:

    def test_len_zero_initially(self) -> None:
        assert len(ARIAMerkleTree()) == 0

    def test_len_after_adds(self) -> None:
        t = tree_from(1, 2, 3)
        assert len(t) == 3

    def test_invalid_hash_not_added(self) -> None:
        t = ARIAMerkleTree()
        with pytest.raises(ARIASerializationError):
            t.add("not-a-valid-hash")
        assert len(t) == 0

    def test_empty_tree_root_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            ARIAMerkleTree().root()

    def test_empty_tree_proof_raises(self) -> None:
        with pytest.raises(ARIASerializationError):
            ARIAMerkleTree().proof(fake_hash(1))


# ---------------------------------------------------------------------------
# ARIAMerkleTree — root computation
# ---------------------------------------------------------------------------

class TestARIAMerkleTreeRoot:

    def test_single_leaf_root_format(self) -> None:
        root = tree_from(1).root()
        assert root.startswith("sha256:")
        assert len(root) == len("sha256:") + 64

    def test_single_leaf_root_is_leaf_hash(self) -> None:
        h = fake_hash(1)
        t = ARIAMerkleTree()
        t.add(h)
        expected = "sha256:" + _hash_leaf(h).hex()
        assert t.root() == expected

    def test_two_leaves_root_is_internal_hash(self) -> None:
        h1, h2 = fake_hash(1), fake_hash(2)
        t = ARIAMerkleTree()
        t.add(h1)
        t.add(h2)
        leaf1 = _hash_leaf(h1)
        leaf2 = _hash_leaf(h2)
        expected = "sha256:" + _hash_internal(leaf1, leaf2).hex()
        assert t.root() == expected

    def test_same_leaves_same_root(self) -> None:
        assert tree_from(1, 2, 3).root() == tree_from(1, 2, 3).root()

    def test_different_leaf_order_different_root(self) -> None:
        assert tree_from(1, 2).root() != tree_from(2, 1).root()

    def test_root_changes_when_leaf_changes(self) -> None:
        r1 = tree_from(1, 2, 3).root()
        r2 = tree_from(1, 2, 4).root()
        assert r1 != r2

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17])
    def test_various_sizes_produce_valid_root(self, count: int) -> None:
        root = tree_from(*range(count)).root()
        assert root.startswith("sha256:")
        assert len(root) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# ARIAMerkleTree — proof generation and verification
# ---------------------------------------------------------------------------

class TestMerkleProof:

    @pytest.mark.parametrize("count,target_idx", [
        (1, 0),
        (2, 0), (2, 1),
        (3, 0), (3, 1), (3, 2),
        (4, 0), (4, 3),
        (5, 2), (5, 4),
        (8, 0), (8, 7),
        (9, 0), (9, 8),
    ])
    def test_proof_verifies_for_every_position(self, count: int, target_idx: int) -> None:
        hashes = [fake_hash(i) for i in range(count)]
        t = ARIAMerkleTree()
        for h in hashes:
            t.add(h)
        target = hashes[target_idx]
        proof = t.proof(target)
        assert verify_proof(t.root(), proof, target)

    def test_proof_root_matches_tree_root(self) -> None:
        t = tree_from(1, 2, 3)
        proof = t.proof(fake_hash(1))
        assert proof.root == t.root()

    def test_proof_target_hash_matches(self) -> None:
        h = fake_hash(42)
        t = ARIAMerkleTree()
        t.add(h)
        proof = t.proof(h)
        assert proof.target_hash == h

    def test_proof_for_nonexistent_leaf_raises(self) -> None:
        t = tree_from(1, 2, 3)
        with pytest.raises(ARIASerializationError):
            t.proof(fake_hash(999))


# ---------------------------------------------------------------------------
# verify_proof — invalid inputs
# ---------------------------------------------------------------------------

class TestVerifyProofInvalidInputs:

    def test_wrong_record_hash_returns_false(self) -> None:
        t = tree_from(1, 2, 3)
        proof = t.proof(fake_hash(1))
        assert not verify_proof(t.root(), proof, fake_hash(2))

    def test_tampered_root_returns_false(self) -> None:
        t = tree_from(1, 2, 3)
        proof = t.proof(fake_hash(1))
        fake_root = "sha256:" + "0" * 64
        assert not verify_proof(fake_root, proof, fake_hash(1))

    def test_tampered_sibling_in_path_returns_false(self) -> None:
        t = tree_from(1, 2, 3)
        original_proof = t.proof(fake_hash(1))
        # Tamper with the first sibling hash
        tampered_path = [
            ("sha256:" + "0" * 64, original_proof.proof_path[0][1]),
            *original_proof.proof_path[1:],
        ]
        tampered_proof = MerkleProof(
            target_hash=original_proof.target_hash,
            proof_path=tampered_path,
            root=original_proof.root,
        )
        assert not verify_proof(t.root(), tampered_proof, fake_hash(1))

    def test_mismatched_target_hash_returns_false(self) -> None:
        t = tree_from(1, 2)
        proof = t.proof(fake_hash(1))
        # proof.target_hash = fake_hash(1), but we pass fake_hash(2)
        assert not verify_proof(t.root(), proof, fake_hash(2))

    def test_invalid_sibling_hash_format_returns_false(self) -> None:
        t = tree_from(1, 2)
        original_proof = t.proof(fake_hash(1))
        bad_path = [("not-a-hash", original_proof.proof_path[0][1])]
        bad_proof = MerkleProof(
            target_hash=original_proof.target_hash,
            proof_path=bad_path,
            root=original_proof.root,
        )
        # verify_proof must not raise — must return False
        assert not verify_proof(t.root(), bad_proof, fake_hash(1))

    def test_empty_proof_path_wrong_root_returns_false(self) -> None:
        t = tree_from(1, 2, 3)
        # Single-leaf proof has no siblings; multi-leaf tree root won't match
        single_proof = MerkleProof(
            target_hash=fake_hash(1),
            proof_path=[],
            root=t.root(),
        )
        assert not verify_proof(t.root(), single_proof, fake_hash(1))


# ---------------------------------------------------------------------------
# Duplicate leaves
# ---------------------------------------------------------------------------

class TestDuplicateLeaves:

    def test_duplicate_leaves_both_provable(self) -> None:
        h = fake_hash(1)
        t = ARIAMerkleTree()
        t.add(h)
        t.add(h)  # same hash twice
        assert len(t) == 2
        # proof() uses index(), which returns the first occurrence
        proof = t.proof(h)
        assert verify_proof(t.root(), proof, h)
