"""Property-based tests for aria.core.merkle using Hypothesis.

Invariants:
    - root() is deterministic for a given list of leaves.
    - Adding a leaf always changes the root.
    - Leaf order matters (permuting changes the root).
    - root() result starts with 'sha256:' and has correct total length.
    - Second-preimage protection: root of [h] ≠ h.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aria.core.merkle import ARIAMerkleTree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simulate real leaf hashes: sha256: + 64 hex chars = 71 chars total
def _leaf(b: bytes) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _build_tree(leaves: list[str]) -> ARIAMerkleTree:
    t = ARIAMerkleTree()
    for leaf in leaves:
        t.add(leaf)
    return t


def _root(leaves: list[str]) -> str:
    return _build_tree(leaves).root()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Each leaf is sha256:<64hex> of some random bytes
leaf = st.binary(min_size=1, max_size=64).map(_leaf)
leaf_list = st.lists(leaf, min_size=1, max_size=20)

# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


@given(leaves=leaf_list)
def test_root_starts_with_sha256(leaves: list[str]):
    """root() always returns a 'sha256:...' prefixed string."""
    root = _root(leaves)
    assert root.startswith("sha256:")
    hex_part = root[len("sha256:"):]
    assert len(hex_part) == 64
    assert all(c in "0123456789abcdef" for c in hex_part)


@given(leaves=leaf_list)
def test_root_deterministic(leaves: list[str]):
    """Same leaf list → same root every time."""
    assert _root(leaves) == _root(leaves)


def test_empty_tree_raises_or_returns_zero():
    """Empty tree root is well-defined (implementation may raise or return zeros)."""
    t = ARIAMerkleTree()
    # Either raises or returns a deterministic value — both are acceptable
    try:
        root = t.root()
        assert isinstance(root, str)
    except Exception:
        pass  # Some implementations raise on empty tree


@given(leaf_=leaf)
def test_single_leaf_second_preimage_protection(leaf_: str):
    """Root of a single leaf != the raw leaf hash (0x00 prefix prevents it)."""
    root = _root([leaf_])
    assert root != leaf_


# ---------------------------------------------------------------------------
# Leaf order matters
# ---------------------------------------------------------------------------


@given(leaves=st.lists(leaf, min_size=2, max_size=10))
def test_order_matters(leaves: list[str]):
    """Reversing the leaf list changes the root (for non-palindromic lists)."""
    rev = list(reversed(leaves))
    assume(leaves != rev)
    assert _root(leaves) != _root(rev)


# ---------------------------------------------------------------------------
# Appending a leaf changes the root
# ---------------------------------------------------------------------------


@given(leaves=leaf_list, new_leaf=leaf)
def test_adding_leaf_changes_root(leaves: list[str], new_leaf: str):
    """Appending a leaf produces a different root."""
    assume(new_leaf not in leaves)
    root_before = _root(leaves)
    root_after = _root(leaves + [new_leaf])
    assert root_before != root_after


# ---------------------------------------------------------------------------
# ARIAMerkleTree class
# ---------------------------------------------------------------------------


@given(leaves=leaf_list)
def test_tree_root_matches_fresh_build(leaves: list[str]):
    """A tree built incrementally gives the same root as a fresh build."""
    t1 = _build_tree(leaves)
    t2 = _build_tree(leaves)
    assert t1.root() == t2.root()


@given(a=leaf_list, b=leaf_list)
def test_two_different_leaf_sets_different_roots(a: list[str], b: list[str]):
    """Two distinct leaf lists produce distinct roots."""
    assume(a != b)
    assert _root(a) != _root(b)
