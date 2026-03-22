"""Performance: Merkle tree construction and proof verification.

These thresholds reflect expected overhead on a standard developer machine.
If they fail on a slow CI runner, increase the ceiling — but investigate first.
"""

from __future__ import annotations

import hashlib
import time

from aria.core.merkle import ARIAMerkleTree, verify_proof


def _fake_hash(i: int) -> str:
    return "sha256:" + hashlib.sha256(i.to_bytes(4, "big")).hexdigest()


class TestMerklePerformance:
    def _build_tree(self, n: int) -> ARIAMerkleTree:
        tree = ARIAMerkleTree()
        for i in range(n):
            tree.add(_fake_hash(i))
        return tree

    def _time_build_ms(self, n: int) -> float:
        start = time.perf_counter()
        tree = self._build_tree(n)
        _ = tree.root()
        return (time.perf_counter() - start) * 1000

    def test_tree_10_records_under_5ms(self):
        assert self._time_build_ms(10) < 5.0

    def test_tree_100_records_under_10ms(self):
        assert self._time_build_ms(100) < 10.0

    def test_tree_500_records_under_50ms(self):
        assert self._time_build_ms(500) < 50.0

    def test_proof_verification_under_1ms(self):
        """Merkle proof verification for a 500-leaf tree must be < 1ms."""
        tree = self._build_tree(500)
        root = tree.root()
        target = _fake_hash(250)
        proof = tree.proof(target)

        start = time.perf_counter()
        for _ in range(1_000):
            verify_proof(root, proof, target)
        mean_ms = (time.perf_counter() - start) / 1_000 * 1000

        assert mean_ms < 1.0, f"proof verification: {mean_ms:.3f}ms > 1ms"

    def test_root_stable_across_equivalent_insertions(self):
        """Sanity: two trees with identical hashes produce identical roots."""
        hashes = [_fake_hash(i) for i in range(50)]
        t1 = ARIAMerkleTree()
        t2 = ARIAMerkleTree()
        for h in hashes:
            t1.add(h)
            t2.add(h)
        assert t1.root() == t2.root()
