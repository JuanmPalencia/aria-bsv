"""Tests for aria.zk.aggregate — MerkleAggregator and AggregateProof."""

from __future__ import annotations

import pytest

from aria.core.errors import ARIAZKError
from aria.zk.aggregate import AggregateProof, MerkleAggregator, NovaAggregator
from aria.zk.base import ZKProof


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_proof(record_id: str, epoch_id: str = "ep_test") -> ZKProof:
    import hashlib
    proof_bytes = hashlib.sha256(f"proof:{record_id}".encode()).digest()
    return ZKProof(
        proof_bytes=proof_bytes,
        public_inputs=["sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64],
        proving_system="mock",
        tier="full_zk",
        model_hash="sha256:" + "d" * 64,
        prover_version="mock-1.0",
        epoch_id=epoch_id,
        record_id=record_id,
    )


# ---------------------------------------------------------------------------
# AggregateProof
# ---------------------------------------------------------------------------

class TestAggregateProof:
    def test_digest_is_sha256_of_aggregate_bytes(self):
        import hashlib
        agg = AggregateProof(
            proofs_merkle_root="sha256:" + "a" * 64,
            n_proofs=3,
            aggregation_scheme="merkle",
            aggregate_bytes=b"test",
            epoch_id="ep_test",
        )
        expected = "sha256:" + hashlib.sha256(b"test").hexdigest()
        assert agg.digest() == expected

    def test_to_dict_keys(self):
        agg = AggregateProof(
            proofs_merkle_root="sha256:" + "a" * 64,
            n_proofs=5,
            aggregation_scheme="merkle",
            aggregate_bytes=b"\x01" * 32,
            epoch_id="ep_test",
        )
        d = agg.to_dict()
        assert "proofs_merkle_root" in d
        assert "n_proofs" in d
        assert "aggregation_scheme" in d
        assert "aggregate_digest" in d


# ---------------------------------------------------------------------------
# MerkleAggregator
# ---------------------------------------------------------------------------

class TestMerkleAggregator:
    def test_aggregate_empty(self):
        agg = MerkleAggregator()
        result = agg.aggregate([], "ep_test")
        assert result.n_proofs == 0
        assert result.aggregation_scheme == "merkle"

    def test_aggregate_single_proof(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof("rec_0")]
        result = agg.aggregate(proofs, "ep_test")
        assert result.n_proofs == 1
        assert result.proofs_merkle_root.startswith("sha256:")

    def test_aggregate_multiple_proofs(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof(f"rec_{i}") for i in range(5)]
        result = agg.aggregate(proofs, "ep_test")
        assert result.n_proofs == 5

    def test_aggregate_is_deterministic(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof(f"rec_{i}") for i in range(3)]
        r1 = agg.aggregate(proofs, "ep_test")
        r2 = agg.aggregate(proofs, "ep_test")
        assert r1.proofs_merkle_root == r2.proofs_merkle_root

    def test_verify_aggregate_passes(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof(f"rec_{i}") for i in range(4)]
        result = agg.aggregate(proofs, "ep_test")
        assert agg.verify_aggregate(result, proofs) is True

    def test_verify_aggregate_fails_with_extra_proof(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof(f"rec_{i}") for i in range(3)]
        result = agg.aggregate(proofs, "ep_test")
        extra = proofs + [_fake_proof("rec_extra")]
        assert agg.verify_aggregate(result, extra) is False

    def test_verify_aggregate_fails_with_tampered_proof(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof("rec_0")]
        result = agg.aggregate(proofs, "ep_test")
        tampered = ZKProof(
            proof_bytes=b"\xff" * 32,  # altered bytes
            public_inputs=proofs[0].public_inputs,
            proving_system="mock",
            tier="full_zk",
            model_hash=proofs[0].model_hash,
            prover_version="mock-1.0",
            epoch_id="ep_test",
            record_id="rec_0",
        )
        assert agg.verify_aggregate(result, [tampered]) is False

    def test_different_proofs_different_root(self):
        agg = MerkleAggregator()
        proofs_a = [_fake_proof("rec_0"), _fake_proof("rec_1")]
        proofs_b = [_fake_proof("rec_0"), _fake_proof("rec_2")]
        r_a = agg.aggregate(proofs_a, "ep_test")
        r_b = agg.aggregate(proofs_b, "ep_test")
        assert r_a.proofs_merkle_root != r_b.proofs_merkle_root

    def test_membership_path_returns_proof(self):
        agg = MerkleAggregator()
        proofs = [_fake_proof(f"rec_{i}") for i in range(4)]
        path = agg.membership_path(proofs[2], proofs)
        assert path is not None


# ---------------------------------------------------------------------------
# NovaAggregator (experimental stub)
# ---------------------------------------------------------------------------

class TestNovaAggregator:
    def test_aggregate_raises_not_implemented(self):
        nova = NovaAggregator()
        with pytest.raises(ARIAZKError, match="NovaAggregator"):
            nova.aggregate([], "ep_test")

    def test_verify_raises_not_implemented(self):
        nova = NovaAggregator()
        dummy = AggregateProof(
            proofs_merkle_root="sha256:" + "a" * 64,
            n_proofs=0,
            aggregation_scheme="nova",
            aggregate_bytes=b"",
            epoch_id="ep_test",
        )
        with pytest.raises(ARIAZKError):
            nova.verify_aggregate(dummy, [])
