"""Proof aggregation — combine N inference proofs into a single on-chain commitment.

Aggregation strategies
----------------------

MerkleAggregator (production-ready):
  Builds a Merkle tree over individual proof digests.  The root is a single
  32-byte hash that commits to all proofs.  Inclusion of any individual proof
  can be verified in O(log N) time.

  Suitable for: all production use cases today.
  On-chain footprint: one sha256 hash.

NovaAggregator (experimental, stub):
  Nova folding scheme (Kothapalli et al., 2022) generates a single succinct
  proof that verifies N inference proofs simultaneously.  The aggregated proof
  is O(1) size regardless of N.

  Suitable for: when verifier computation overhead matters.
  Status: Python bindings not yet stable.  Stub provided for future integration.

PlonkRecursiveAggregator (experimental, stub):
  Recursive Plonk aggregation — each step folds the previous aggregate.
  Status: requires Halo2 recursive configuration; planned for v0.3.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.hasher import canonical_json
from ..core.merkle import ARIAMerkleTree, verify_proof as verify_merkle_proof
from .base import ZKProof


# ---------------------------------------------------------------------------
# AggregateProof
# ---------------------------------------------------------------------------


@dataclass
class AggregateProof:
    """Aggregation of N individual ZKProofs into a single compact commitment.

    Included in the EPOCH_CLOSE BSV payload so regulators can independently
    verify that the operator committed to all N inference proofs at close time.

    Attributes:
        proofs_merkle_root: Merkle root of all individual proof digests.
        n_proofs:           Number of proofs aggregated.
        aggregation_scheme: ``"merkle"``, ``"nova"``, or ``"plonk_recursive"``.
        aggregate_bytes:    Serialized aggregate (scheme-dependent).
        epoch_id:           The epoch these proofs belong to.
    """

    proofs_merkle_root: str
    n_proofs: int
    aggregation_scheme: str
    aggregate_bytes: bytes
    epoch_id: str

    def digest(self) -> str:
        """SHA-256 fingerprint of the aggregate commitment."""
        return "sha256:" + hashlib.sha256(self.aggregate_bytes).hexdigest()

    def to_dict(self) -> dict:
        return {
            "proofs_merkle_root": self.proofs_merkle_root,
            "n_proofs": self.n_proofs,
            "aggregation_scheme": self.aggregation_scheme,
            "aggregate_digest": self.digest(),
            "epoch_id": self.epoch_id,
        }


# ---------------------------------------------------------------------------
# Aggregator interface
# ---------------------------------------------------------------------------


class ProofAggregatorInterface(ABC):
    """Abstract aggregator interface."""

    @abstractmethod
    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        """Aggregate N proofs into a single AggregateProof."""

    @abstractmethod
    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Verify that an AggregateProof commits to exactly the given proofs."""

    def verify_membership(
        self,
        agg: AggregateProof,
        proof: ZKProof,
        all_proofs: list[ZKProof],
    ) -> bool:
        """Verify that a single proof is a member of the aggregate.

        Default implementation recomputes the aggregate and checks membership.
        Override for more efficient implementations (e.g., Merkle path lookup).
        """
        digests = [p.digest() for p in all_proofs]
        return proof.digest() in digests and self.verify_aggregate(agg, all_proofs)


# ---------------------------------------------------------------------------
# MerkleAggregator
# ---------------------------------------------------------------------------


class MerkleAggregator(ProofAggregatorInterface):
    """Production-ready Merkle-based proof aggregation.

    Builds an ARIAMerkleTree (RFC 6962, second-preimage protected) over the
    digest of each individual ZKProof.  The root is stored in the EPOCH_CLOSE
    BSV transaction.

    Membership proof: use ``aria.core.merkle.verify_proof`` with the Merkle
    path for any individual proof digest.

    On-chain footprint: one SHA-256 hash per epoch (no matter how many
    inferences: 10, 10_000, or 10_000_000).
    """

    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        if not proofs:
            empty_root = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            return AggregateProof(
                proofs_merkle_root=empty_root,
                n_proofs=0,
                aggregation_scheme="merkle",
                aggregate_bytes=bytes.fromhex(empty_root[7:]),
                epoch_id=epoch_id,
            )

        tree = ARIAMerkleTree()
        for p in sorted(proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())

        root = tree.root()
        root_hex = bytes.fromhex(root[7:])  # strip "sha256:" prefix

        return AggregateProof(
            proofs_merkle_root=root,
            n_proofs=len(proofs),
            aggregation_scheme="merkle",
            aggregate_bytes=root_hex,
            epoch_id=epoch_id,
        )

    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Recompute the Merkle root and compare to the aggregate."""
        if agg.aggregation_scheme != "merkle":
            return False

        recomputed = self.aggregate(proofs, agg.epoch_id)
        return recomputed.proofs_merkle_root == agg.proofs_merkle_root

    def membership_path(self, proof: ZKProof, all_proofs: list[ZKProof]):  # type: ignore[return]
        """Return the Merkle path for a specific proof (for independent verification)."""
        tree = ARIAMerkleTree()
        for p in sorted(all_proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())
        return tree.proof(proof.digest())


# ---------------------------------------------------------------------------
# NovaAggregator (experimental stub)
# ---------------------------------------------------------------------------


class NovaAggregator(ProofAggregatorInterface):
    """Nova folding scheme aggregator — EXPERIMENTAL, not yet production-ready.

    Nova (Kothapalli, Setty, Tzialla, 2022) is an incrementally verifiable
    computation (IVC) scheme that folds N proof instances into a single
    succinct proof with O(1) verifier cost.

    When available, this will provide:
      - One ~2KB proof verifies all N inference proofs simultaneously
      - Sub-millisecond verification regardless of N
      - No trusted setup required

    Status: Python bindings for Nova are not yet in stable release.
            This stub raises ARIAZKError until bindings are available.
            Fallback: use MerkleAggregator.

    References:
      https://eprint.iacr.org/2021/370 (Nova paper)
      https://github.com/microsoft/nova (Rust implementation)
    """

    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        from ..core.errors import ARIAZKError
        raise ARIAZKError(
            "NovaAggregator is not yet implemented. "
            "Use MerkleAggregator for production deployments. "
            "Track progress: https://github.com/microsoft/nova"
        )

    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        from ..core.errors import ARIAZKError
        raise ARIAZKError("NovaAggregator is not yet implemented.")
