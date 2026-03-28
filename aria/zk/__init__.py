"""aria.zk — Zero-knowledge proof layer for ARIA.

This package implements the ZK extension to BRC-121:

  - ProverInterface / ZKProof / ProvingKey / VerifyingKey  (base.py)
  - MockProver                                              (mock_prover.py)
  - EZKLProver / CommitmentProver                          (ezkl_prover.py)
  - MerkleAggregator / NovaAggregator / AggregateProof     (aggregate.py)
  - Claims DSL (7 EU AI Act claim types)                   (claims.py)
  - EpochStatement                                         (statement.py)

Quick start::

    from aria.zk import MockProver, EpochStatement
    from aria.zk.claims import ConfidencePercentile, ModelUnchanged

    config = AuditConfig(
        system_id="my-system",
        bsv_key="...",
        zk_prover=MockProver(),
        model_paths={"my-model": "/path/to/model.onnx"},
        zk_claims=[
            ConfidencePercentile(p=99, threshold=0.85),
            ModelUnchanged(),
        ],
    )
"""

from .aggregate import AggregateProof, MerkleAggregator, NovaAggregator, ProofAggregatorInterface
from .base import ProvingKey, ProverInterface, ProverTier, VerifyingKey, ZKProof
from .claims import (
    AllModelsRegistered,
    Claim,
    ClaimResult,
    ConfidencePercentile,
    LatencyBound,
    ModelUnchanged,
    NoPIIInInputs,
    OutputDistribution,
    RecordCountRange,
)
from .ezkl_prover import CommitmentProver, EZKLProver
from .mock_prover import MockProver
from .statement import EpochStatement

__all__ = [
    # base
    "ProverTier",
    "ProvingKey",
    "VerifyingKey",
    "ZKProof",
    "ProverInterface",
    # provers
    "MockProver",
    "EZKLProver",
    "CommitmentProver",
    # aggregate
    "AggregateProof",
    "ProofAggregatorInterface",
    "MerkleAggregator",
    "NovaAggregator",
    # claims
    "Claim",
    "ClaimResult",
    "ConfidencePercentile",
    "ModelUnchanged",
    "NoPIIInInputs",
    "OutputDistribution",
    "LatencyBound",
    "RecordCountRange",
    "AllModelsRegistered",
    # statement
    "EpochStatement",
]
