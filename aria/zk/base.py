"""ZK proof layer — base abstractions for ARIA's zero-knowledge extension.

Architecture overview
---------------------
ARIA's ZK layer adds cryptographic proof-of-computation on top of the existing
pre-commitment protocol.  Instead of only anchoring *that* a decision happened,
it proves *which* computation produced it.

Three tiers, chosen by model size and proving budget:

  Tier 1 — FULL_ZK      (≤10M params, EZKL/Halo2):
    Full ZK proof that model(input) = output.  No trust in the operator required.
    Proving time: seconds to minutes.

  Tier 2 — COMMITMENT   (any size, CommitmentProver):
    Keyed hash commitment anchored to the epoch nonce.  Proves the operator
    committed to the I/O values at epoch-open time; does not prove correctness
    of the computation itself.  Proving time: <1ms.

  Tier 3 — TEE          (1B+ params, future):
    Intel SGX / AMD SEV attestation quote embedded in the EPOCH_CLOSE payload.
    Hardware-rooted integrity without ZK overhead.

The ProverInterface is backend-agnostic.  Adding a new proving system
(Groth16, STARKs, Nova recursive, SP1 zkVM…) requires only a new
ProverInterface subclass.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.hasher import canonical_json


class ProverTier(str, Enum):
    """Tiered proving strategy based on model size and capability."""

    FULL_ZK = "full_zk"        # Cryptographic proof of entire inference
    COMMITMENT = "commitment"   # Hash-based commitment (any model size)
    TEE = "tee"                 # Trusted Execution Environment attestation


# ---------------------------------------------------------------------------
# Key material dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProvingKey:
    """Per-model proving key.  Kept private by the operator.

    Generated once per model version via ``ProverInterface.setup()``.
    Large keys (EZKL can produce 100MB+ pk files) should be cached to disk.
    """

    pk_bytes: bytes
    model_hash: str       # sha256 of the model file — ties pk to a specific version
    proving_system: str   # "halo2", "groth16", "stark", "mock"
    metadata: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """SHA-256 fingerprint of the proving key bytes."""
        return "sha256:" + hashlib.sha256(self.pk_bytes).hexdigest()


@dataclass
class VerifyingKey:
    """Per-model verifying key.  Public — safe to publish on BSV.

    The vk_digest is stored in EPOCH_OPEN so regulators can retrieve the
    full key and verify proofs independently without operator cooperation.
    """

    vk_bytes: bytes
    model_hash: str
    proving_system: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """SHA-256 fingerprint — used as the on-chain vk identifier."""
        return "sha256:" + hashlib.sha256(self.vk_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vk_hex": self.vk_bytes.hex(),
            "model_hash": self.model_hash,
            "proving_system": self.proving_system,
            "vk_digest": self.digest(),
        }


# ---------------------------------------------------------------------------
# ZKProof
# ---------------------------------------------------------------------------


@dataclass
class ZKProof:
    """A zero-knowledge proof of model inference.

    Attributes:
        proof_bytes:    Serialized proof (format depends on ``proving_system``).
        public_inputs:  Public signals visible to the verifier — typically
                        [input_hash, output_hash, model_hash] as hex strings.
        proving_system: ZK backend that produced this proof.
        tier:           ProverTier value as string.
        model_hash:     SHA-256 of the model file (matches ProvingKey.model_hash).
        prover_version: Version string of the prover implementation.
        epoch_id:       Epoch this proof belongs to.
        record_id:      ARIA record_id (None for aggregate proofs).
    """

    proof_bytes: bytes
    public_inputs: list[str]
    proving_system: str
    tier: str
    model_hash: str
    prover_version: str
    epoch_id: str
    record_id: str | None = None

    def digest(self) -> str:
        """Canonical SHA-256 fingerprint of this proof."""
        msg = canonical_json({
            "proof_hex": self.proof_bytes.hex(),
            "public_inputs": sorted(self.public_inputs),
            "proving_system": self.proving_system,
            "model_hash": self.model_hash,
        })
        return "sha256:" + hashlib.sha256(msg).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_hex": self.proof_bytes.hex(),
            "public_inputs": self.public_inputs,
            "proving_system": self.proving_system,
            "tier": self.tier,
            "model_hash": self.model_hash,
            "prover_version": self.prover_version,
            "epoch_id": self.epoch_id,
            "record_id": self.record_id,
            "digest": self.digest(),
        }


# ---------------------------------------------------------------------------
# ProverInterface
# ---------------------------------------------------------------------------


class ProverInterface(ABC):
    """Abstract interface for ZK proving backends.

    Concrete implementations
    ------------------------
    MockProver         — deterministic mock for testing (no ZK guarantees).
    EZKLProver         — EZKL-based Halo2/KZG prover for ONNX models ≤10M params.
    CommitmentProver   — keyed-hash commitment for any model size.

    Future
    ------
    RiscZeroProver     — zkVM-based prover for arbitrary Python computation.
    NovaProver         — Nova folding scheme for recursive aggregation.
    TEEProver          — Intel SGX / AMD SEV attestation integration.
    """

    @abstractmethod
    async def setup(
        self,
        model_id: str,
        model_path: str,
    ) -> tuple[ProvingKey, VerifyingKey]:
        """Compile a model to a ZK circuit and generate proving/verifying keys.

        This is the most expensive operation — typically run once per model
        version and the keys cached.  EZKL setup can take 30–300 seconds
        depending on model size.

        Args:
            model_id:   Logical name for this model (must match model_hashes key).
            model_path: Path to the model file (ONNX, .pkl, .pt, .h5, …).

        Returns:
            (ProvingKey, VerifyingKey)
        """

    @abstractmethod
    async def prove(
        self,
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        pk: ProvingKey,
        record_id: str,
        epoch_id: str,
    ) -> ZKProof:
        """Generate a ZK proof that model(input) = output.

        The proof is generated per inference.  For FULL_ZK tier this commits to
        the actual computation; for COMMITMENT tier it commits to the I/O hashes
        under the epoch nonce.

        Args:
            model_id:    Logical model identifier.
            input_data:  Sanitized (PII-stripped) input dict.
            output_data: Model output dict.
            pk:          Proving key from ``setup()``.
            record_id:   ARIA record_id for this inference.
            epoch_id:    Current epoch identifier.

        Returns:
            ZKProof binding (model, input, output) together.
        """

    @abstractmethod
    def verify(self, proof: ZKProof, vk: VerifyingKey) -> bool:
        """Verify a ZK proof against a public verifying key.

        This method is synchronous and should complete in milliseconds.
        It is designed to be called by regulators and third-party auditors
        without any cooperation from the operator.

        Args:
            proof: The proof produced by ``prove()``.
            vk:    The public verifying key from ``setup()``.

        Returns:
            True if the proof is cryptographically valid, False otherwise.
        """

    @property
    @abstractmethod
    def proving_system(self) -> str:
        """Name of the proving system (e.g., ``"halo2"``, ``"mock"``)."""

    @property
    @abstractmethod
    def tier(self) -> str:
        """ProverTier as string (``"full_zk"``, ``"commitment"``, ``"tee"``)."""

    @property
    def max_model_params(self) -> int | None:
        """Maximum supported model parameter count.  None = unlimited."""
        return None
