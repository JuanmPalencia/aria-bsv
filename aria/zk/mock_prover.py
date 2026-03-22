"""MockProver — deterministic test double for the ZK proof layer.

Does NOT provide zero-knowledge guarantees.  The "proof" is simply
SHA-256(pk_digest || input_hash || output_hash || record_id), which
is deterministic and structurally verifiable without real ZK math.

Use this in:
  - Unit and integration tests (fast, no ezkl dependency)
  - CI pipelines
  - Development environments without GPU

Never use MockProver in production — it provides no cryptographic security.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..core.hasher import canonical_json, hash_object
from .base import ProvingKey, ProverInterface, VerifyingKey, ZKProof


class MockProver(ProverInterface):
    """Deterministic mock prover for testing and development.

    The mock proof is structurally identical to a real ZKProof — it has
    proof_bytes, public_inputs, proving_system, etc. — but the proof_bytes
    is a simple SHA-256 digest rather than a real Halo2/Groth16 proof.

    Verification checks structural consistency (right proving_system,
    right model_hash, right public_inputs format) rather than real ZK math.
    """

    _VERSION = "mock-1.0"

    async def setup(
        self,
        model_id: str,
        model_path: str,
    ) -> tuple[ProvingKey, VerifyingKey]:
        """Generate deterministic mock keys from model_id."""
        model_hash = "sha256:" + hashlib.sha256(model_id.encode()).hexdigest()
        pk_bytes = hashlib.sha256(f"pk:{model_id}:{model_hash}".encode()).digest()
        vk_bytes = hashlib.sha256(f"vk:{model_id}:{model_hash}".encode()).digest()

        pk = ProvingKey(pk_bytes=pk_bytes, model_hash=model_hash, proving_system="mock")
        vk = VerifyingKey(vk_bytes=vk_bytes, model_hash=model_hash, proving_system="mock")
        return pk, vk

    async def prove(
        self,
        model_id: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        pk: ProvingKey,
        record_id: str,
        epoch_id: str,
    ) -> ZKProof:
        """Generate a deterministic mock proof.

        The proof bytes are SHA-256(pk_digest || input_hash || output_hash || record_id).
        This is deterministic: given the same inputs, the same proof is produced every time.
        """
        input_hash = hash_object(input_data)
        output_hash = hash_object(output_data)

        msg = canonical_json({
            "pk_digest": pk.digest(),
            "input_hash": input_hash,
            "output_hash": output_hash,
            "record_id": record_id,
            "epoch_id": epoch_id,
        })
        proof_bytes = hashlib.sha256(msg).digest()

        return ZKProof(
            proof_bytes=proof_bytes,
            public_inputs=[input_hash, output_hash, pk.model_hash],
            proving_system="mock",
            tier="full_zk",
            model_hash=pk.model_hash,
            prover_version=self._VERSION,
            epoch_id=epoch_id,
            record_id=record_id,
        )

    def verify(self, proof: ZKProof, vk: VerifyingKey) -> bool:
        """Mock verification: structural consistency check only."""
        if proof.proving_system != "mock":
            return False
        if proof.model_hash != vk.model_hash:
            return False
        if len(proof.proof_bytes) != 32:
            return False
        if len(proof.public_inputs) != 3:
            return False
        # public_inputs[2] must be the model_hash
        if proof.public_inputs[2] != vk.model_hash:
            return False
        return True

    @property
    def proving_system(self) -> str:
        return "mock"

    @property
    def tier(self) -> str:
        return "full_zk"
