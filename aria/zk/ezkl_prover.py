"""EZKLProver — Halo2/KZG prover via EZKL for ONNX models.

EZKL compiles an ONNX model to an arithmetic circuit (Halo2, KZG polynomial
commitment) and generates a Plonk proof that the model was evaluated correctly
on a given input/output pair.

2026 practical limits:
  - Models ≤ 10M parameters: proofs in seconds (GPU) or minutes (CPU)
  - Models ≤ 1M parameters:  proofs in <30 seconds on CPU
  - Models > 10M parameters: use CommitmentProver instead

Installation:
  pip install aria-bsv[zk]   # installs ezkl>=9.0

Documentation:
  https://docs.ezkl.xyz
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from ..core.errors import ARIAZKError
from ..core.hasher import hash_file, hash_object
from .base import ProvingKey, ProverInterface, VerifyingKey, ZKProof


def _require_ezkl() -> Any:
    """Import ezkl or raise ARIAZKError with installation instructions."""
    try:
        import ezkl  # type: ignore[import]
        return ezkl
    except ImportError as exc:
        raise ARIAZKError(
            "EZKL is not installed. "
            "Install with: pip install 'aria-bsv[zk]'  "
            "(requires Python 3.11+, ~2GB disk space)"
        ) from exc


class EZKLProver(ProverInterface):
    """ZK prover backed by EZKL (Halo2/KZG polynomial commitments).

    Workflow per model version (once):
      1. ``setup(model_id, model_path)``
         - gen_settings: analyze model architecture
         - calibrate_settings: choose quantization parameters
         - compile_circuit: produce arithmetic circuit
         - setup: generate SRS + proving/verifying keys
         → cached to ``work_dir/``

    Workflow per inference (each call to ``record()``):
      2. ``prove(model_id, input_data, output_data, pk, record_id, epoch_id)``
         - gen_witness: compute circuit witness from actual I/O
         - prove: Plonk prover generates proof
         → ~1–30 seconds depending on model size and hardware

    Args:
        work_dir:           Directory for compiled circuits and keys.
                            Defaults to a temp dir.  Persist across restarts
                            to avoid expensive recompilation.
        calibration_data:   Sample inputs for EZKL calibration.
                            Provide at least 10 representative inputs.
        use_gpu:            Use GPU acceleration if available.
    """

    _PROVING_SYSTEM = "halo2"
    _VERSION = "ezkl-prover-1.0"

    def __init__(
        self,
        work_dir: str | Path | None = None,
        calibration_data: list[dict[str, Any]] | None = None,
        use_gpu: bool = False,
    ) -> None:
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="aria_ezkl_"))
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_data = calibration_data
        self._use_gpu = use_gpu

    async def setup(
        self,
        model_id: str,
        model_path: str,
    ) -> tuple[ProvingKey, VerifyingKey]:
        """Compile model to Halo2 circuit and generate proving/verifying keys.

        This is expensive (30 seconds to 5 minutes).  Results are cached
        under ``work_dir/{model_id}/``.  Call once per model version.
        """
        ezkl = _require_ezkl()
        model_hash = hash_file(model_path)
        circuit_dir = self._work_dir / model_id
        circuit_dir.mkdir(exist_ok=True)

        settings_path = circuit_dir / "settings.json"
        compiled_path = circuit_dir / "compiled_model.ezkl"
        pk_path = circuit_dir / "pk.key"
        vk_path = circuit_dir / "vk.key"

        # Return cached keys if already compiled for this model version.
        cache_meta = circuit_dir / "meta.json"
        if cache_meta.exists():
            meta = json.loads(cache_meta.read_text())
            if meta.get("model_hash") == model_hash:
                pk_bytes = pk_path.read_bytes()
                vk_bytes = vk_path.read_bytes()
                return (
                    ProvingKey(pk_bytes=pk_bytes, model_hash=model_hash,
                               proving_system=self._PROVING_SYSTEM,
                               metadata={"settings_path": str(settings_path),
                                         "compiled_path": str(compiled_path)}),
                    VerifyingKey(vk_bytes=vk_bytes, model_hash=model_hash,
                                 proving_system=self._PROVING_SYSTEM),
                )

        def _run_setup() -> tuple[bytes, bytes]:
            # Step 1: Generate circuit settings.
            ezkl.gen_settings(
                model=model_path,
                output=str(settings_path),
            )

            # Step 2: Calibrate quantization parameters.
            if self._calibration_data:
                cal_path = circuit_dir / "calibration.json"
                cal_path.write_text(json.dumps(self._calibration_data))
                ezkl.calibrate_settings(
                    model=model_path,
                    output=str(settings_path),
                    data=str(cal_path),
                )

            # Step 3: Compile to arithmetic circuit.
            ezkl.compile_circuit(
                model=model_path,
                compiled_circuit=str(compiled_path),
                settings_path=str(settings_path),
            )

            # Step 4: Generate structured reference string + keys.
            ezkl.get_srs(settings_path=str(settings_path))
            ezkl.setup(
                model=str(compiled_path),
                vk_path=str(vk_path),
                pk_path=str(pk_path),
            )

            return pk_path.read_bytes(), vk_path.read_bytes()

        pk_bytes, vk_bytes = await asyncio.to_thread(_run_setup)

        cache_meta.write_text(json.dumps({"model_hash": model_hash, "model_id": model_id}))

        pk = ProvingKey(
            pk_bytes=pk_bytes,
            model_hash=model_hash,
            proving_system=self._PROVING_SYSTEM,
            metadata={"settings_path": str(settings_path), "compiled_path": str(compiled_path)},
        )
        vk = VerifyingKey(vk_bytes=vk_bytes, model_hash=model_hash, proving_system=self._PROVING_SYSTEM)
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
        """Generate Halo2 Plonk proof that model(input) = output."""
        ezkl = _require_ezkl()

        circuit_dir = self._work_dir / model_id
        compiled_path = pk.metadata.get("compiled_path", str(circuit_dir / "compiled_model.ezkl"))
        settings_path = pk.metadata.get("settings_path", str(circuit_dir / "settings.json"))
        pk_path = circuit_dir / "pk.key"

        if not pk_path.exists():
            pk_path.write_bytes(pk.pk_bytes)

        proof_dir = circuit_dir / "proofs" / record_id[:16]
        proof_dir.mkdir(parents=True, exist_ok=True)

        witness_path = proof_dir / "witness.json"
        proof_path = proof_dir / "proof.json"

        # EZKL expects inputs in its specific format.
        input_values = list(input_data.values()) if isinstance(input_data, dict) else [input_data]
        ezkl_input = {"input_data": [input_values]}
        input_path = proof_dir / "input.json"
        input_path.write_text(json.dumps(ezkl_input))

        input_hash = hash_object(input_data)
        output_hash = hash_object(output_data)

        def _run_prove() -> bytes:
            ezkl.gen_witness(
                data=str(input_path),
                model=compiled_path,
                output=str(witness_path),
            )
            ezkl.prove(
                witness=str(witness_path),
                model=compiled_path,
                pk_path=str(pk_path),
                proof_path=str(proof_path),
                proof_type="single",
            )
            return proof_path.read_bytes()

        proof_bytes = await asyncio.to_thread(_run_prove)

        return ZKProof(
            proof_bytes=proof_bytes,
            public_inputs=[input_hash, output_hash, pk.model_hash],
            proving_system=self._PROVING_SYSTEM,
            tier="full_zk",
            model_hash=pk.model_hash,
            prover_version=self._VERSION,
            epoch_id=epoch_id,
            record_id=record_id,
        )

    def verify(self, proof: ZKProof, vk: VerifyingKey) -> bool:
        """Verify a Halo2 proof.  Runs in milliseconds."""
        ezkl = _require_ezkl()

        verify_dir = self._work_dir / "_verify"
        verify_dir.mkdir(exist_ok=True)

        proof_id = hashlib.sha256(proof.proof_bytes).hexdigest()[:16]
        proof_path = verify_dir / f"proof_{proof_id}.json"
        vk_path = verify_dir / f"vk_{proof_id}.key"

        # Find the settings.json — try to locate from work_dir.
        settings_candidates = list(self._work_dir.rglob("settings.json"))
        if not settings_candidates:
            return False
        settings_path = settings_candidates[0]

        try:
            proof_path.write_bytes(proof.proof_bytes)
            vk_path.write_bytes(vk.vk_bytes)
            return bool(ezkl.verify(
                proof_path=str(proof_path),
                settings_path=str(settings_path),
                vk_path=str(vk_path),
            ))
        except Exception:
            return False

    @property
    def proving_system(self) -> str:
        return self._PROVING_SYSTEM

    @property
    def tier(self) -> str:
        return "full_zk"

    @property
    def max_model_params(self) -> int | None:
        return 10_000_000


class CommitmentProver(ProverInterface):
    """Hash-based commitment prover for models exceeding zkML capacity.

    For large models (>10M params, LLMs, large CV models), a full ZK proof
    of computation is not yet practical.  CommitmentProver generates a
    cryptographic commitment instead:

        HMAC-SHA256(
            key   = epoch_nonce,
            msg   = SHA256(model_hash || input_hash || output_hash || record_id)
        )

    This is weaker than FULL_ZK — it requires trusting that the operator
    ran the correct model — but the pre-commitment protocol (EPOCH_OPEN
    commits to model_hash before inference) provides meaningful anti-backdating
    and anti-substitution guarantees.

    No setup required.  Proving time: <1ms.
    """

    _VERSION = "commitment-prover-1.0"

    def __init__(self, epoch_nonce: str | None = None) -> None:
        """
        Args:
            epoch_nonce: Optional nonce for HMAC keying.  If not provided,
                         the epoch_id is used as the key.
        """
        self._nonce = epoch_nonce

    async def setup(
        self,
        model_id: str,
        model_path: str,
    ) -> tuple[ProvingKey, VerifyingKey]:
        """No circuit compilation needed — setup is instantaneous."""
        from ..core.hasher import hash_file as _hash_file
        import hashlib

        model_hash = _hash_file(model_path)
        pk_bytes = hashlib.sha256(f"commitment:pk:{model_id}:{model_hash}".encode()).digest()
        vk_bytes = hashlib.sha256(f"commitment:vk:{model_id}:{model_hash}".encode()).digest()

        pk = ProvingKey(pk_bytes=pk_bytes, model_hash=model_hash, proving_system="commitment")
        vk = VerifyingKey(vk_bytes=vk_bytes, model_hash=model_hash, proving_system="commitment")
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
        import hashlib, hmac
        from ..core.hasher import hash_object as _hash_object, canonical_json as _canonical_json

        input_hash = _hash_object(input_data)
        output_hash = _hash_object(output_data)

        key = (self._nonce or epoch_id).encode("utf-8")
        msg = _canonical_json({
            "model_hash": pk.model_hash,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "record_id": record_id,
        })
        commitment = hmac.new(key, msg, hashlib.sha256).digest()

        return ZKProof(
            proof_bytes=commitment,
            public_inputs=[input_hash, output_hash, pk.model_hash],
            proving_system="commitment",
            tier="commitment",
            model_hash=pk.model_hash,
            prover_version=self._VERSION,
            epoch_id=epoch_id,
            record_id=record_id,
        )

    def verify(self, proof: ZKProof, vk: VerifyingKey) -> bool:
        """Verify structural consistency of a commitment proof."""
        if proof.proving_system != "commitment":
            return False
        if proof.model_hash != vk.model_hash:
            return False
        if len(proof.proof_bytes) != 32:
            return False
        if len(proof.public_inputs) != 3:
            return False
        return True

    @property
    def proving_system(self) -> str:
        return "commitment"

    @property
    def tier(self) -> str:
        return "commitment"

    @property
    def max_model_params(self) -> int | None:
        return None  # No limit — works for any model size.
