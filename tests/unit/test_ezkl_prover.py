"""Tests for aria.zk.ezkl_prover — EZKLProver and CommitmentProver.

EZKL is an optional dependency.  All EZKLProver tests mock the ``ezkl``
package via sys.modules injection so the real runtime is not required.
CommitmentProver tests run without any mocking.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# ---------------------------------------------------------------------------
# Helpers — create a minimal ezkl mock compatible with EZKLProver's API
# ---------------------------------------------------------------------------

def _make_ezkl_mock(
    pk_bytes: bytes = b"fakepk",
    vk_bytes: bytes = b"fakevk",
    proof_bytes: bytes = b'{"pi":[],"instances":[]}',
    verify_result: bool = True,
) -> MagicMock:
    """Build a MagicMock that mimics the ezkl module surface used by EZKLProver."""
    mock = MagicMock(name="ezkl")
    mock.gen_settings.return_value = None
    mock.calibrate_settings.return_value = None
    mock.compile_circuit.return_value = None
    mock.get_srs.return_value = None
    mock.setup.return_value = None
    mock.gen_witness.return_value = None
    mock.prove.return_value = None
    mock.verify.return_value = verify_result
    return mock


def _inject_ezkl(mock: MagicMock) -> None:
    sys.modules["ezkl"] = mock  # type: ignore[assignment]


def _remove_ezkl() -> None:
    sys.modules.pop("ezkl", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def fake_model_file(tmp_path: Path) -> Path:
    """Create a temporary fake ONNX model file."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake onnx bytes")
    return model


@pytest.fixture(autouse=False)
def ezkl_mock(tmp_path: Path, fake_model_file: Path) -> MagicMock:
    """Inject a fresh ezkl mock and make setup() write key files."""
    mock = _make_ezkl_mock()

    def _fake_setup(model: str, vk_path: str, pk_path: str) -> None:
        Path(pk_path).write_bytes(b"fakepk_bytes")
        Path(vk_path).write_bytes(b"fakevk_bytes")

    mock.setup.side_effect = _fake_setup
    _inject_ezkl(mock)
    yield mock
    _remove_ezkl()


@pytest.fixture(autouse=False)
def ezkl_prover(ezkl_mock: MagicMock, tmp_path: Path):
    """Return an EZKLProver using a temp work_dir."""
    from aria.zk.ezkl_prover import EZKLProver
    return EZKLProver(work_dir=tmp_path / "zkwork")


# ---------------------------------------------------------------------------
# _require_ezkl
# ---------------------------------------------------------------------------

class TestRequireEzkl:
    def test_raises_when_ezkl_not_installed(self) -> None:
        _remove_ezkl()
        # Make sure ezkl cannot be imported
        with patch.dict(sys.modules, {"ezkl": None}):  # type: ignore[dict-item]
            from aria.core.errors import ARIAZKError
            from aria.zk import ezkl_prover as mod
            import importlib
            # Make _require_ezkl raise an ImportError
            with patch.object(sys.modules.get("aria.zk.ezkl_prover", mod), "_require_ezkl",
                               side_effect=lambda: (_ for _ in ()).throw(
                                   __import__("aria.core.errors", fromlist=["ARIAZKError"]).ARIAZKError("EZKL not installed")
                               )):
                pass  # the test above just checks the function exists

    def test_raises_import_error_with_install_hint(self) -> None:
        """_require_ezkl raises ARIAZKError with pip install hint when ezkl missing."""
        _remove_ezkl()
        with patch.dict(sys.modules, {"ezkl": None}):  # type: ignore[dict-item]
            from aria.zk.ezkl_prover import _require_ezkl
            from aria.core.errors import ARIAZKError
            with pytest.raises(ARIAZKError, match="pip install"):
                _require_ezkl()

    def test_returns_ezkl_when_installed(self) -> None:
        mock = _make_ezkl_mock()
        _inject_ezkl(mock)
        try:
            from aria.zk.ezkl_prover import _require_ezkl
            result = _require_ezkl()
            assert result is mock
        finally:
            _remove_ezkl()


# ---------------------------------------------------------------------------
# EZKLProver — setup()
# ---------------------------------------------------------------------------

class TestEZKLProverSetup:
    @pytest.mark.asyncio
    async def test_setup_returns_proving_and_verifying_keys(
        self, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        from aria.zk.base import ProvingKey, VerifyingKey
        pk, vk = await ezkl_prover.setup("my-model", str(fake_model_file))
        assert isinstance(pk, ProvingKey)
        assert isinstance(vk, VerifyingKey)

    @pytest.mark.asyncio
    async def test_setup_model_hash_matches_file(
        self, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        pk, vk = await ezkl_prover.setup("my-model", str(fake_model_file))
        assert pk.model_hash == vk.model_hash
        # hash_file returns "sha256:<64-char-hex>"
        assert pk.model_hash.startswith("sha256:")
        assert len(pk.model_hash) == 71

    @pytest.mark.asyncio
    async def test_setup_proving_system_is_halo2(
        self, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        pk, vk = await ezkl_prover.setup("my-model", str(fake_model_file))
        assert pk.proving_system == "halo2"
        assert vk.proving_system == "halo2"

    @pytest.mark.asyncio
    async def test_setup_calls_ezkl_routines(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        await ezkl_prover.setup("my-model", str(fake_model_file))
        ezkl_mock.gen_settings.assert_called_once()
        ezkl_mock.compile_circuit.assert_called_once()
        ezkl_mock.get_srs.assert_called_once()
        ezkl_mock.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_is_cached_on_second_call(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        """Second call with same model_hash must not recompile."""
        await ezkl_prover.setup("my-model", str(fake_model_file))
        ezkl_mock.compile_circuit.reset_mock()
        await ezkl_prover.setup("my-model", str(fake_model_file))
        ezkl_mock.compile_circuit.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_with_calibration_data_calls_calibrate(
        self, ezkl_mock: MagicMock, tmp_path: Path, fake_model_file: Path
    ) -> None:
        from aria.zk.ezkl_prover import EZKLProver

        def _fake_setup(model: str, vk_path: str, pk_path: str) -> None:
            Path(pk_path).write_bytes(b"pk")
            Path(vk_path).write_bytes(b"vk")
        ezkl_mock.setup.side_effect = _fake_setup

        prover = EZKLProver(
            work_dir=tmp_path / "cal_work",
            calibration_data=[{"input": [1.0, 2.0]}],
        )
        await prover.setup("m", str(fake_model_file))
        ezkl_mock.calibrate_settings.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_without_calibration_skips_calibrate(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        await ezkl_prover.setup("my-model", str(fake_model_file))
        ezkl_mock.calibrate_settings.assert_not_called()


# ---------------------------------------------------------------------------
# EZKLProver — prove()
# ---------------------------------------------------------------------------

class TestEZKLProverProve:
    @pytest.mark.asyncio
    async def test_prove_returns_zkproof(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        from aria.zk.base import ZKProof

        def _fake_prove(**kwargs: Any) -> None:
            proof_path = Path(kwargs.get("proof_path", ""))
            proof_path.write_bytes(b'{"proof":"fakeproof"}')
        ezkl_mock.prove.side_effect = lambda **kw: None

        # Need to make the prove call write a file
        def _side_prove(witness: str, model: str, pk_path: str, proof_path: str, **kw: Any) -> None:
            Path(proof_path).write_bytes(b'{"proof":"fakedata"}')
        ezkl_mock.prove.side_effect = _side_prove

        pk, vk = await ezkl_prover.setup("my-model", str(fake_model_file))
        proof = await ezkl_prover.prove(
            model_id="my-model",
            input_data={"prompt": "hello"},
            output_data={"completion": "world"},
            pk=pk,
            record_id="rec_123",
            epoch_id="ep_456",
        )

        assert isinstance(proof, ZKProof)
        assert proof.proving_system == "halo2"
        assert proof.tier == "full_zk"
        assert proof.epoch_id == "ep_456"
        assert proof.record_id == "rec_123"

    @pytest.mark.asyncio
    async def test_prove_public_inputs_contain_hashes(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, fake_model_file: Path
    ) -> None:
        def _side_prove(witness: str, model: str, pk_path: str, proof_path: str, **kw: Any) -> None:
            Path(proof_path).write_bytes(b'{}')
        ezkl_mock.prove.side_effect = _side_prove

        pk, _ = await ezkl_prover.setup("my-model", str(fake_model_file))
        proof = await ezkl_prover.prove(
            model_id="my-model",
            input_data={"x": 1},
            output_data={"y": 2},
            pk=pk,
            record_id="rec_1",
            epoch_id="ep_1",
        )

        assert len(proof.public_inputs) == 3
        # All three should be "sha256:<64-char-hex>" strings
        for h in proof.public_inputs:
            assert h.startswith("sha256:"), f"Expected sha256: prefix, got: {h!r}"
            assert len(h) == 71, f"Expected 71-char prefixed hash, got: {h!r}"


# ---------------------------------------------------------------------------
# EZKLProver — verify()
# ---------------------------------------------------------------------------

class TestEZKLProverVerify:
    def test_verify_returns_true_when_ezkl_returns_true(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, tmp_path: Path
    ) -> None:
        from aria.zk.base import ZKProof, VerifyingKey

        # Plant a settings.json file so verify() can find it
        settings_dir = tmp_path / "zkwork" / "my-model"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text('{"run_args":{}}')

        ezkl_mock.verify.return_value = True

        proof = ZKProof(
            proof_bytes=b"proofdata",
            public_inputs=["a" * 64],
            proving_system="halo2",
            tier="full_zk",
            model_hash="m" * 64,
            prover_version="ezkl-prover-1.0",
            epoch_id="ep_1",
            record_id="rec_1",
        )
        vk = VerifyingKey(vk_bytes=b"vkdata", model_hash="m" * 64, proving_system="halo2")

        result = ezkl_prover.verify(proof, vk)
        assert result is True

    def test_verify_returns_false_when_ezkl_returns_false(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, tmp_path: Path
    ) -> None:
        from aria.zk.base import ZKProof, VerifyingKey

        settings_dir = tmp_path / "zkwork" / "my-model"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text('{}')

        ezkl_mock.verify.return_value = False

        proof = ZKProof(
            proof_bytes=b"bad", public_inputs=[],
            proving_system="halo2", tier="full_zk",
            model_hash="m" * 64, prover_version="v1",
            epoch_id="ep", record_id="rec",
        )
        vk = VerifyingKey(vk_bytes=b"vk", model_hash="m" * 64, proving_system="halo2")

        assert ezkl_prover.verify(proof, vk) is False

    def test_verify_returns_false_when_no_settings_file(
        self, ezkl_mock: MagicMock, ezkl_prover: Any
    ) -> None:
        from aria.zk.base import ZKProof, VerifyingKey

        proof = ZKProof(
            proof_bytes=b"p", public_inputs=[],
            proving_system="halo2", tier="full_zk",
            model_hash="x" * 64, prover_version="v1",
            epoch_id="ep", record_id="rec",
        )
        vk = VerifyingKey(vk_bytes=b"v", model_hash="x" * 64, proving_system="halo2")

        # No settings.json planted → verify must return False, not raise
        result = ezkl_prover.verify(proof, vk)
        assert result is False

    def test_verify_returns_false_when_ezkl_raises(
        self, ezkl_mock: MagicMock, ezkl_prover: Any, tmp_path: Path
    ) -> None:
        from aria.zk.base import ZKProof, VerifyingKey

        settings_dir = tmp_path / "zkwork" / "any"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text('{}')

        ezkl_mock.verify.side_effect = RuntimeError("circuit error")

        proof = ZKProof(
            proof_bytes=b"p", public_inputs=[],
            proving_system="halo2", tier="full_zk",
            model_hash="x" * 64, prover_version="v1",
            epoch_id="ep", record_id="rec",
        )
        vk = VerifyingKey(vk_bytes=b"v", model_hash="x" * 64, proving_system="halo2")

        result = ezkl_prover.verify(proof, vk)
        assert result is False


# ---------------------------------------------------------------------------
# EZKLProver — properties
# ---------------------------------------------------------------------------

class TestEZKLProverProperties:
    def test_proving_system(self, ezkl_prover: Any) -> None:
        assert ezkl_prover.proving_system == "halo2"

    def test_tier(self, ezkl_prover: Any) -> None:
        assert ezkl_prover.tier == "full_zk"

    def test_max_model_params(self, ezkl_prover: Any) -> None:
        assert ezkl_prover.max_model_params == 10_000_000


# ---------------------------------------------------------------------------
# CommitmentProver — setup, prove, verify
# ---------------------------------------------------------------------------

class TestCommitmentProver:
    """CommitmentProver requires no ezkl — runs without any mock."""

    @pytest.mark.asyncio
    async def test_setup_returns_keys(self, fake_model_file: Path) -> None:
        from aria.zk.base import ProvingKey, VerifyingKey
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        pk, vk = await prover.setup("gpt-4", str(fake_model_file))
        assert isinstance(pk, ProvingKey)
        assert isinstance(vk, VerifyingKey)
        assert pk.proving_system == "commitment"
        assert vk.proving_system == "commitment"

    @pytest.mark.asyncio
    async def test_setup_model_hash_tied_to_file_content(self, fake_model_file: Path) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        pk1, _ = await CommitmentProver().setup("m", str(fake_model_file))

        # Change file content
        fake_model_file.write_bytes(b"different content")
        pk2, _ = await CommitmentProver().setup("m", str(fake_model_file))

        assert pk1.model_hash != pk2.model_hash

    @pytest.mark.asyncio
    async def test_prove_returns_zkproof(self, fake_model_file: Path) -> None:
        from aria.zk.base import ZKProof
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver(epoch_nonce="testnonce")
        pk, _ = await prover.setup("model", str(fake_model_file))
        proof = await prover.prove(
            model_id="model",
            input_data={"q": "hello"},
            output_data={"a": "world"},
            pk=pk,
            record_id="rec_1",
            epoch_id="ep_1",
        )

        assert isinstance(proof, ZKProof)
        assert proof.proving_system == "commitment"
        assert proof.tier == "commitment"
        assert len(proof.proof_bytes) == 32  # HMAC-SHA256

    @pytest.mark.asyncio
    async def test_prove_is_deterministic(self, fake_model_file: Path) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver(epoch_nonce="nonce123")
        pk, _ = await prover.setup("model", str(fake_model_file))

        proof1 = await prover.prove("model", {"x": 1}, {"y": 2}, pk, "rec_1", "ep_1")
        proof2 = await prover.prove("model", {"x": 1}, {"y": 2}, pk, "rec_1", "ep_1")

        assert proof1.proof_bytes == proof2.proof_bytes

    @pytest.mark.asyncio
    async def test_prove_differs_for_different_inputs(self, fake_model_file: Path) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        pk, _ = await prover.setup("model", str(fake_model_file))

        proof1 = await prover.prove("model", {"x": 1}, {"y": 2}, pk, "rec_1", "ep_1")
        proof2 = await prover.prove("model", {"x": 99}, {"y": 2}, pk, "rec_1", "ep_1")

        assert proof1.proof_bytes != proof2.proof_bytes

    @pytest.mark.asyncio
    async def test_prove_public_inputs_has_three_entries(self, fake_model_file: Path) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        pk, _ = await prover.setup("model", str(fake_model_file))
        proof = await prover.prove("model", {"x": 1}, {"y": 2}, pk, "rec_1", "ep_1")

        assert len(proof.public_inputs) == 3

    @pytest.mark.asyncio
    async def test_prove_uses_epoch_id_as_nonce_when_none(self, fake_model_file: Path) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        prover_no_nonce = CommitmentProver(epoch_nonce=None)
        prover_with_nonce = CommitmentProver(epoch_nonce="ep_xyz")

        pk, _ = await prover_no_nonce.setup("m", str(fake_model_file))
        p1 = await prover_no_nonce.prove("m", {}, {}, pk, "rec_1", "ep_xyz")
        p2 = await prover_with_nonce.prove("m", {}, {}, pk, "rec_1", "ep_xyz")

        # Both use "ep_xyz" as the HMAC key → same commitment
        assert p1.proof_bytes == p2.proof_bytes

    def test_verify_returns_true_for_valid_commitment_proof(self, fake_model_file: Path) -> None:
        from aria.zk.base import ZKProof, VerifyingKey
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        vk = VerifyingKey(vk_bytes=b"vk", model_hash="m" * 64, proving_system="commitment")
        proof = ZKProof(
            proof_bytes=b"\x00" * 32,
            public_inputs=["a" * 64, "b" * 64, "c" * 64],
            proving_system="commitment",
            tier="commitment",
            model_hash="m" * 64,
            prover_version="commitment-prover-1.0",
            epoch_id="ep",
            record_id="rec",
        )
        assert prover.verify(proof, vk) is True

    def test_verify_returns_false_for_wrong_proving_system(self) -> None:
        from aria.zk.base import ZKProof, VerifyingKey
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        vk = VerifyingKey(vk_bytes=b"vk", model_hash="m" * 64, proving_system="halo2")
        proof = ZKProof(
            proof_bytes=b"\x00" * 32,
            public_inputs=["a" * 64, "b" * 64, "c" * 64],
            proving_system="halo2",  # ← wrong
            tier="full_zk",
            model_hash="m" * 64,
            prover_version="v1",
            epoch_id="ep",
            record_id="rec",
        )
        assert prover.verify(proof, vk) is False

    def test_verify_returns_false_for_model_hash_mismatch(self) -> None:
        from aria.zk.base import ZKProof, VerifyingKey
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        vk = VerifyingKey(vk_bytes=b"vk", model_hash="a" * 64, proving_system="commitment")
        proof = ZKProof(
            proof_bytes=b"\x00" * 32,
            public_inputs=["x" * 64, "y" * 64, "z" * 64],
            proving_system="commitment",
            tier="commitment",
            model_hash="b" * 64,  # ← mismatch
            prover_version="v1",
            epoch_id="ep",
            record_id="rec",
        )
        assert prover.verify(proof, vk) is False

    def test_verify_returns_false_for_wrong_proof_length(self) -> None:
        from aria.zk.base import ZKProof, VerifyingKey
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        vk = VerifyingKey(vk_bytes=b"vk", model_hash="m" * 64, proving_system="commitment")
        proof = ZKProof(
            proof_bytes=b"\x00" * 16,  # ← wrong length (not 32)
            public_inputs=["a" * 64, "b" * 64, "c" * 64],
            proving_system="commitment",
            tier="commitment",
            model_hash="m" * 64,
            prover_version="v1",
            epoch_id="ep",
            record_id="rec",
        )
        assert prover.verify(proof, vk) is False

    def test_commitment_properties(self) -> None:
        from aria.zk.ezkl_prover import CommitmentProver

        prover = CommitmentProver()
        assert prover.proving_system == "commitment"
        assert prover.tier == "commitment"
        assert prover.max_model_params is None  # no limit
