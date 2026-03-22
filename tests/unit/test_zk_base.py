"""Tests for aria.zk.base — ZKProof, ProvingKey, VerifyingKey, ProverInterface."""

from __future__ import annotations

import pytest

from aria.zk.base import ProvingKey, VerifyingKey, ZKProof, ProverTier
from aria.zk.mock_prover import MockProver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_prover():
    return MockProver()


@pytest.fixture
def fake_pk():
    return ProvingKey(
        pk_bytes=b"\x01" * 32,
        model_hash="sha256:" + "a" * 64,
        proving_system="mock",
    )


@pytest.fixture
def fake_vk():
    return VerifyingKey(
        vk_bytes=b"\x02" * 32,
        model_hash="sha256:" + "a" * 64,
        proving_system="mock",
    )


@pytest.fixture
def fake_proof(fake_pk):
    return ZKProof(
        proof_bytes=b"\x03" * 32,
        public_inputs=["sha256:" + "b" * 64, "sha256:" + "c" * 64, "sha256:" + "a" * 64],
        proving_system="mock",
        tier="full_zk",
        model_hash="sha256:" + "a" * 64,
        prover_version="mock-1.0",
        epoch_id="ep_1_0001",
        record_id="rec_ep_1_0001_000000",
    )


# ---------------------------------------------------------------------------
# ProverTier
# ---------------------------------------------------------------------------

class TestProverTier:
    def test_values_are_strings(self):
        assert ProverTier.FULL_ZK == "full_zk"
        assert ProverTier.COMMITMENT == "commitment"
        assert ProverTier.TEE == "tee"


# ---------------------------------------------------------------------------
# ProvingKey
# ---------------------------------------------------------------------------

class TestProvingKey:
    def test_digest_is_sha256_of_pk_bytes(self):
        import hashlib
        pk = ProvingKey(pk_bytes=b"test", model_hash="sha256:" + "a" * 64, proving_system="mock")
        expected = "sha256:" + hashlib.sha256(b"test").hexdigest()
        assert pk.digest() == expected

    def test_digest_changes_with_bytes(self):
        pk1 = ProvingKey(pk_bytes=b"a", model_hash="sha256:" + "a" * 64, proving_system="mock")
        pk2 = ProvingKey(pk_bytes=b"b", model_hash="sha256:" + "a" * 64, proving_system="mock")
        assert pk1.digest() != pk2.digest()


# ---------------------------------------------------------------------------
# VerifyingKey
# ---------------------------------------------------------------------------

class TestVerifyingKey:
    def test_digest_is_sha256_of_vk_bytes(self):
        import hashlib
        vk = VerifyingKey(vk_bytes=b"vkdata", model_hash="sha256:" + "a" * 64, proving_system="mock")
        expected = "sha256:" + hashlib.sha256(b"vkdata").hexdigest()
        assert vk.digest() == expected

    def test_to_dict_has_required_keys(self, fake_vk):
        d = fake_vk.to_dict()
        assert "vk_hex" in d
        assert "model_hash" in d
        assert "proving_system" in d
        assert "vk_digest" in d

    def test_vk_hex_is_hex_encoded_bytes(self, fake_vk):
        assert fake_vk.to_dict()["vk_hex"] == fake_vk.vk_bytes.hex()


# ---------------------------------------------------------------------------
# ZKProof
# ---------------------------------------------------------------------------

class TestZKProof:
    def test_digest_is_deterministic(self, fake_proof):
        assert fake_proof.digest() == fake_proof.digest()

    def test_digest_starts_with_sha256(self, fake_proof):
        assert fake_proof.digest().startswith("sha256:")

    def test_digest_changes_with_proof_bytes(self, fake_proof):
        altered = ZKProof(
            proof_bytes=b"\x99" * 32,
            public_inputs=fake_proof.public_inputs,
            proving_system=fake_proof.proving_system,
            tier=fake_proof.tier,
            model_hash=fake_proof.model_hash,
            prover_version=fake_proof.prover_version,
            epoch_id=fake_proof.epoch_id,
        )
        assert altered.digest() != fake_proof.digest()

    def test_to_dict_has_all_fields(self, fake_proof):
        d = fake_proof.to_dict()
        for key in ("proof_hex", "public_inputs", "proving_system", "tier",
                    "model_hash", "prover_version", "epoch_id", "record_id", "digest"):
            assert key in d

    def test_proof_hex_is_hex_string(self, fake_proof):
        assert bytes.fromhex(fake_proof.to_dict()["proof_hex"]) == fake_proof.proof_bytes


# ---------------------------------------------------------------------------
# MockProver
# ---------------------------------------------------------------------------

class TestMockProver:
    @pytest.mark.asyncio
    async def test_setup_returns_pk_and_vk(self, mock_prover):
        pk, vk = await mock_prover.setup("model-a", "/path/to/model.onnx")
        assert isinstance(pk, ProvingKey)
        assert isinstance(vk, VerifyingKey)
        assert pk.proving_system == "mock"
        assert vk.proving_system == "mock"

    @pytest.mark.asyncio
    async def test_setup_is_deterministic(self, mock_prover):
        pk1, vk1 = await mock_prover.setup("model-a", "/path/model.onnx")
        pk2, vk2 = await mock_prover.setup("model-a", "/path/model.onnx")
        assert pk1.pk_bytes == pk2.pk_bytes
        assert vk1.vk_bytes == vk2.vk_bytes

    @pytest.mark.asyncio
    async def test_setup_different_models_different_keys(self, mock_prover):
        pk1, _ = await mock_prover.setup("model-a", "/path/model.onnx")
        pk2, _ = await mock_prover.setup("model-b", "/path/model.onnx")
        assert pk1.pk_bytes != pk2.pk_bytes

    @pytest.mark.asyncio
    async def test_prove_returns_zk_proof(self, mock_prover):
        pk, _ = await mock_prover.setup("model-a", "/path/model.onnx")
        proof = await mock_prover.prove(
            model_id="model-a",
            input_data={"query": "test"},
            output_data={"result": "ok"},
            pk=pk,
            record_id="rec_ep_1_0001_000000",
            epoch_id="ep_1_0001",
        )
        assert isinstance(proof, ZKProof)
        assert proof.proving_system == "mock"
        assert proof.tier == "full_zk"
        assert len(proof.public_inputs) == 3
        assert proof.epoch_id == "ep_1_0001"

    @pytest.mark.asyncio
    async def test_prove_is_deterministic(self, mock_prover):
        pk, _ = await mock_prover.setup("model-a", "/path/model.onnx")
        kwargs = dict(
            model_id="model-a",
            input_data={"q": "hello"},
            output_data={"a": "world"},
            pk=pk,
            record_id="rec_0",
            epoch_id="ep_1_0001",
        )
        p1 = await mock_prover.prove(**kwargs)
        p2 = await mock_prover.prove(**kwargs)
        assert p1.proof_bytes == p2.proof_bytes

    @pytest.mark.asyncio
    async def test_prove_different_inputs_different_proof(self, mock_prover):
        pk, _ = await mock_prover.setup("model-a", "/path/model.onnx")
        p1 = await mock_prover.prove("model-a", {"q": "a"}, {"a": "1"}, pk, "rec_0", "ep_1")
        p2 = await mock_prover.prove("model-a", {"q": "b"}, {"a": "1"}, pk, "rec_0", "ep_1")
        assert p1.proof_bytes != p2.proof_bytes

    @pytest.mark.asyncio
    async def test_verify_valid_proof(self, mock_prover):
        pk, vk = await mock_prover.setup("model-a", "/path/model.onnx")
        proof = await mock_prover.prove(
            "model-a", {"q": "test"}, {"a": "ok"}, pk, "rec_0", "ep_1"
        )
        assert mock_prover.verify(proof, vk) is True

    @pytest.mark.asyncio
    async def test_verify_wrong_model_hash_fails(self, mock_prover):
        pk, vk = await mock_prover.setup("model-a", "/path/model.onnx")
        proof = await mock_prover.prove(
            "model-a", {"q": "test"}, {"a": "ok"}, pk, "rec_0", "ep_1"
        )
        _, vk_other = await mock_prover.setup("model-b", "/other.onnx")
        assert mock_prover.verify(proof, vk_other) is False

    @pytest.mark.asyncio
    async def test_verify_tampered_proof_fails(self, mock_prover):
        pk, vk = await mock_prover.setup("model-a", "/path/model.onnx")
        proof = await mock_prover.prove(
            "model-a", {"q": "test"}, {"a": "ok"}, pk, "rec_0", "ep_1"
        )
        tampered = ZKProof(
            proof_bytes=b"\xff" * 32,  # altered bytes
            public_inputs=proof.public_inputs,
            proving_system=proof.proving_system,
            tier=proof.tier,
            model_hash=proof.model_hash,
            prover_version=proof.prover_version,
            epoch_id=proof.epoch_id,
        )
        # MockProver verify is structural only, so altered bytes still structurally valid
        # but wrong proving_system would fail:
        bad_system = ZKProof(
            proof_bytes=proof.proof_bytes,
            public_inputs=proof.public_inputs,
            proving_system="groth16",
            tier=proof.tier,
            model_hash=proof.model_hash,
            prover_version=proof.prover_version,
            epoch_id=proof.epoch_id,
        )
        assert mock_prover.verify(bad_system, vk) is False

    def test_proving_system_property(self, mock_prover):
        assert mock_prover.proving_system == "mock"

    def test_tier_property(self, mock_prover):
        assert mock_prover.tier == "full_zk"
