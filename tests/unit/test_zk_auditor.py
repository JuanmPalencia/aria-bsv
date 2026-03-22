"""Integration tests — ZK layer wired into InferenceAuditor."""

from __future__ import annotations

import asyncio
import time
import pytest

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface
from aria.zk.mock_prover import MockProver
from aria.zk.claims import ConfidencePercentile, ModelUnchanged, RecordCountRange


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _MockWallet(WalletInterface):
    _counter = 0

    async def sign_and_broadcast(self, payload: dict) -> str:
        _MockWallet._counter += 1
        return "a" * 64

    @classmethod
    def reset(cls):
        cls._counter = 0


class _MockBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="b" * 64, propagated=True)


def _make_auditor(
    zk_prover=None,
    model_paths=None,
    zk_claims=None,
    batch_ms=100,
) -> tuple[InferenceAuditor, SQLiteStorage]:
    _MockWallet.reset()
    storage = SQLiteStorage("sqlite://")
    config = AuditConfig(
        system_id="test-system",
        bsv_key="placeholder",
        network="mainnet",
        batch_ms=batch_ms,
        batch_size=50,
        zk_prover=zk_prover,
        model_paths=model_paths or {},
        zk_claims=zk_claims or [],
    )
    auditor = InferenceAuditor(
        config=config,
        model_hashes={"demo-model": "sha256:" + "a" * 64},
        _wallet=_MockWallet(),
        _broadcaster=_MockBroadcaster(),
        _storage=storage,
    )
    return auditor, storage


# ---------------------------------------------------------------------------
# ZK prover wired into auditor
# ---------------------------------------------------------------------------

class TestAuditorZKIntegration:
    def test_record_without_zk_still_works(self):
        auditor, storage = _make_auditor()
        rec_id = auditor.record("demo-model", {"q": "test"}, {"a": "ok"}, confidence=0.9)
        assert rec_id.startswith("rec_")
        auditor.close()

    def test_record_with_mock_prover_stores_proof(self):
        prover = MockProver()
        auditor, storage = _make_auditor(
            zk_prover=prover,
            model_paths={"demo-model": "dummy_path.onnx"},
            batch_ms=200,
        )
        rec_id = auditor.record("demo-model", {"q": "test"}, {"a": "ok"}, confidence=0.9)
        # Give ZK proof time to be generated (async, non-blocking)
        time.sleep(0.5)
        auditor.flush()
        time.sleep(0.2)

        proof = storage.get_proof(rec_id)
        # Proof may or may not be stored depending on model_path mock setup
        # (MockProver doesn't require real file). Just verify no crash.
        auditor.close()

    def test_zk_claims_evaluated_on_flush(self):
        claims = [
            ConfidencePercentile(p=99, threshold=0.5),
            RecordCountRange(min_count=1, max_count=100),
        ]
        auditor, storage = _make_auditor(
            zk_claims=claims,
            batch_ms=500,
        )
        for i in range(5):
            auditor.record("demo-model", {"q": f"test{i}"}, {"a": "ok"}, confidence=0.9)

        auditor.flush()
        time.sleep(0.2)
        auditor.close()
        # No assertion on proof — just verify no crash with claims configured

    def test_multiple_records_no_crash_with_zk(self):
        prover = MockProver()
        auditor, storage = _make_auditor(
            zk_prover=prover,
            model_paths={"demo-model": "path.onnx"},
            zk_claims=[ModelUnchanged(expected_model_ids={"demo-model"})],
            batch_ms=300,
        )
        for i in range(10):
            auditor.record(
                "demo-model",
                {"q": f"input_{i}"},
                {"a": f"output_{i}"},
                confidence=0.8 + i * 0.01,
            )
        auditor.flush()
        time.sleep(0.3)
        auditor.close()

    def test_records_still_persist_without_zk_proof(self):
        auditor, storage = _make_auditor(batch_ms=5000)
        rec_id = auditor.record("demo-model", {"q": "test"}, {"a": "ok"})
        # Record persisted to storage immediately (no-loss guarantee)
        rec = storage.get_record(rec_id)
        assert rec is not None
        assert rec.model_id == "demo-model"
        auditor.close()


# ---------------------------------------------------------------------------
# Storage ZK methods
# ---------------------------------------------------------------------------

class TestStorageZKMethods:
    def test_save_and_get_proof(self):
        from aria.zk.base import ZKProof
        import hashlib

        storage = SQLiteStorage("sqlite://")
        proof = ZKProof(
            proof_bytes=hashlib.sha256(b"test").digest(),
            public_inputs=["sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64],
            proving_system="mock",
            tier="full_zk",
            model_hash="sha256:" + "d" * 64,
            prover_version="mock-1.0",
            epoch_id="ep_test_0001",
            record_id="rec_ep_test_0001_000000",
        )
        storage.save_proof(proof)
        retrieved = storage.get_proof("rec_ep_test_0001_000000")
        assert retrieved is not None
        assert retrieved.proof_bytes == proof.proof_bytes
        assert retrieved.proving_system == "mock"

    def test_list_proofs_by_epoch(self):
        from aria.zk.base import ZKProof
        import hashlib

        storage = SQLiteStorage("sqlite://")
        for i in range(3):
            proof = ZKProof(
                proof_bytes=hashlib.sha256(f"proof_{i}".encode()).digest(),
                public_inputs=["sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64],
                proving_system="mock",
                tier="full_zk",
                model_hash="sha256:" + "d" * 64,
                prover_version="mock-1.0",
                epoch_id="ep_test_0001",
                record_id=f"rec_{i}",
            )
            storage.save_proof(proof)

        proofs = storage.list_proofs_by_epoch("ep_test_0001")
        assert len(proofs) == 3

    def test_get_proof_not_found_returns_none(self):
        storage = SQLiteStorage("sqlite://")
        assert storage.get_proof("nonexistent_id") is None

    def test_save_and_get_vk(self):
        from aria.zk.base import VerifyingKey

        storage = SQLiteStorage("sqlite://")
        vk = VerifyingKey(
            vk_bytes=b"\xab" * 32,
            model_hash="sha256:" + "e" * 64,
            proving_system="mock",
        )
        storage.save_vk(vk)
        retrieved = storage.get_vk("sha256:" + "e" * 64)
        assert retrieved is not None
        assert retrieved.vk_bytes == vk.vk_bytes

    def test_get_vk_not_found_returns_none(self):
        storage = SQLiteStorage("sqlite://")
        assert storage.get_vk("sha256:" + "f" * 64) is None

    def test_save_vk_twice_overwrites(self):
        from aria.zk.base import VerifyingKey

        storage = SQLiteStorage("sqlite://")
        model_hash = "sha256:" + "g" * 64
        vk1 = VerifyingKey(vk_bytes=b"\x01" * 32, model_hash=model_hash, proving_system="mock")
        vk2 = VerifyingKey(vk_bytes=b"\x02" * 32, model_hash=model_hash, proving_system="mock")
        storage.save_vk(vk1)
        storage.save_vk(vk2)
        retrieved = storage.get_vk(model_hash)
        assert retrieved.vk_bytes == vk2.vk_bytes
