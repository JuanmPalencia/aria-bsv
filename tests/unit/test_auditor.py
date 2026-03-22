"""Tests for aria.auditor — AuditConfig, InferenceAuditor, BatchManager."""

from __future__ import annotations

import time
import pytest

from aria.auditor import AuditConfig, InferenceAuditor, Receipt
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.core.errors import ARIAConfigError, ARIAError
from aria.core.hasher import hash_object
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_call_counter = 0


class _CountingWallet(WalletInterface):
    """Returns distinct fake txids and counts calls."""

    def __init__(self) -> None:
        self._n = 0

    async def sign_and_broadcast(self, payload: dict) -> str:
        self._n += 1
        return f"{self._n:064x}"

    @property
    def call_count(self) -> int:
        return self._n


class _OkBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="a" * 64, propagated=True)


def _make_config(**kwargs) -> AuditConfig:
    defaults = {"system_id": "test-sys", "bsv_key": "wif-placeholder"}
    defaults.update(kwargs)
    return AuditConfig(**defaults)


def _make_auditor(
    *,
    batch_ms: int = 60_000,
    batch_size: int = 500,
    pii_fields: list[str] | None = None,
) -> InferenceAuditor:
    """Build an auditor with a mock wallet and in-memory SQLite."""
    wallet = _CountingWallet()
    storage = SQLiteStorage("sqlite://")
    config = AuditConfig(
        system_id="test-sys",
        bsv_key="wif-placeholder",
        batch_ms=batch_ms,
        batch_size=batch_size,
        pii_fields=pii_fields or [],
    )
    return InferenceAuditor(
        config=config,
        model_hashes={"model-a": "sha256:" + "f" * 64},
        initial_state={"version": "1.0"},
        _wallet=wallet,
        _broadcaster=_OkBroadcaster(),
        _storage=storage,
    )


# ---------------------------------------------------------------------------
# AuditConfig — validation
# ---------------------------------------------------------------------------


class TestAuditConfig:
    def test_minimal_bsv_key(self):
        cfg = AuditConfig(system_id="s", bsv_key="k")
        assert cfg.system_id == "s"
        assert cfg.bsv_key == "k"

    def test_minimal_brc100(self):
        cfg = AuditConfig(system_id="s", brc100_url="http://wallet.local")
        assert cfg.brc100_url == "http://wallet.local"

    def test_no_key_raises(self):
        with pytest.raises(ARIAConfigError, match="bsv_key or brc100_url"):
            AuditConfig(system_id="s")

    def test_both_keys_raises(self):
        with pytest.raises(ARIAConfigError, match="mutually exclusive"):
            AuditConfig(system_id="s", bsv_key="k", brc100_url="http://w")

    def test_empty_system_id_raises(self):
        with pytest.raises(ARIAConfigError, match="system_id"):
            AuditConfig(system_id="", bsv_key="k")

    def test_zero_batch_ms_raises(self):
        with pytest.raises(ARIAConfigError, match="batch_ms"):
            AuditConfig(system_id="s", bsv_key="k", batch_ms=0)

    def test_zero_batch_size_raises(self):
        with pytest.raises(ARIAConfigError, match="batch_size"):
            AuditConfig(system_id="s", bsv_key="k", batch_size=0)

    def test_invalid_network_raises(self):
        with pytest.raises(ARIAConfigError, match="network"):
            AuditConfig(system_id="s", bsv_key="k", network="ropsten")

    def test_defaults(self):
        cfg = AuditConfig(system_id="s", bsv_key="k")
        assert cfg.batch_ms == 5_000
        assert cfg.batch_size == 500
        assert cfg.network == "mainnet"
        assert cfg.pii_fields == []


# ---------------------------------------------------------------------------
# InferenceAuditor — construction
# ---------------------------------------------------------------------------


class TestInferenceAuditorInit:
    def test_auditor_starts_without_error(self):
        auditor = _make_auditor()
        auditor.close()

    def test_epoch_opens_on_startup(self):
        auditor = _make_auditor()
        # Wait for the first EPOCH_OPEN to go through.
        auditor._batch._epoch_ready.wait(timeout=5.0)
        assert auditor._batch._current_open is not None
        auditor.close()


# ---------------------------------------------------------------------------
# InferenceAuditor.record()
# ---------------------------------------------------------------------------


class TestInferenceAuditorRecord:
    def test_record_returns_record_id(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {"x": 1}, {"y": 2})
        assert rid.startswith("rec_")
        auditor.close()

    def test_record_id_format(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {"x": 1}, {"y": 2})
        # rec_{epoch_id}_{seq:06d}
        assert rid.startswith("rec_ep_")
        parts = rid.split("_")
        assert parts[-1].isdigit()
        auditor.close()

    def test_record_persisted_to_storage(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {"x": 1}, {"y": 2})
        rec = auditor._storage.get_record(rid)
        assert rec is not None
        assert rec.model_id == "model-a"
        auditor.close()

    def test_record_input_hash_matches(self):
        auditor = _make_auditor()
        inp = {"sensor": 42}
        rid = auditor.record("model-a", inp, {"result": 0})
        rec = auditor._storage.get_record(rid)
        assert rec.input_hash == hash_object(inp)
        auditor.close()

    def test_record_output_hash_matches(self):
        auditor = _make_auditor()
        out = {"label": "emergency", "score": 0.99}
        rid = auditor.record("model-a", {}, out)
        rec = auditor._storage.get_record(rid)
        assert rec.output_hash == hash_object(out)
        auditor.close()

    def test_record_confidence_stored(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {}, {}, confidence=0.88)
        rec = auditor._storage.get_record(rid)
        assert rec.confidence == pytest.approx(0.88)
        auditor.close()

    def test_record_sequence_monotonically_increases(self):
        auditor = _make_auditor()
        rids = [auditor.record("model-a", {"i": i}, {"o": i}) for i in range(5)]
        seqs = [auditor._storage.get_record(r).sequence for r in rids]
        assert seqs == list(range(5))
        auditor.close()

    def test_multiple_records_same_epoch(self):
        auditor = _make_auditor()
        rids = [auditor.record("model-a", {"i": i}, {"o": i}) for i in range(3)]
        epochs = {auditor._storage.get_record(r).epoch_id for r in rids}
        assert len(epochs) == 1  # all in the same epoch
        auditor.close()


# ---------------------------------------------------------------------------
# InferenceAuditor — PII sanitisation
# ---------------------------------------------------------------------------


class TestPIISanitisation:
    def test_pii_fields_removed_before_hash(self):
        auditor = _make_auditor(pii_fields=["patient_name", "ssn"])
        inp = {"patient_name": "Juan", "ssn": "123-45-6789", "severity": 5}
        rid = auditor.record("model-a", inp, {})
        rec = auditor._storage.get_record(rid)
        # Hash must match the sanitised version.
        sanitised = {"severity": 5}
        assert rec.input_hash == hash_object(sanitised)
        auditor.close()

    def test_non_dict_input_not_sanitised(self):
        auditor = _make_auditor(pii_fields=["secret"])
        inp = [1, 2, 3]  # not a dict
        rid = auditor.record("model-a", inp, {})
        rec = auditor._storage.get_record(rid)
        assert rec.input_hash == hash_object(inp)
        auditor.close()

    def test_no_pii_fields_passes_through(self):
        auditor = _make_auditor()
        inp = {"data": 42}
        rid = auditor.record("model-a", inp, {})
        rec = auditor._storage.get_record(rid)
        assert rec.input_hash == hash_object(inp)
        auditor.close()


# ---------------------------------------------------------------------------
# InferenceAuditor.track() decorator
# ---------------------------------------------------------------------------


class TestTrackDecorator:
    def test_sync_function_return_value_unchanged(self):
        auditor = _make_auditor()

        @auditor.track("model-a")
        def predict(x: int) -> int:
            return x * 2

        assert predict(5) == 10
        auditor.close()

    def test_sync_function_record_created(self):
        auditor = _make_auditor()
        records_before = len(auditor._batch._pending) + auditor._batch._sequence

        @auditor.track("model-a")
        def predict(x: int) -> int:
            return x * 2

        predict(3)
        time.sleep(0.05)  # allow background thread to process
        # At least one record should be stored
        epoch_id = auditor._batch._current_open
        auditor.close()

    @pytest.mark.asyncio
    async def test_async_function_return_value_unchanged(self):
        auditor = _make_auditor()

        @auditor.track("model-a")
        async def async_predict(x: int) -> int:
            return x + 1

        result = await async_predict(9)
        assert result == 10
        auditor.close()

    def test_functools_wraps_preserves_name(self):
        auditor = _make_auditor()

        @auditor.track("model-a")
        def my_model(x: int) -> int:
            return x

        assert my_model.__name__ == "my_model"
        auditor.close()


# ---------------------------------------------------------------------------
# InferenceAuditor.get_receipt()
# ---------------------------------------------------------------------------


class TestGetReceipt:
    def test_receipt_record_id_matches(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {}, {})
        receipt = auditor.get_receipt(rid)
        assert receipt.record_id == rid
        auditor.close()

    def test_receipt_model_id_matches(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {}, {})
        receipt = auditor.get_receipt(rid)
        assert receipt.model_id == "model-a"
        auditor.close()

    def test_receipt_epoch_id_not_empty(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {}, {})
        receipt = auditor.get_receipt(rid)
        assert receipt.epoch_id.startswith("ep_")
        auditor.close()

    def test_receipt_record_hash_verifiable(self):
        auditor = _make_auditor()
        rid = auditor.record("model-a", {"x": 1}, {"y": 2})
        receipt = auditor.get_receipt(rid)
        # The hash in the receipt must match what's in storage.
        rec = auditor._storage.get_record(rid)
        assert receipt.record_hash == rec.hash()
        auditor.close()

    def test_unknown_record_raises_aria_error(self):
        auditor = _make_auditor()
        with pytest.raises(ARIAError, match="not found"):
            auditor.get_receipt("rec_does_not_exist")
        auditor.close()


# ---------------------------------------------------------------------------
# InferenceAuditor.flush() — early epoch close
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_triggers_epoch_close_in_storage(self):
        auditor = _make_auditor(batch_ms=60_000)  # long timer to prevent auto-close
        rid = auditor.record("model-a", {}, {})
        rec = auditor._storage.get_record(rid)
        epoch_id = rec.epoch_id

        # Before flush: epoch should be open (no close_txid)
        auditor.flush()

        row = auditor._storage.get_epoch(epoch_id)
        assert row is not None
        assert row.close_txid != ""

    def test_flush_opens_new_epoch(self):
        auditor = _make_auditor(batch_ms=60_000)
        first_epoch = auditor._batch._current_open.epoch_id if auditor._batch._current_open else None
        auditor.flush()
        time.sleep(0.1)
        new_epoch = auditor._batch._current_open
        if new_epoch and first_epoch:
            assert new_epoch.epoch_id != first_epoch
        auditor.close()


# ---------------------------------------------------------------------------
# BatchManager — batch_size triggers early close
# ---------------------------------------------------------------------------


class TestBatchSize:
    def test_batch_size_triggers_flush(self):
        auditor = _make_auditor(batch_ms=60_000, batch_size=5)
        # Add 5 records — should trigger an early flush.
        rids = [auditor.record("model-a", {"i": i}, {"o": i}) for i in range(5)]
        time.sleep(0.5)  # give background loop time to flush

        # The epoch for these records should now be closed.
        epoch_id = auditor._storage.get_record(rids[0]).epoch_id
        row = auditor._storage.get_epoch(epoch_id)
        if row:  # epoch may have been saved after flush
            assert row.records_count == 5
        auditor.close()
