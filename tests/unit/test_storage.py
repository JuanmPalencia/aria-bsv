"""Tests for aria.storage — StorageInterface contract and SQLiteStorage."""

from __future__ import annotations

import pytest

from aria.storage.base import StorageInterface, EpochRow
from aria.storage.sqlite import SQLiteStorage
from aria.core.record import AuditRecord
from aria.core.hasher import hash_object
from aria.core.errors import ARIAStorageError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_HASH = "sha256:" + "a" * 64
_OPEN_TXID = "b" * 64
_CLOSE_TXID = "c" * 64


def _make_record(epoch_id: str, seq: int, model_id: str = "model-x") -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id=model_id,
        input_hash=hash_object({"seq": seq}),
        output_hash=hash_object({"result": seq}),
        sequence=seq,
        confidence=0.9,
        latency_ms=12,
        metadata={"k": "v"},
    )


def _make_store() -> SQLiteStorage:
    return SQLiteStorage(dsn="sqlite://")  # in-memory


# ---------------------------------------------------------------------------
# StorageInterface — ABC contract
# ---------------------------------------------------------------------------


class TestStorageInterfaceABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            StorageInterface()  # type: ignore[abstract]

    def test_concrete_subclass_missing_method_raises(self):
        class Incomplete(StorageInterface):
            def save_record(self, record): ...
            def save_epoch_open(self, *a, **kw): ...
            def save_epoch_close(self, *a, **kw): ...
            def get_record(self, record_id): ...
            def get_epoch(self, epoch_id): ...
            # missing list_records_by_epoch

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_full_concrete_subclass_instantiates(self):
        class Full(StorageInterface):
            def save_record(self, record): ...
            def save_epoch_open(self, *a, **kw): ...
            def save_epoch_close(self, *a, **kw): ...
            def get_record(self, record_id): return None
            def get_epoch(self, epoch_id): return None
            def list_records_by_epoch(self, epoch_id): return []

        s = Full()
        assert isinstance(s, StorageInterface)


# ---------------------------------------------------------------------------
# EpochRow dataclass
# ---------------------------------------------------------------------------


class TestEpochRow:
    def test_fields(self):
        row = EpochRow(
            epoch_id="ep_1",
            system_id="sys-1",
            open_txid=_OPEN_TXID,
            close_txid="",
            state_hash=_FAKE_HASH,
            model_hashes={"m": _FAKE_HASH},
            opened_at=1000,
            closed_at=0,
            records_count=0,
            merkle_root="",
        )
        assert row.epoch_id == "ep_1"
        assert row.model_hashes == {"m": _FAKE_HASH}


# ---------------------------------------------------------------------------
# SQLiteStorage — save and retrieve records
# ---------------------------------------------------------------------------


class TestSQLiteStorageRecords:
    def test_save_and_get_record(self):
        store = _make_store()
        rec = _make_record("ep_001", seq=0)
        store.save_record(rec)
        fetched = store.get_record(rec.record_id)
        assert fetched is not None
        assert fetched.record_id == rec.record_id

    def test_get_nonexistent_record_returns_none(self):
        store = _make_store()
        assert store.get_record("rec_does_not_exist") is None

    def test_record_fields_round_trip(self):
        store = _make_store()
        rec = _make_record("ep_001", seq=3)
        store.save_record(rec)
        fetched = store.get_record(rec.record_id)
        assert fetched.epoch_id == rec.epoch_id
        assert fetched.model_id == rec.model_id
        assert fetched.input_hash == rec.input_hash
        assert fetched.output_hash == rec.output_hash
        assert fetched.sequence == rec.sequence
        assert fetched.confidence == pytest.approx(rec.confidence)
        assert fetched.latency_ms == rec.latency_ms
        assert fetched.metadata == rec.metadata

    def test_record_hash_preserved(self):
        store = _make_store()
        rec = _make_record("ep_001", seq=0)
        expected_hash = rec.hash()
        store.save_record(rec)
        fetched = store.get_record(rec.record_id)
        assert fetched.hash() == expected_hash

    def test_record_none_confidence_preserved(self):
        store = _make_store()
        rec = AuditRecord(
            epoch_id="ep_001",
            model_id="m",
            input_hash=_FAKE_HASH,
            output_hash=_FAKE_HASH,
            sequence=0,
            confidence=None,
        )
        store.save_record(rec)
        fetched = store.get_record(rec.record_id)
        assert fetched.confidence is None

    def test_list_records_by_epoch_ordered_by_sequence(self):
        store = _make_store()
        records = [_make_record("ep_001", seq=i) for i in range(5)]
        for rec in reversed(records):  # insert in reverse order
            store.save_record(rec)
        fetched = store.list_records_by_epoch("ep_001")
        seqs = [r.sequence for r in fetched]
        assert seqs == list(range(5))

    def test_list_records_empty_epoch_returns_empty(self):
        store = _make_store()
        result = store.list_records_by_epoch("ep_unknown")
        assert result == []

    def test_list_records_only_returns_matching_epoch(self):
        store = _make_store()
        store.save_record(_make_record("ep_A", seq=0))
        store.save_record(_make_record("ep_B", seq=0))
        assert len(store.list_records_by_epoch("ep_A")) == 1
        assert len(store.list_records_by_epoch("ep_B")) == 1


# ---------------------------------------------------------------------------
# SQLiteStorage — save and retrieve epochs
# ---------------------------------------------------------------------------


class TestSQLiteStorageEpochs:
    def test_save_epoch_open(self):
        store = _make_store()
        store.save_epoch_open(
            epoch_id="ep_001",
            system_id="sys-1",
            open_txid=_OPEN_TXID,
            model_hashes={"m": _FAKE_HASH},
            state_hash=_FAKE_HASH,
            opened_at=1000,
        )
        row = store.get_epoch("ep_001")
        assert row is not None
        assert row.open_txid == _OPEN_TXID
        assert row.close_txid == ""
        assert row.model_hashes == {"m": _FAKE_HASH}
        assert row.opened_at == 1000
        assert row.records_count == 0

    def test_save_epoch_close_updates_row(self):
        store = _make_store()
        store.save_epoch_open("ep_001", "sys-1", _OPEN_TXID, {}, _FAKE_HASH, 1000)
        store.save_epoch_close("ep_001", _CLOSE_TXID, _FAKE_HASH, 5, 2000)
        row = store.get_epoch("ep_001")
        assert row.close_txid == _CLOSE_TXID
        assert row.records_count == 5
        assert row.closed_at == 2000
        assert row.merkle_root == _FAKE_HASH

    def test_get_nonexistent_epoch_returns_none(self):
        store = _make_store()
        assert store.get_epoch("ep_unknown") is None

    def test_close_nonexistent_epoch_raises_storage_error(self):
        store = _make_store()
        with pytest.raises(ARIAStorageError, match="not found"):
            store.save_epoch_close("ep_missing", _CLOSE_TXID, _FAKE_HASH, 0, 1)

    def test_epoch_model_hashes_round_trip(self):
        store = _make_store()
        mh = {"model-a": "sha256:" + "a" * 64, "model-b": "sha256:" + "b" * 64}
        store.save_epoch_open("ep_001", "sys", _OPEN_TXID, mh, _FAKE_HASH, 1000)
        row = store.get_epoch("ep_001")
        assert row.model_hashes == mh

    def test_epoch_state_hash_preserved(self):
        store = _make_store()
        state_hash = hash_object({"fleet": 10, "active": True})
        store.save_epoch_open("ep_001", "sys", _OPEN_TXID, {}, state_hash, 1000)
        row = store.get_epoch("ep_001")
        assert row.state_hash == state_hash


# ---------------------------------------------------------------------------
# SQLiteStorage — thread safety (basic check)
# ---------------------------------------------------------------------------


class TestSQLiteStorageThreadSafety:
    def test_concurrent_saves_do_not_crash(self):
        import threading

        store = _make_store()
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                rec = _make_record(f"ep_{i:03d}", seq=i)
                store.save_record(rec)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All records should be findable.
        for i in range(20):
            rec = _make_record(f"ep_{i:03d}", seq=i)
            assert store.get_record(rec.record_id) is not None
