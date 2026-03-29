"""Tests for aria.offline — offline mode and sync."""

from __future__ import annotations

import pytest

from aria.offline import OfflineAuditor, OfflineEpochResult, list_pending
from aria.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    return SQLiteStorage(dsn="sqlite://")


class TestOfflineAuditor:
    """Tests for the OfflineAuditor class."""

    def test_init_creates_epoch(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        assert a.current_epoch_id is not None
        assert a.current_epoch_id.startswith("offline-")

    def test_record_returns_id(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        rid = a.record("gpt-4", {"prompt": "hi"}, {"text": "hello"}, confidence=0.9)
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_record_increments_count(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        a.record("model-a", "in1", "out1")
        a.record("model-a", "in2", "out2")
        assert a.record_count == 2

    def test_close_epoch(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        a.record("model-a", "in1", "out1")
        a.record("model-a", "in2", "out2")
        result = a.close_epoch()
        assert isinstance(result, OfflineEpochResult)
        assert result.records_count == 2
        assert result.merkle_root.startswith("sha256:")
        assert result.synced is False

    def test_close_epoch_opens_new_one(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        old_id = a.current_epoch_id
        a.record("m", "i", "o")
        a.close_epoch()
        assert a.current_epoch_id is not None
        assert a.current_epoch_id != old_id

    def test_multiple_epochs(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        a.record("m", "i", "o")
        r1 = a.close_epoch()
        a.record("m", "i2", "o2")
        a.record("m", "i3", "o3")
        r2 = a.close_epoch()

        assert r1.epoch_id != r2.epoch_id
        assert r1.records_count == 1
        assert r2.records_count == 2

    def test_storage_property(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        assert a.storage is not None

    def test_record_with_metadata(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        rid = a.record("m", "i", "o", metadata={"key": "val"})
        assert isinstance(rid, str)

    def test_custom_model_hashes(self):
        a = OfflineAuditor("test-sys", db="sqlite://", model_hashes={"gpt-4": "abc123"})
        a.record("gpt-4", "i", "o")
        result = a.close_epoch()
        assert result.records_count == 1


class TestListPending:
    """Tests for list_pending."""

    def test_lists_pending_epochs(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        a.record("m", "i", "o")
        a.close_epoch()
        pending = list_pending(a.storage)
        assert len(pending) >= 1

    def test_empty_when_no_epochs(self):
        storage = SQLiteStorage(dsn="sqlite://")
        pending = list_pending(storage)
        assert pending == []

    def test_returns_correct_epoch_ids(self):
        a = OfflineAuditor("test-sys", db="sqlite://")
        a.record("m", "i1", "o1")
        r1 = a.close_epoch()
        a.record("m", "i2", "o2")
        r2 = a.close_epoch()

        pending = list_pending(a.storage)
        ids = set(pending)
        assert r1.epoch_id in ids
        assert r2.epoch_id in ids


class TestOfflineEpochResult:
    """Tests for OfflineEpochResult dataclass."""

    def test_defaults(self):
        r = OfflineEpochResult(epoch_id="ep-1", records_count=10, merkle_root="abc")
        assert r.synced is False
        assert r.open_txid == ""
        assert r.close_txid == ""
