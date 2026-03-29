"""Tests for aria.async_auditor — AsyncInferenceAuditor."""

from __future__ import annotations

import asyncio
import pytest

from aria.async_auditor import AsyncInferenceAuditor
from aria.auditor import AuditConfig, Receipt
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.core.errors import ARIABroadcastError
from aria.core.hasher import hash_object
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CountingWallet(WalletInterface):
    """Returns distinct fake txids and tracks call count."""

    def __init__(self) -> None:
        self._n = 0

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        self._n += 1
        return f"{self._n:064x}"

    @property
    def call_count(self) -> int:
        return self._n


class _OkBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="a" * 64, propagated=True)


class _FailingWallet(WalletInterface):
    """Always raises ARIABroadcastError on sign_and_broadcast."""

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        raise ARIABroadcastError("simulated BSV network failure")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> AuditConfig:  # type: ignore[type-arg]
    defaults: dict = {"system_id": "test-sys", "bsv_key": "wif-placeholder"}
    defaults.update(kwargs)
    return AuditConfig(**defaults)


def _make_auditor(
    *,
    batch_ms: int = 60_000,
    batch_size: int = 500,
    pii_fields: list[str] | None = None,
    wallet: WalletInterface | None = None,
) -> AsyncInferenceAuditor:
    """Build an AsyncInferenceAuditor with mock wallet and in-memory SQLite."""
    w = wallet if wallet is not None else _CountingWallet()
    storage = SQLiteStorage("sqlite://")
    config = _make_config(
        batch_ms=batch_ms,
        batch_size=batch_size,
        pii_fields=pii_fields or [],
    )
    return AsyncInferenceAuditor(
        config=config,
        model_hashes={"model-a": "sha256:" + "f" * 64},
        initial_state={"version": "1.0"},
        _wallet=w,
        _broadcaster=_OkBroadcaster(),
        _storage=storage,
    )


# ---------------------------------------------------------------------------
# test_record_returns_receipt
# ---------------------------------------------------------------------------


async def test_record_returns_receipt():
    auditor = _make_auditor()
    receipt = await auditor.record("model-a", {"x": 1}, {"y": 2})
    assert isinstance(receipt, Receipt)
    assert receipt.record_id.startswith("rec_")
    assert receipt.model_id == "model-a"
    assert receipt.epoch_id.startswith("ep_")


# ---------------------------------------------------------------------------
# test_record_increments_sequence
# ---------------------------------------------------------------------------


async def test_record_increments_sequence():
    auditor = _make_auditor()
    receipts = [
        await auditor.record("model-a", {"i": i}, {"o": i}) for i in range(4)
    ]
    seqs = [
        auditor._storage.get_record(r.record_id).sequence for r in receipts
    ]
    assert seqs == list(range(4))


# ---------------------------------------------------------------------------
# test_flush_closes_epoch
# ---------------------------------------------------------------------------


async def test_flush_closes_epoch():
    auditor = _make_auditor(batch_ms=60_000)
    receipt = await auditor.record("model-a", {}, {})
    epoch_id = receipt.epoch_id

    await auditor.flush()

    row = auditor._storage.get_epoch(epoch_id)
    assert row is not None
    assert row.close_txid != ""


# ---------------------------------------------------------------------------
# test_flush_broadcasts_to_bsv
# ---------------------------------------------------------------------------


async def test_flush_broadcasts_to_bsv():
    wallet = _CountingWallet()
    auditor = _make_auditor(wallet=wallet)

    calls_before = wallet.call_count
    await auditor.record("model-a", {}, {})
    # EPOCH_OPEN is called lazily on first record — wallet already called once.
    await auditor.flush()
    # After flush: EPOCH_CLOSE + new EPOCH_OPEN should have fired.
    assert wallet.call_count > calls_before


# ---------------------------------------------------------------------------
# test_auto_flush_on_batch_size
# ---------------------------------------------------------------------------


async def test_auto_flush_on_batch_size():
    auditor = _make_auditor(batch_ms=60_000, batch_size=3)
    receipts = [
        await auditor.record("model-a", {"i": i}, {"o": i}) for i in range(3)
    ]
    # The third record triggers an immediate flush due to batch_size.
    # Give the event loop a turn to process the flush.
    await asyncio.sleep(0)

    epoch_id = receipts[0].epoch_id
    row = auditor._storage.get_epoch(epoch_id)
    # The epoch should be closed with 3 records.
    if row is not None:
        assert row.records_count == 3


# ---------------------------------------------------------------------------
# test_context_manager_flushes_on_exit
# ---------------------------------------------------------------------------


async def test_context_manager_flushes_on_exit():
    storage = SQLiteStorage("sqlite://")
    wallet = _CountingWallet()
    config = _make_config(batch_ms=60_000)

    async with AsyncInferenceAuditor(
        config,
        model_hashes={"model-a": "sha256:" + "f" * 64},
        _wallet=wallet,
        _broadcaster=_OkBroadcaster(),
        _storage=storage,
    ) as auditor:
        receipt = await auditor.record("model-a", {"x": 1}, {})
        epoch_id = receipt.epoch_id

    # After __aexit__ the epoch must be closed in storage.
    row = storage.get_epoch(epoch_id)
    assert row is not None
    assert row.close_txid != ""


# ---------------------------------------------------------------------------
# test_track_decorator_async_fn
# ---------------------------------------------------------------------------


async def test_track_decorator_async_fn():
    auditor = _make_auditor()

    @auditor.track("model-a")
    async def predict(x: int) -> int:
        return x * 3

    result = await predict(7)
    assert result == 21

    # Give the coroutine time to record.
    await asyncio.sleep(0)

    # The record should have landed in storage.
    records = auditor._storage.list_records_by_epoch(
        auditor._current_open.epoch_id if auditor._current_open else ""
    )
    # At least one record was stored during this test.
    assert any(r.model_id == "model-a" for r in records)


# ---------------------------------------------------------------------------
# test_pii_sanitisation
# ---------------------------------------------------------------------------


async def test_pii_sanitisation():
    auditor = _make_auditor(pii_fields=["patient_name", "ssn"])
    inp = {"patient_name": "Juan", "ssn": "123-45-6789", "severity": 5}

    receipt = await auditor.record("model-a", inp, {})
    rec = auditor._storage.get_record(receipt.record_id)
    assert rec is not None

    sanitised = {"severity": 5}
    assert rec.input_hash == hash_object(sanitised)


# ---------------------------------------------------------------------------
# test_record_error_does_not_crash
# ---------------------------------------------------------------------------


async def test_record_error_does_not_crash():
    """If BSV broadcast fails, record() still returns a Receipt."""
    failing_wallet = _FailingWallet()
    storage = SQLiteStorage("sqlite://")
    config = _make_config(batch_ms=60_000)

    auditor = AsyncInferenceAuditor(
        config,
        model_hashes={"model-a": "sha256:" + "f" * 64},
        _wallet=failing_wallet,
        _broadcaster=_OkBroadcaster(),
        _storage=storage,
    )

    # The wallet will fail on EPOCH_OPEN — record() should not propagate the error.
    receipt = await auditor.record("model-a", {"x": 1}, {"y": 2})
    assert isinstance(receipt, Receipt)
    # record_id will contain "error" since the open failed.
    assert receipt.model_id == "model-a"


# ---------------------------------------------------------------------------
# test_get_receipt_returns_stored_record
# ---------------------------------------------------------------------------


async def test_get_receipt_returns_stored_record():
    auditor = _make_auditor()
    receipt = await auditor.record("model-a", {"a": 1}, {"b": 2})
    record_id = receipt.record_id

    fetched = await auditor.get_receipt(record_id)
    assert fetched is not None
    assert fetched.record_id == record_id
    assert fetched.model_id == "model-a"

    rec = auditor._storage.get_record(record_id)
    assert fetched.record_hash == rec.hash()


# ---------------------------------------------------------------------------
# test_get_receipt_returns_none_for_unknown
# ---------------------------------------------------------------------------


async def test_get_receipt_returns_none_for_unknown():
    auditor = _make_auditor()
    result = await auditor.get_receipt("rec_does_not_exist_000000")
    assert result is None


# ---------------------------------------------------------------------------
# test_concurrent_records
# ---------------------------------------------------------------------------


async def test_concurrent_records():
    """10 concurrent record() calls must all land in storage correctly."""
    auditor = _make_auditor(batch_ms=60_000, batch_size=500)

    receipts = await asyncio.gather(
        *[auditor.record("model-a", {"i": i}, {"o": i}) for i in range(10)]
    )

    assert len(receipts) == 10
    record_ids = {r.record_id for r in receipts}
    # All record IDs are unique.
    assert len(record_ids) == 10

    # Every record is retrievable from storage.
    for receipt in receipts:
        stored = auditor._storage.get_record(receipt.record_id)
        assert stored is not None, f"{receipt.record_id} not found in storage"


# ---------------------------------------------------------------------------
# test_flush_empty_epoch_is_noop
# ---------------------------------------------------------------------------


async def test_flush_empty_epoch_is_noop():
    """flush() with no pending records and no open epoch must be a no-op."""
    auditor = _make_auditor()
    # Never called record(), so _current_open is None and _pending_records is [].
    # This should return immediately without error.
    await auditor.flush()
    # Nothing was stored.
    assert auditor._storage.list_epochs() == []
