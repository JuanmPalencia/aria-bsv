"""Chaos: thread-safety under concurrent load.

InferenceAuditor.record() is designed to be called from multiple threads
simultaneously.  This test verifies that concurrent calls produce no
data races: all records are saved, all sequence numbers are unique.
"""

from __future__ import annotations

import threading

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_N = 0


class _FakeWallet(WalletInterface):
    async def sign_and_broadcast(self, payload: dict) -> str:
        global _N
        _N += 1
        return f"{_N:064x}"


class _FakeBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="c" * 64, propagated=True)


def _make_auditor(storage: SQLiteStorage) -> InferenceAuditor:
    config = AuditConfig(system_id="concurrent-test", bsv_key="placeholder", batch_ms=60_000)
    return InferenceAuditor(
        config=config,
        model_hashes={"m": "sha256:" + "e" * 64},
        _wallet=_FakeWallet(),
        _broadcaster=_FakeBroadcaster(),
        _storage=storage,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentRecords:
    def test_all_records_saved_under_concurrent_load(self):
        """N_THREADS × RECORDS_PER_THREAD records must all appear in storage."""
        N_THREADS = 20
        RECORDS_PER_THREAD = 5
        TOTAL = N_THREADS * RECORDS_PER_THREAD

        storage = SQLiteStorage("sqlite://")
        auditor = _make_auditor(storage)
        auditor._batch._epoch_ready.wait(timeout=5.0)

        record_ids: list[str] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def worker() -> None:
            for i in range(RECORDS_PER_THREAD):
                try:
                    rid = auditor.record("m", {"thread": i}, {"out": i * 2})
                    with lock:
                        record_ids.append(rid)
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Worker errors: {errors}"
        assert len(record_ids) == TOTAL

        # Verify all records are actually in storage.
        for rid in record_ids:
            assert storage.get_record(rid) is not None, f"{rid} missing from storage"

        auditor.close()

    def test_sequence_numbers_are_unique_within_epoch(self):
        """Concurrent calls must produce unique sequence numbers — no collisions."""
        N_THREADS = 15
        RECORDS_PER_THREAD = 4

        storage = SQLiteStorage("sqlite://")
        auditor = _make_auditor(storage)
        auditor._batch._epoch_ready.wait(timeout=5.0)

        record_ids: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            for i in range(RECORDS_PER_THREAD):
                rid = auditor.record("m", {"x": i}, {"y": i})
                with lock:
                    record_ids.append(rid)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # Retrieve all records and check sequence uniqueness within the epoch.
        epoch_id = storage.get_record(record_ids[0]).epoch_id  # type: ignore[union-attr]
        all_records = storage.list_records_by_epoch(epoch_id)
        sequences = [r.sequence for r in all_records]
        assert len(sequences) == len(set(sequences)), "Duplicate sequence numbers detected"

        auditor.close()

    def test_no_records_lost_during_flush(self):
        """Records recorded just before flush() must not be lost."""
        storage = SQLiteStorage("sqlite://")
        auditor = _make_auditor(storage)
        auditor._batch._epoch_ready.wait(timeout=5.0)

        pre_flush_ids = [auditor.record("m", {"i": i}, {"r": i}) for i in range(8)]
        auditor.flush()
        post_flush_ids = [auditor.record("m", {"j": j}, {"s": j}) for j in range(4)]

        for rid in pre_flush_ids + post_flush_ids:
            assert storage.get_record(rid) is not None

        auditor.close()
