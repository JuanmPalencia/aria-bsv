"""Performance: InferenceAuditor.record() call overhead.

The synchronous portion of record() (acquire lock + create AuditRecord +
write to SQLite + append to queue) must stay under 5ms per call so it
does not materially affect AI inference latency.
"""

from __future__ import annotations

import time

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

_N = 0


class _FakeWallet(WalletInterface):
    async def sign_and_broadcast(self, payload: dict) -> str:
        global _N
        _N += 1
        return f"{_N:064x}"


class _FakeBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="e" * 64, propagated=True)


def _make_auditor() -> InferenceAuditor:
    config = AuditConfig(
        system_id="perf-test",
        bsv_key="placeholder",
        batch_ms=60_000,  # never auto-flush during benchmark
        batch_size=10_000,
    )
    return InferenceAuditor(
        config=config,
        model_hashes={"m": "sha256:" + "a" * 64},
        _wallet=_FakeWallet(),
        _broadcaster=_FakeBroadcaster(),
        _storage=SQLiteStorage("sqlite://"),
    )


class TestAuditorPerformance:
    _MAX_RECORD_LATENCY_MS = 5.0  # per-call ceiling

    def test_record_call_latency_under_threshold(self):
        """Single record() call must complete in under 5ms."""
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)

        # Warm-up
        for _ in range(5):
            auditor.record("m", {"warm": 1}, {"up": 1})

        # Benchmark
        n = 100
        start = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i, "data": "x" * 32}, {"result": i * 2})
        elapsed_ms = (time.perf_counter() - start) / n * 1000

        auditor.close()

        assert elapsed_ms < self._MAX_RECORD_LATENCY_MS, (
            f"record() mean latency {elapsed_ms:.2f}ms exceeds {self._MAX_RECORD_LATENCY_MS}ms"
        )

    def test_throughput_100_records_per_second(self):
        """Must sustain at least 100 records/second (baseline for batch systems)."""
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)

        n = 200
        start = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i}, {"o": i})
        elapsed_s = time.perf_counter() - start

        auditor.close()

        rps = n / elapsed_s
        assert rps >= 100, f"Throughput {rps:.0f} rec/s < 100 rec/s minimum"

    def test_pii_stripping_does_not_add_significant_overhead(self):
        """PII stripping must not push record() over the latency ceiling."""
        config = AuditConfig(
            system_id="perf-pii",
            bsv_key="placeholder",
            batch_ms=60_000,
            batch_size=10_000,
            pii_fields=["patient_id", "name", "ssn", "dob", "address"],
        )
        auditor = InferenceAuditor(
            config=config,
            model_hashes={"m": "sha256:" + "b" * 64},
            _wallet=_FakeWallet(),
            _broadcaster=_FakeBroadcaster(),
            _storage=SQLiteStorage("sqlite://"),
        )
        auditor._batch._epoch_ready.wait(timeout=5.0)

        n = 100
        start = time.perf_counter()
        for i in range(n):
            auditor.record(
                "m",
                {
                    "patient_id": f"P{i:06d}",
                    "name": "John Doe",
                    "ssn": "123-45-6789",
                    "dob": "1980-01-01",
                    "address": "123 Main St",
                    "symptoms": "chest pain",
                    "priority_score": 0.95,
                },
                {"priority": 1, "confidence": 0.97},
            )
        elapsed_ms = (time.perf_counter() - start) / n * 1000

        auditor.close()

        assert elapsed_ms < self._MAX_RECORD_LATENCY_MS, (
            f"record() with PII stripping: {elapsed_ms:.2f}ms > {self._MAX_RECORD_LATENCY_MS}ms"
        )
