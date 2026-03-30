"""Performance benchmark report — prints a formatted summary table.

Run with ``pytest tests/perf/test_bench_report.py -v -s`` to see the report.
Each test measures actual numbers, asserts thresholds, and prints the result.
The final summary test collects all measurements into one consolidated table.
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable

import pytest

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.core.hasher import canonical_json, hash_object
from aria.core.merkle import ARIAMerkleTree, verify_proof
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

# ---------------------------------------------------------------------------
# Fake infrastructure (mirrors test_bench_auditor.py)
# ---------------------------------------------------------------------------

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
        system_id="bench-report",
        bsv_key="placeholder",
        batch_ms=60_000,
        batch_size=10_000,
    )
    return InferenceAuditor(
        config=config,
        model_hashes={"m": "sha256:" + "a" * 64},
        _wallet=_FakeWallet(),
        _broadcaster=_FakeBroadcaster(),
        _storage=SQLiteStorage("sqlite://"),
    )


def _fake_hash(i: int) -> str:
    return "sha256:" + hashlib.sha256(i.to_bytes(4, "big")).hexdigest()


# ---------------------------------------------------------------------------
# Shared results store — populated per-test, printed in the summary
# ---------------------------------------------------------------------------

_RESULTS: dict[str, tuple[float, float, str]] = {}
# key -> (measured_value, threshold, unit)


def _mean_ms(fn: Callable, iterations: int, *args) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    return (time.perf_counter() - start) / iterations * 1000


# ---------------------------------------------------------------------------
# Hasher benchmarks
# ---------------------------------------------------------------------------


class TestHasherReport:
    """Prints hasher latency numbers with thresholds."""

    _ITERATIONS = 2_000
    _THRESHOLD_MS = 1.0

    def test_canonical_json_small(self):
        obj = {"model_id": "triage", "confidence": 0.97, "sequence": 42}
        mean = _mean_ms(canonical_json, self._ITERATIONS, obj)
        _RESULTS["Hasher / canonical_json (small)"] = (mean, self._THRESHOLD_MS, "ms/call")
        print(f"\n  [Hasher] canonical_json small:  {mean:.4f} ms  (threshold < {self._THRESHOLD_MS} ms)")
        assert mean < self._THRESHOLD_MS

    def test_canonical_json_nested(self):
        obj = {
            "metadata": {"decision_class": "triage_1", "custom": {"ward": "A3"}},
            "input_hash": "sha256:" + "a" * 64,
            "output_hash": "sha256:" + "b" * 64,
            "epoch_id": "ep_1742848200000_0001",
            "model_id": "triage-v2.3",
            "sequence": 999,
            "confidence": 0.991,
            "latency_ms": 47,
        }
        mean = _mean_ms(canonical_json, self._ITERATIONS, obj)
        _RESULTS["Hasher / canonical_json (nested)"] = (mean, self._THRESHOLD_MS, "ms/call")
        print(f"\n  [Hasher] canonical_json nested: {mean:.4f} ms  (threshold < {self._THRESHOLD_MS} ms)")
        assert mean < self._THRESHOLD_MS

    def test_hash_object(self):
        obj = {"a": 1, "b": [2, 3], "c": {"d": 4.5, "e": None}}
        mean = _mean_ms(hash_object, self._ITERATIONS, obj)
        _RESULTS["Hasher / hash_object"] = (mean, self._THRESHOLD_MS, "ms/call")
        print(f"\n  [Hasher] hash_object:           {mean:.4f} ms  (threshold < {self._THRESHOLD_MS} ms)")
        assert mean < self._THRESHOLD_MS

    def test_throughput(self):
        """Report calls/sec throughput derived from latency benchmark."""
        obj = {"x": 1, "y": "hello", "z": [1, 2, 3]}
        iterations = 5_000
        start = time.perf_counter()
        for _ in range(iterations):
            hash_object(obj)
        elapsed = time.perf_counter() - start
        cps = iterations / elapsed
        threshold_cps = 5_000.0
        _RESULTS["Hasher / hash_object throughput"] = (cps, threshold_cps, "calls/sec")
        print(f"\n  [Hasher] throughput:            {cps:,.0f} calls/sec  (threshold > {threshold_cps:,.0f})")
        assert cps >= threshold_cps, f"{cps:.0f} calls/sec < {threshold_cps:.0f}"

    def test_generate_report(self):
        """Print hasher report table (run after individual tests)."""
        obj_small = {"model_id": "triage", "confidence": 0.97, "sequence": 42}
        obj_nested = {
            "metadata": {"decision_class": "triage_1"},
            "input_hash": "sha256:" + "a" * 64,
            "sequence": 999,
            "confidence": 0.991,
        }
        obj_plain = {"a": 1, "b": [2, 3], "c": {"d": 4.5}}

        rows = [
            ("canonical_json (small obj)",  _mean_ms(canonical_json, 2_000, obj_small),  1.0, "ms/call"),
            ("canonical_json (nested obj)", _mean_ms(canonical_json, 2_000, obj_nested), 1.0, "ms/call"),
            ("hash_object (small)",         _mean_ms(hash_object, 2_000, obj_plain),     1.0, "ms/call"),
        ]

        # Throughput
        iterations = 5_000
        start = time.perf_counter()
        for _ in range(iterations):
            hash_object(obj_small)
        elapsed = time.perf_counter() - start
        cps = iterations / elapsed
        rows_thr = [("hash_object throughput", cps, 5_000, "calls/sec")]

        print("\n")
        print("=" * 70)
        print("  ARIA Hasher Performance Report")
        print("=" * 70)
        print(f"  {'Operation':<30} | {'Result':>12} | {'Threshold':>12}")
        print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}")
        for name, val, thresh, unit in rows:
            status = "OK" if val < thresh else "FAIL"
            print(f"  {name:<30} | {val:>9.4f} ms | < {thresh:>8.1f} ms  [{status}]")
        for name, val, thresh, unit in rows_thr:
            status = "OK" if val >= thresh else "FAIL"
            print(f"  {name:<30} | {val:>9,.0f}    | > {thresh:>8,.0f}     [{status}]")
        print("=" * 70)

        for _name, val, thresh, _unit in rows:
            assert val < thresh
        for _name, val, thresh, _unit in rows_thr:
            assert val >= thresh


# ---------------------------------------------------------------------------
# Merkle benchmarks
# ---------------------------------------------------------------------------


class TestMerkleReport:
    """Prints Merkle build time and proof verification numbers."""

    def _time_build_ms(self, n: int) -> float:
        tree = ARIAMerkleTree()
        start = time.perf_counter()
        for i in range(n):
            tree.add(_fake_hash(i))
        _ = tree.root()
        return (time.perf_counter() - start) * 1000

    def test_build_10(self):
        t = self._time_build_ms(10)
        _RESULTS["Merkle / build 10 leaves"] = (t, 5.0, "ms")
        print(f"\n  [Merkle] build  10 leaves: {t:.3f} ms  (threshold < 5.0 ms)")
        assert t < 5.0

    def test_build_100(self):
        t = self._time_build_ms(100)
        _RESULTS["Merkle / build 100 leaves"] = (t, 10.0, "ms")
        print(f"\n  [Merkle] build 100 leaves: {t:.3f} ms  (threshold < 10.0 ms)")
        assert t < 10.0

    def test_build_500(self):
        t = self._time_build_ms(500)
        _RESULTS["Merkle / build 500 leaves"] = (t, 50.0, "ms")
        print(f"\n  [Merkle] build 500 leaves: {t:.3f} ms  (threshold < 50.0 ms)")
        assert t < 50.0

    def test_proof_verification(self):
        tree = ARIAMerkleTree()
        for i in range(500):
            tree.add(_fake_hash(i))
        root = tree.root()
        target = _fake_hash(250)
        proof = tree.proof(target)

        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            verify_proof(root, proof, target)
        mean_ms = (time.perf_counter() - start) / iterations * 1000
        _RESULTS["Merkle / proof verify (500-leaf)"] = (mean_ms, 1.0, "ms/call")
        print(f"\n  [Merkle] proof verify (500-leaf): {mean_ms:.4f} ms  (threshold < 1.0 ms)")
        assert mean_ms < 1.0

    def test_generate_report(self):
        """Print Merkle report table."""
        sizes = [10, 50, 100, 250, 500, 1000]
        timings = [(n, self._time_build_ms(n)) for n in sizes]
        thresholds = {10: 5.0, 50: 8.0, 100: 10.0, 250: 30.0, 500: 50.0, 1000: 100.0}

        # Proof verification
        tree = ARIAMerkleTree()
        for i in range(500):
            tree.add(_fake_hash(i))
        root = tree.root()
        target = _fake_hash(250)
        proof = tree.proof(target)
        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            verify_proof(root, proof, target)
        verify_ms = (time.perf_counter() - start) / iterations * 1000

        print("\n")
        print("=" * 70)
        print("  ARIA Merkle Performance Report")
        print("=" * 70)
        print(f"  {'Operation':<35} | {'Result':>10} | {'Threshold':>10}")
        print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}")
        for n, t in timings:
            thresh = thresholds[n]
            status = "OK" if t < thresh else "FAIL"
            print(f"  {'build ' + str(n) + ' leaves':<35} | {t:>7.3f} ms | < {thresh:>6.1f} ms  [{status}]")
        status = "OK" if verify_ms < 1.0 else "FAIL"
        print(f"  {'proof verify (500-leaf tree)':<35} | {verify_ms:>7.4f} ms | < {'1.0':>6} ms  [{status}]")
        print("=" * 70)

        for n, t in timings:
            assert t < thresholds[n], f"build {n} took {t:.3f}ms > {thresholds[n]}ms"
        assert verify_ms < 1.0


# ---------------------------------------------------------------------------
# Auditor benchmarks
# ---------------------------------------------------------------------------


class TestAuditorReport:
    """Prints InferenceAuditor.record() throughput numbers."""

    _MAX_LATENCY_MS = 5.0

    def test_record_latency(self):
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)
        for _ in range(5):
            auditor.record("m", {"warm": 1}, {"up": 1})
        n = 200
        start = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i, "data": "x" * 32}, {"result": i * 2})
        elapsed_ms = (time.perf_counter() - start) / n * 1000
        auditor.close()
        _RESULTS["Auditor / record() mean latency"] = (elapsed_ms, self._MAX_LATENCY_MS, "ms/call")
        print(f"\n  [Auditor] record() mean latency: {elapsed_ms:.3f} ms  (threshold < {self._MAX_LATENCY_MS} ms)")
        assert elapsed_ms < self._MAX_LATENCY_MS

    def test_throughput(self):
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)
        n = 500
        start = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i}, {"o": i})
        elapsed_s = time.perf_counter() - start
        auditor.close()
        rps = n / elapsed_s
        _RESULTS["Auditor / record() throughput"] = (rps, 100.0, "rec/sec")
        print(f"\n  [Auditor] throughput: {rps:,.0f} rec/sec  (threshold > 100 rec/sec)")
        assert rps >= 100

    def test_pii_stripping_latency(self):
        config = AuditConfig(
            system_id="bench-pii-report",
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
        n = 200
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
        _RESULTS["Auditor / record() with PII strip"] = (elapsed_ms, self._MAX_LATENCY_MS, "ms/call")
        print(f"\n  [Auditor] record() + PII strip: {elapsed_ms:.3f} ms  (threshold < {self._MAX_LATENCY_MS} ms)")
        assert elapsed_ms < self._MAX_LATENCY_MS

    def test_generate_report(self):
        """Print auditor report table."""
        # Latency benchmark
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)
        for _ in range(5):
            auditor.record("m", {"warm": 1}, {"up": 1})
        n = 300
        start = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i, "data": "x" * 64}, {"result": i})
        elapsed_s = time.perf_counter() - start
        auditor.close()

        mean_ms = elapsed_s / n * 1000
        rps = n / elapsed_s

        # PII latency
        config = AuditConfig(
            system_id="bench-pii2",
            bsv_key="placeholder",
            batch_ms=60_000,
            batch_size=10_000,
            pii_fields=["patient_id", "name", "ssn"],
        )
        auditor_pii = InferenceAuditor(
            config=config,
            model_hashes={"m": "sha256:" + "c" * 64},
            _wallet=_FakeWallet(),
            _broadcaster=_FakeBroadcaster(),
            _storage=SQLiteStorage("sqlite://"),
        )
        auditor_pii._batch._epoch_ready.wait(timeout=5.0)
        start = time.perf_counter()
        for i in range(200):
            auditor_pii.record(
                "m",
                {"patient_id": f"P{i}", "name": "Jane", "ssn": "000", "score": 0.9},
                {"out": i},
            )
        pii_elapsed_ms = (time.perf_counter() - start) / 200 * 1000
        auditor_pii.close()

        rows = [
            ("record() mean latency",       mean_ms,    5.0,   "ms/call",  "<"),
            ("record() throughput",         rps,        100.0, "rec/sec",  ">"),
            ("record() + PII strip",        pii_elapsed_ms, 5.0, "ms/call", "<"),
        ]

        print("\n")
        print("=" * 70)
        print("  ARIA Auditor Performance Report")
        print("=" * 70)
        print(f"  {'Operation':<30} | {'Result':>14} | {'Threshold':>12}")
        print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*12}")
        for name, val, thresh, unit, op in rows:
            if unit == "ms/call":
                ok = val < thresh
                print(f"  {name:<30} | {val:>10.3f} ms  | {op} {thresh:>7.1f} ms  [{'OK' if ok else 'FAIL'}]")
            else:
                ok = val >= thresh
                print(f"  {name:<30} | {val:>9,.0f} r/s | {op} {thresh:>7,.0f} r/s  [{'OK' if ok else 'FAIL'}]")
        print("=" * 70)

        assert mean_ms < 5.0
        assert rps >= 100
        assert pii_elapsed_ms < 5.0


# ---------------------------------------------------------------------------
# Full summary benchmark
# ---------------------------------------------------------------------------


class TestFullBenchmarkSummary:
    """Consolidated benchmark summary — run last for the master table."""

    def test_full_benchmark_summary(self):
        """Run all core benchmarks and print a single consolidated table."""
        from aria import __version__

        version = getattr(__version__, "__version__", __version__) if not isinstance(__version__, str) else __version__

        # --- Hasher ---
        obj_small  = {"model_id": "triage", "confidence": 0.97, "sequence": 42}
        obj_nested = {
            "metadata": {"class": "triage_1", "custom": {"ward": "A3"}},
            "input_hash": "sha256:" + "a" * 64,
            "epoch_id": "ep_1742848200000_0001",
            "model_id": "triage-v2.3",
            "sequence": 999,
            "confidence": 0.991,
        }
        obj_plain  = {"a": 1, "b": [2, 3], "c": {"d": 4.5, "e": None}}

        cj_small  = _mean_ms(canonical_json, 2_000, obj_small)
        cj_nested = _mean_ms(canonical_json, 2_000, obj_nested)
        ho_plain  = _mean_ms(hash_object, 2_000, obj_plain)

        iters = 5_000
        t0 = time.perf_counter()
        for _ in range(iters):
            hash_object(obj_small)
        hasher_rps = iters / (time.perf_counter() - t0)

        # --- Merkle ---
        def _build_ms(n: int) -> float:
            tree = ARIAMerkleTree()
            t0 = time.perf_counter()
            for i in range(n):
                tree.add(_fake_hash(i))
            _ = tree.root()
            return (time.perf_counter() - t0) * 1000

        m10   = _build_ms(10)
        m100  = _build_ms(100)
        m500  = _build_ms(500)
        m1000 = _build_ms(1000)

        tree = ARIAMerkleTree()
        for i in range(500):
            tree.add(_fake_hash(i))
        root    = tree.root()
        target  = _fake_hash(250)
        proof   = tree.proof(target)
        t0 = time.perf_counter()
        for _ in range(1_000):
            verify_proof(root, proof, target)
        proof_ms = (time.perf_counter() - t0) / 1_000 * 1000

        # --- Auditor ---
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)
        for _ in range(5):
            auditor.record("m", {"w": 1}, {"u": 1})
        n = 400
        t0 = time.perf_counter()
        for i in range(n):
            auditor.record("m", {"i": i, "data": "x" * 32}, {"result": i})
        elapsed = time.perf_counter() - t0
        auditor.close()

        aud_ms  = elapsed / n * 1000
        aud_rps = n / elapsed

        # --- Print table ---
        print("\n")
        print("=" * 72)
        print(f"  ARIA Performance Benchmark Report  —  aria-bsv v{version}")
        print("=" * 72)
        hdr = f"  {'Component':<12}  {'Operation':<28}  {'Result':>12}  {'Threshold':>14}"
        sep = f"  {'-'*12}  {'-'*28}  {'-'*12}  {'-'*14}"
        print(hdr)
        print(sep)

        rows = [
            ("Hasher",  "canonical_json (small)",   f"{cj_small:.4f} ms",   "< 1.0 ms"),
            ("Hasher",  "canonical_json (nested)",  f"{cj_nested:.4f} ms",  "< 1.0 ms"),
            ("Hasher",  "hash_object",              f"{ho_plain:.4f} ms",   "< 1.0 ms"),
            ("Hasher",  "throughput",               f"{hasher_rps:,.0f} c/s", "> 5,000 c/s"),
            ("Merkle",  "build 10 leaves",          f"{m10:.3f} ms",        "< 5.0 ms"),
            ("Merkle",  "build 100 leaves",         f"{m100:.3f} ms",       "< 10.0 ms"),
            ("Merkle",  "build 500 leaves",         f"{m500:.3f} ms",       "< 50.0 ms"),
            ("Merkle",  "build 1000 leaves",        f"{m1000:.3f} ms",      "< 100.0 ms"),
            ("Merkle",  "proof verify (500-leaf)",  f"{proof_ms:.4f} ms",   "< 1.0 ms"),
            ("Auditor", "record() latency",         f"{aud_ms:.3f} ms",     "< 5.0 ms"),
            ("Auditor", "record() throughput",      f"{aud_rps:,.0f} r/s",  "> 100 r/s"),
        ]

        prev_comp = ""
        for comp, op, result, threshold in rows:
            comp_label = comp if comp != prev_comp else ""
            prev_comp = comp
            print(f"  {comp_label:<12}  {op:<28}  {result:>12}  {threshold:>14}")

        print("=" * 72)

        # Assertions
        assert cj_small  < 1.0, f"cj_small {cj_small:.4f}ms >= 1ms"
        assert cj_nested < 1.0, f"cj_nested {cj_nested:.4f}ms >= 1ms"
        assert ho_plain  < 1.0, f"ho_plain {ho_plain:.4f}ms >= 1ms"
        assert hasher_rps >= 5_000, f"hasher throughput {hasher_rps:.0f} c/s < 5000"
        assert m10   < 5.0,    f"merkle 10   {m10:.3f}ms >= 5ms"
        assert m100  < 10.0,   f"merkle 100  {m100:.3f}ms >= 10ms"
        assert m500  < 50.0,   f"merkle 500  {m500:.3f}ms >= 50ms"
        assert m1000 < 100.0,  f"merkle 1000 {m1000:.3f}ms >= 100ms"
        assert proof_ms < 1.0, f"proof verify {proof_ms:.4f}ms >= 1ms"
        assert aud_ms  < 5.0,  f"auditor latency {aud_ms:.3f}ms >= 5ms"
        assert aud_rps >= 100, f"auditor throughput {aud_rps:.0f} r/s < 100"
