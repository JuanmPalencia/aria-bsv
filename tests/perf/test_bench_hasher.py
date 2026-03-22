"""Performance: canonical JSON hashing must stay under 1ms per call.

These are latency regression guards, not throughput benchmarks.
They run in the normal pytest suite and fail if overhead regresses.
"""

from __future__ import annotations

import time

import pytest

from aria.core.hasher import canonical_json, hash_object


class TestHasherPerformance:
    _ITERATIONS = 1_000
    _MAX_LATENCY_MS = 1.0  # 1ms ceiling per call

    def _mean_ms(self, fn, *args) -> float:
        start = time.perf_counter()
        for _ in range(self._ITERATIONS):
            fn(*args)
        return (time.perf_counter() - start) / self._ITERATIONS * 1000

    def test_canonical_json_small_object(self):
        obj = {"model_id": "triage", "confidence": 0.97, "sequence": 42}
        mean = self._mean_ms(canonical_json, obj)
        assert mean < self._MAX_LATENCY_MS, (
            f"canonical_json small object: {mean:.3f}ms > {self._MAX_LATENCY_MS}ms"
        )

    def test_canonical_json_nested_object(self):
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
        mean = self._mean_ms(canonical_json, obj)
        assert mean < self._MAX_LATENCY_MS, (
            f"canonical_json nested: {mean:.3f}ms > {self._MAX_LATENCY_MS}ms"
        )

    def test_hash_object_under_threshold(self):
        obj = {"a": 1, "b": [2, 3], "c": {"d": 4.5, "e": None}}
        mean = self._mean_ms(hash_object, obj)
        assert mean < self._MAX_LATENCY_MS, (
            f"hash_object: {mean:.3f}ms > {self._MAX_LATENCY_MS}ms"
        )

    def test_hash_object_deterministic_across_calls(self):
        """Sanity: hash is stable (determinism test collocated with perf)."""
        obj = {"z": 1, "a": 2, "m": {"x": 3}}
        hashes = {hash_object(obj) for _ in range(100)}
        assert len(hashes) == 1
