"""Tests for aria.compare — Model comparison across epochs."""

from __future__ import annotations

import time
import pytest

from aria.core.record import AuditRecord
from aria.compare import ModelComparator, ComparisonResult, ModelStats, _compute_stats
from aria.storage.sqlite import SQLiteStorage


def _storage_with_two_models():
    """Create storage with two models in one epoch."""
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-1",
        system_id="sys-1",
        open_txid="tx_" + "a" * 60,
        model_hashes={},
        state_hash="sha256:" + "b" * 64,
        opened_at=now,
    )

    # Model A: high confidence, high latency
    for i in range(20):
        storage.save_record(AuditRecord(
            epoch_id="ep-1",
            model_id="model-a",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.9 + (i % 3) * 0.03,
            latency_ms=300 + i * 10,
        ))

    # Model B: lower confidence, lower latency
    for i in range(20, 40):
        storage.save_record(AuditRecord(
            epoch_id="ep-1",
            model_id="model-b",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.7 + (i % 3) * 0.03,
            latency_ms=100 + i * 5,
        ))

    return storage


def _storage_with_two_epochs():
    """Create storage with same model across two epochs."""
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    for eid, offset in [("ep-1", 0), ("ep-2", 100)]:
        storage.save_epoch_open(
            epoch_id=eid,
            system_id="sys-1",
            open_txid=f"tx_{eid}_" + "a" * 55,
            model_hashes={},
            state_hash="sha256:" + "b" * 64,
            opened_at=now + offset,
        )
        for i in range(15):
            storage.save_record(AuditRecord(
                epoch_id=eid,
                model_id="model-x",
                input_hash="sha256:" + f"{i + offset:064x}",
                output_hash="sha256:" + f"{i + offset + 100:064x}",
                sequence=i,
                confidence=0.85 + offset * 0.001,
                latency_ms=200 + offset,
            ))

    return storage


class TestComputeStats:
    def test_empty_records(self):
        stats = _compute_stats("empty", [])
        assert stats.count == 0
        assert stats.avg_confidence is None

    def test_basic_stats(self):
        storage = _storage_with_two_models()
        records = storage.list_records_by_epoch("ep-1")
        model_a_recs = [r for r in records if r.model_id == "model-a"]
        stats = _compute_stats("model-a", model_a_recs)
        assert stats.count == 20
        assert stats.avg_confidence is not None
        assert stats.avg_confidence > 0.8
        assert stats.p95_latency_ms > 0


class TestModelComparator:
    def test_compare_models_basic(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        result = cmp.compare_models("model-a", "model-b", epoch_id="ep-1")
        assert isinstance(result, ComparisonResult)
        assert result.group_a.count == 20
        assert result.group_b.count == 20

    def test_compare_detects_winner(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        result = cmp.compare_models("model-a", "model-b", epoch_id="ep-1")
        # Model A has higher confidence
        assert result.winner == "model-a"
        assert result.confidence_delta > 0

    def test_compare_to_dict(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        result = cmp.compare_models("model-a", "model-b", epoch_id="ep-1")
        d = result.to_dict()
        assert "group_a" in d
        assert "group_b" in d
        assert "confidence_delta" in d

    def test_compare_summary(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        result = cmp.compare_models("model-a", "model-b", epoch_id="ep-1")
        text = result.summary()
        assert "model-a" in text
        assert "model-b" in text

    def test_compare_epochs(self):
        storage = _storage_with_two_epochs()
        cmp = ModelComparator(storage)
        result = cmp.compare_epochs("ep-1", "ep-2", model_id="model-x")
        assert result.group_a.count == 15
        assert result.group_b.count == 15

    def test_rank_models_by_confidence(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        ranking = cmp.rank_models(epoch_id="ep-1")
        assert len(ranking) == 2
        assert ranking[0].label == "model-a"  # higher confidence

    def test_rank_models_by_latency(self):
        storage = _storage_with_two_models()
        cmp = ModelComparator(storage)
        ranking = cmp.rank_models(epoch_id="ep-1", by="latency")
        assert len(ranking) == 2
        # Model B has lower latency
        assert ranking[0].avg_latency_ms <= ranking[1].avg_latency_ms

    def test_small_sample_warning(self):
        storage = SQLiteStorage(dsn="sqlite://")
        now = int(time.time())
        storage.save_epoch_open(
            epoch_id="ep-small",
            system_id="sys",
            open_txid="tx_" + "z" * 60,
            model_hashes={},
            state_hash="sha256:" + "b" * 64,
            opened_at=now,
        )
        for i in range(5):
            storage.save_record(AuditRecord(
                epoch_id="ep-small",
                model_id="tiny-model",
                input_hash="sha256:" + f"{i:064x}",
                output_hash="sha256:" + f"{i + 100:064x}",
                sequence=i,
                confidence=0.8,
                latency_ms=100,
            ))
        cmp = ModelComparator(storage)
        result = cmp.compare_models("tiny-model", "nonexistent", epoch_id="ep-small")
        assert any("small sample" in n for n in result.notes)
