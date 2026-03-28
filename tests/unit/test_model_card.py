"""Tests for aria.model_card — ModelCardGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.model_card import (
    EpochMetrics,
    ModelCard,
    ModelCardConfig,
    ModelCardGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(confidence: float | None = None, latency_ms: float | None = None):
    r = MagicMock()
    r.confidence = confidence
    r.latency_ms = latency_ms
    return r


def _storage(epoch_data: dict[str, list]):
    storage = MagicMock()
    storage.list_records_by_epoch.side_effect = lambda eid: epoch_data.get(eid, [])
    return storage


def _config(**kwargs):
    defaults = dict(
        model_id="test-model",
        model_name="Test Model",
        version="1.0.0",
        intended_use="Testing",
    )
    defaults.update(kwargs)
    return ModelCardConfig(**defaults)


# ---------------------------------------------------------------------------
# EpochMetrics
# ---------------------------------------------------------------------------

class TestEpochMetrics:
    def test_defaults(self):
        m = EpochMetrics()
        assert m.n_records == 0
        assert m.mean_confidence is None

    def test_fields(self):
        m = EpochMetrics(
            n_records=50,
            mean_confidence=0.87,
            mean_latency_ms=120.0,
            error_rate=0.05,
        )
        assert m.n_records == 50
        assert m.mean_confidence == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# ModelCardGenerator
# ---------------------------------------------------------------------------

class TestModelCardGeneratorGenerate:
    def test_empty_epochs(self):
        gen = ModelCardGenerator(_storage({}))
        card = gen.generate(_config(), epoch_ids=[])
        assert card.metrics.n_records == 0

    def test_single_epoch(self):
        recs = [_record(0.8, 100.0) for _ in range(10)]
        gen = ModelCardGenerator(_storage({"ep-1": recs}))
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert card.metrics.n_records == 10
        assert card.metrics.mean_confidence == pytest.approx(0.8)
        assert card.metrics.mean_latency_ms == pytest.approx(100.0)

    def test_multiple_epochs(self):
        recs1 = [_record(0.7, 100.0) for _ in range(10)]
        recs2 = [_record(0.9, 200.0) for _ in range(10)]
        gen = ModelCardGenerator(_storage({"ep-1": recs1, "ep-2": recs2}))
        card = gen.generate(_config(), epoch_ids=["ep-1", "ep-2"])
        assert card.metrics.n_records == 20
        assert card.metrics.mean_confidence == pytest.approx(0.8)
        assert card.metrics.mean_latency_ms == pytest.approx(150.0)

    def test_p95_latency(self):
        # All 10ms except one 100ms → p95 should be ≤ 100
        recs = [_record(0.8, 10.0) for _ in range(19)] + [_record(0.8, 100.0)]
        gen = ModelCardGenerator(_storage({"ep-1": recs}))
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert card.metrics.p95_latency_ms is not None
        assert card.metrics.p95_latency_ms <= 100.0

    def test_error_rate(self):
        # 5 records with confidence < 0.5, 5 with >= 0.5
        recs = [_record(0.3) for _ in range(5)] + [_record(0.8) for _ in range(5)]
        gen = ModelCardGenerator(_storage({"ep-1": recs}), conf_threshold=0.5)
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert card.metrics.error_rate == pytest.approx(0.5)

    def test_no_confidence_records(self):
        recs = [_record(confidence=None, latency_ms=100.0) for _ in range(5)]
        gen = ModelCardGenerator(_storage({"ep-1": recs}))
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert card.metrics.mean_confidence is None
        assert card.metrics.error_rate is None

    def test_std_confidence(self):
        recs = [_record(0.7), _record(0.9)]
        gen = ModelCardGenerator(_storage({"ep-1": recs}))
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert card.metrics.std_confidence is not None
        assert card.metrics.std_confidence > 0

    def test_epoch_ids_stored(self):
        gen = ModelCardGenerator(_storage({"ep-1": []}))
        card = gen.generate(_config(), epoch_ids=["ep-1"])
        assert "ep-1" in card.epoch_ids

    def test_config_preserved(self):
        cfg = _config(model_name="My Model", version="2.0")
        gen = ModelCardGenerator(_storage({}))
        card = gen.generate(cfg, epoch_ids=[])
        assert card.config.model_name == "My Model"
        assert card.config.version == "2.0"

    def test_generated_at_set(self):
        gen = ModelCardGenerator(_storage({}))
        card = gen.generate(_config())
        assert card.generated_at != ""


# ---------------------------------------------------------------------------
# ModelCard.to_markdown
# ---------------------------------------------------------------------------

class TestModelCardToMarkdown:
    def _card(self, **metrics_kwargs):
        cfg = _config(
            intended_use="AI inference",
            limitations="No support for non-English",
            bias_considerations="May reflect training data bias",
            training_data="SST-2 dataset",
        )
        metrics = EpochMetrics(n_records=50, **metrics_kwargs)
        return ModelCard(config=cfg, metrics=metrics, epoch_ids=["ep-1"])

    def test_contains_model_name(self):
        md = self._card().to_markdown()
        assert "Test Model" in md

    def test_contains_model_id(self):
        md = self._card().to_markdown()
        assert "test-model" in md

    def test_contains_intended_use(self):
        md = self._card().to_markdown()
        assert "AI inference" in md

    def test_contains_limitations(self):
        md = self._card().to_markdown()
        assert "No support for non-English" in md

    def test_contains_metrics_section(self):
        md = self._card(mean_confidence=0.85, mean_latency_ms=120.0).to_markdown()
        assert "Performance Metrics" in md
        assert "0.8500" in md
        assert "120" in md

    def test_no_metrics_message(self):
        card = ModelCard(config=_config(), metrics=EpochMetrics(n_records=0), epoch_ids=[])
        md = card.to_markdown()
        assert "No metrics available" in md

    def test_contains_training_data(self):
        md = self._card().to_markdown()
        assert "SST-2" in md

    def test_extra_sections(self):
        cfg = _config(extra_sections={"Custom Section": "Custom content here"})
        card = ModelCard(config=cfg, metrics=EpochMetrics(), epoch_ids=[])
        md = card.to_markdown()
        assert "Custom Section" in md
        assert "Custom content here" in md

    def test_aria_footer(self):
        md = self._card().to_markdown()
        assert "ARIA BSV" in md


# ---------------------------------------------------------------------------
# ModelCard.to_dict
# ---------------------------------------------------------------------------

class TestModelCardToDict:
    def test_basic(self):
        cfg = _config()
        metrics = EpochMetrics(n_records=10, mean_confidence=0.8)
        card = ModelCard(config=cfg, metrics=metrics, epoch_ids=["ep-1"])
        d = card.to_dict()
        assert d["model_id"] == "test-model"
        assert d["metrics"]["n_records"] == 10
        assert d["epoch_ids"] == ["ep-1"]

    def test_json_serializable(self):
        import json
        card = ModelCard(config=_config(), metrics=EpochMetrics(), epoch_ids=[])
        d = card.to_dict()
        json.dumps(d)  # Should not raise
