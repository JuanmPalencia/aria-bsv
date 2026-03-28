"""Tests for aria.lineage — LineageTracker."""

from __future__ import annotations

import json

import pytest

from aria.lineage import (
    DatasetRef,
    EpochLineage,
    LineageTracker,
    ModelVersion,
)


# ---------------------------------------------------------------------------
# DatasetRef
# ---------------------------------------------------------------------------

class TestDatasetRef:
    def test_basic(self):
        ds = DatasetRef("sst2", "1.0", 67349)
        assert ds.name == "sst2"
        assert ds.version == "1.0"
        assert ds.num_rows == 67349

    def test_to_dict(self):
        ds = DatasetRef("imdb", "2.0", 25000, "s3://bucket/imdb")
        d = ds.to_dict()
        assert d["name"] == "imdb"
        assert d["source_uri"] == "s3://bucket/imdb"

    def test_from_dict(self):
        d = {"name": "sst2", "version": "1.0", "num_rows": 100, "source_uri": ""}
        ds = DatasetRef.from_dict(d)
        assert ds.name == "sst2"
        assert ds.num_rows == 100

    def test_roundtrip(self):
        ds = DatasetRef("sst2", "1.0", 67349, "s3://test")
        ds2 = DatasetRef.from_dict(ds.to_dict())
        assert ds == ds2


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------

class TestModelVersion:
    def test_basic(self):
        mv = ModelVersion(
            model_id="sentiment-v2",
            version="2.1.0",
            training_epochs=5,
            base_model="bert-base-uncased",
        )
        assert mv.model_id == "sentiment-v2"
        assert mv.version == "2.1.0"
        assert mv.created_at != ""

    def test_auto_created_at(self):
        mv = ModelVersion(model_id="m", version="1.0")
        assert "T" in mv.created_at  # ISO 8601

    def test_to_dict(self):
        ds = DatasetRef("sst2", "1.0", 1000)
        mv = ModelVersion(
            model_id="m",
            version="1.0",
            datasets=[ds],
            hyperparams={"lr": 2e-5},
        )
        d = mv.to_dict()
        assert isinstance(d["datasets"], list)
        assert d["datasets"][0]["name"] == "sst2"
        assert d["hyperparams"]["lr"] == pytest.approx(2e-5)

    def test_from_dict(self):
        ds = DatasetRef("sst2", "1.0", 100)
        mv = ModelVersion(
            model_id="m",
            version="2.0",
            datasets=[ds],
            hyperparams={"lr": 1e-4},
            base_model="bert",
        )
        d = mv.to_dict()
        mv2 = ModelVersion.from_dict(d)
        assert mv2.model_id == "m"
        assert mv2.version == "2.0"
        assert len(mv2.datasets) == 1
        assert mv2.datasets[0].name == "sst2"
        assert mv2.hyperparams["lr"] == pytest.approx(1e-4)

    def test_roundtrip(self):
        mv = ModelVersion(
            model_id="x",
            version="3.0",
            training_epochs=10,
            datasets=[DatasetRef("d1", "1.0", 500)],
            hyperparams={"batch": 32},
            base_model="roberta",
            framework="pytorch",
        )
        mv2 = ModelVersion.from_dict(mv.to_dict())
        assert mv2.model_id == mv.model_id
        assert mv2.version == mv.version
        assert mv2.framework == mv.framework
        assert mv2.datasets[0].name == mv.datasets[0].name


# ---------------------------------------------------------------------------
# LineageTracker — model registration
# ---------------------------------------------------------------------------

class TestLineageTrackerRegistration:
    def test_register_and_get(self):
        t = LineageTracker()
        mv = ModelVersion(model_id="m", version="1.0")
        t.register_model(mv)
        assert t.get_model("m", "1.0") is mv

    def test_get_unknown(self):
        t = LineageTracker()
        assert t.get_model("unknown", "1.0") is None

    def test_list_versions(self):
        t = LineageTracker()
        t.register_model(ModelVersion(model_id="m", version="1.0"))
        t.register_model(ModelVersion(model_id="m", version="2.0"))
        versions = t.list_versions("m")
        assert "1.0" in versions
        assert "2.0" in versions

    def test_list_versions_unknown_model(self):
        t = LineageTracker()
        assert t.list_versions("unknown") == []

    def test_multiple_models(self):
        t = LineageTracker()
        t.register_model(ModelVersion(model_id="model-a", version="1.0"))
        t.register_model(ModelVersion(model_id="model-b", version="1.0"))
        assert t.get_model("model-a", "1.0") is not None
        assert t.get_model("model-b", "1.0") is not None


# ---------------------------------------------------------------------------
# LineageTracker — epoch linking
# ---------------------------------------------------------------------------

class TestLineageTrackerLinking:
    def test_link_and_get(self):
        t = LineageTracker()
        mv = ModelVersion(model_id="m", version="1.0")
        t.register_model(mv)
        lineage = t.link_epoch("epoch-1", "m", "1.0")
        assert lineage.epoch_id == "epoch-1"
        assert lineage.model_id == "m"
        assert lineage.model_version == "1.0"
        assert lineage.model_meta is mv

    def test_link_without_registered_model(self):
        t = LineageTracker()
        lineage = t.link_epoch("epoch-x", "unregistered", "1.0")
        assert lineage.model_meta is None

    def test_get_lineage(self):
        t = LineageTracker()
        t.link_epoch("epoch-1", "m", "1.0")
        lineage = t.get_lineage("epoch-1")
        assert lineage is not None
        assert lineage.epoch_id == "epoch-1"

    def test_get_lineage_unknown(self):
        t = LineageTracker()
        assert t.get_lineage("nonexistent") is None

    def test_epochs_for_model(self):
        t = LineageTracker()
        t.link_epoch("ep-1", "m", "1.0")
        t.link_epoch("ep-2", "m", "1.0")
        t.link_epoch("ep-3", "m", "2.0")
        epochs = t.epochs_for_model("m")
        assert set(epochs) == {"ep-1", "ep-2", "ep-3"}

    def test_epochs_for_model_by_version(self):
        t = LineageTracker()
        t.link_epoch("ep-1", "m", "1.0")
        t.link_epoch("ep-2", "m", "2.0")
        epochs = t.epochs_for_model("m", version="1.0")
        assert epochs == ["ep-1"]

    def test_linked_at_set(self):
        t = LineageTracker()
        lineage = t.link_epoch("ep-1", "m", "1.0")
        assert lineage.linked_at != ""
        assert "T" in lineage.linked_at


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestLineageSerialization:
    def test_export_basic(self):
        t = LineageTracker()
        t.link_epoch("ep-1", "model-a", "1.0")
        d = t.export_lineage("ep-1")
        assert d["epoch_id"] == "ep-1"
        assert d["model_id"] == "model-a"
        assert d["model_version"] == "1.0"
        assert "model_meta" not in d  # no registered model

    def test_export_with_model_meta(self):
        t = LineageTracker()
        mv = ModelVersion(
            model_id="m",
            version="1.0",
            datasets=[DatasetRef("sst2", "1.0", 1000)],
            hyperparams={"lr": 2e-5},
        )
        t.register_model(mv)
        t.link_epoch("ep-1", "m", "1.0")
        d = t.export_lineage("ep-1")
        assert "model_meta" in d
        assert d["model_meta"]["model_id"] == "m"
        assert d["model_meta"]["datasets"][0]["name"] == "sst2"

    def test_export_json_serializable(self):
        t = LineageTracker()
        mv = ModelVersion(model_id="m", version="1.0", hyperparams={"lr": 2e-5})
        t.register_model(mv)
        t.link_epoch("ep-1", "m", "1.0")
        d = t.export_lineage("ep-1")
        json.dumps(d)  # Should not raise

    def test_export_unknown_epoch(self):
        t = LineageTracker()
        assert t.export_lineage("unknown") == {}

    def test_import_lineage(self):
        t = LineageTracker()
        mv = ModelVersion(
            model_id="m",
            version="1.0",
            datasets=[DatasetRef("sst2", "1.0", 500)],
        )
        t.register_model(mv)
        t.link_epoch("ep-1", "m", "1.0")
        exported = t.export_lineage("ep-1")

        # Import into a new tracker
        t2 = LineageTracker()
        lineage2 = t2.import_lineage(exported)
        assert lineage2.epoch_id == "ep-1"
        assert lineage2.model_id == "m"
        assert lineage2.model_meta is not None
        assert lineage2.model_meta.datasets[0].name == "sst2"

    def test_import_roundtrip_lookup(self):
        t = LineageTracker()
        t.link_epoch("ep-1", "m", "1.0")
        exported = t.export_lineage("ep-1")
        t2 = LineageTracker()
        t2.import_lineage(exported)
        assert t2.get_lineage("ep-1") is not None


# ---------------------------------------------------------------------------
# EpochLineage.__str__
# ---------------------------------------------------------------------------

class TestEpochLineageStr:
    def test_str_basic(self):
        lineage = EpochLineage(epoch_id="ep-1", model_id="m", model_version="1.0")
        s = str(lineage)
        assert "ep-1" in s
        assert "m" in s

    def test_str_with_meta(self):
        mv = ModelVersion(
            model_id="m",
            version="1.0",
            base_model="bert",
            datasets=[DatasetRef("sst2")],
            hyperparams={"lr": 1e-5},
        )
        lineage = EpochLineage(
            epoch_id="ep-1",
            model_id="m",
            model_version="1.0",
            model_meta=mv,
        )
        s = str(lineage)
        assert "bert" in s
        assert "sst2" in s
        assert "lr" in s
