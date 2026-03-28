"""Tests for aria.regulatory_export — RegulatoryExporter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aria.regulatory_export import (
    EpochSummaryExport,
    ExportFormat,
    RegulatoryExporter,
    RegulatoryPackage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(confidence: float | None = 0.8, latency_ms: float | None = 100.0):
    r = MagicMock()
    r.confidence = confidence
    r.latency_ms = latency_ms
    return r


def _storage(epoch_data: dict[str, list], has_get_epoch: bool = False):
    storage = MagicMock()
    storage.list_records_by_epoch.side_effect = lambda eid: epoch_data.get(eid, [])
    if has_get_epoch:
        mock_epoch = MagicMock()
        mock_epoch.opened_at  = "2025-01-01T00:00:00"
        mock_epoch.closed_at  = "2025-01-01T01:00:00"
        mock_epoch.tx_open    = "abc123"
        mock_epoch.tx_close   = "def456"
        mock_epoch.model_id   = "test-model"
        storage.get_epoch.return_value = mock_epoch
    return storage


# ---------------------------------------------------------------------------
# ExportFormat enum
# ---------------------------------------------------------------------------

class TestExportFormat:
    def test_values(self):
        assert ExportFormat.EU_AI_ACT.value   == "eu_ai_act"
        assert ExportFormat.NIST_AI_RMF.value == "nist_ai_rmf"
        assert ExportFormat.ISO_42001.value   == "iso_42001"
        assert ExportFormat.GENERIC.value     == "generic"


# ---------------------------------------------------------------------------
# RegulatoryExporter.export
# ---------------------------------------------------------------------------

class TestRegulatoryExporterExport:
    def test_empty_epochs(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], system_id="my-system")
        assert pkg.system_id == "my-system"
        assert pkg.epoch_summaries == []

    def test_single_epoch_summary(self):
        recs = [_record(0.8, 100.0) for _ in range(5)]
        exp = RegulatoryExporter(_storage({"ep-1": recs}))
        pkg = exp.export(["ep-1"])
        assert len(pkg.epoch_summaries) == 1
        s = pkg.epoch_summaries[0]
        assert s.epoch_id == "ep-1"
        assert s.n_records == 5
        assert s.mean_confidence == pytest.approx(0.8)
        assert s.mean_latency_ms == pytest.approx(100.0)

    def test_multiple_epochs(self):
        storage = _storage({"ep-1": [_record()], "ep-2": [_record()]})
        exp = RegulatoryExporter(storage)
        pkg = exp.export(["ep-1", "ep-2"])
        assert len(pkg.epoch_summaries) == 2

    def test_format_stored(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], fmt=ExportFormat.NIST_AI_RMF)
        assert pkg.format == ExportFormat.NIST_AI_RMF

    def test_eu_ai_act_compliance_notes(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], fmt=ExportFormat.EU_AI_ACT)
        assert len(pkg.compliance_notes) > 0
        assert any("Art." in n for n in pkg.compliance_notes)

    def test_nist_compliance_notes(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], fmt=ExportFormat.NIST_AI_RMF)
        assert any("GOVERN" in n for n in pkg.compliance_notes)

    def test_iso_compliance_notes(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], fmt=ExportFormat.ISO_42001)
        assert any("ISO 42001" in n for n in pkg.compliance_notes)

    def test_generic_no_notes(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([], fmt=ExportFormat.GENERIC)
        assert pkg.compliance_notes == []

    def test_generated_at_set(self):
        exp = RegulatoryExporter(_storage({}))
        pkg = exp.export([])
        assert "T" in pkg.generated_at  # ISO 8601

    def test_metadata_epoch_count(self):
        exp = RegulatoryExporter(_storage({"ep-1": [], "ep-2": []}))
        pkg = exp.export(["ep-1", "ep-2"])
        assert pkg.metadata["epoch_count"] == 2


# ---------------------------------------------------------------------------
# With lineage tracker
# ---------------------------------------------------------------------------

class TestRegulatoryExporterWithLineage:
    def test_lineage_included(self):
        tracker = MagicMock()
        tracker.export_lineage.return_value = {
            "epoch_id": "ep-1",
            "model_id": "m",
            "model_version": "1.0",
        }
        exp = RegulatoryExporter(_storage({"ep-1": []}), tracker=tracker)
        pkg = exp.export(["ep-1"])
        assert len(pkg.lineage_records) == 1
        assert pkg.lineage_records[0]["epoch_id"] == "ep-1"

    def test_no_lineage_when_empty(self):
        tracker = MagicMock()
        tracker.export_lineage.return_value = {}  # empty = no lineage
        exp = RegulatoryExporter(_storage({}), tracker=tracker)
        pkg = exp.export([])
        assert pkg.lineage_records == []


# ---------------------------------------------------------------------------
# RegulatoryPackage.to_dict / to_json
# ---------------------------------------------------------------------------

class TestRegulatoryPackageSerialization:
    def _pkg(self):
        return RegulatoryPackage(
            format=ExportFormat.EU_AI_ACT,
            system_id="test-system",
            generated_at="2025-01-01T00:00:00+00:00",
            epoch_summaries=[
                EpochSummaryExport(
                    epoch_id="ep-1",
                    n_records=10,
                    mean_confidence=0.8,
                    mean_latency_ms=100.0,
                    opened_at="2025-01-01",
                    closed_at="2025-01-02",
                    bsv_tx_open="abc",
                    bsv_tx_close="def",
                    model_id="m",
                )
            ],
            compliance_notes=["Note 1"],
        )

    def test_to_dict(self):
        d = self._pkg().to_dict()
        assert d["format"] == "eu_ai_act"
        assert d["system_id"] == "test-system"
        assert len(d["epoch_summaries"]) == 1
        assert d["epoch_summaries"][0]["epoch_id"] == "ep-1"

    def test_to_json_valid(self):
        j = self._pkg().to_json()
        parsed = json.loads(j)
        assert parsed["format"] == "eu_ai_act"

    def test_to_json_pretty(self):
        j = self._pkg().to_json(indent=4)
        assert "\n" in j

    def test_str_representation(self):
        s = str(self._pkg())
        assert "eu_ai_act" in s
        assert "test-system" in s


# ---------------------------------------------------------------------------
# EpochSummaryExport with no confidence/latency
# ---------------------------------------------------------------------------

class TestEpochSummaryNoMetrics:
    def test_no_confidence_records(self):
        recs = [_record(confidence=None, latency_ms=None) for _ in range(5)]
        exp = RegulatoryExporter(_storage({"ep-1": recs}))
        pkg = exp.export(["ep-1"])
        s = pkg.epoch_summaries[0]
        assert s.mean_confidence is None
        assert s.mean_latency_ms is None
