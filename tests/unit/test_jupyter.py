"""Tests for aria.jupyter — Jupyter/notebook cell magic."""

from __future__ import annotations

import pytest

from aria.jupyter import NotebookRecord, NotebookTracker


class TestNotebookTracker:
    """Tests for NotebookTracker (works without IPython)."""

    def test_create(self):
        tracker = NotebookTracker(system_id="test-nb")
        assert tracker.system_id == "test-nb"
        assert tracker.records == []

    def test_track_cell(self):
        tracker = NotebookTracker()
        rec = tracker.track_cell(
            cell_source="x = 1 + 1",
            cell_output=2,
            model_id="test-model",
        )
        assert isinstance(rec, NotebookRecord)
        assert rec.model_id == "test-model"
        assert rec.cell_number == 1
        assert len(rec.cell_hash) == 64
        assert len(rec.output_hash) == 64

    def test_track_multiple_cells(self):
        tracker = NotebookTracker()
        tracker.track_cell("cell1", "out1")
        tracker.track_cell("cell2", "out2")
        tracker.track_cell("cell3", "out3")
        assert len(tracker.records) == 3
        assert tracker.records[0].cell_number == 1
        assert tracker.records[1].cell_number == 2
        assert tracker.records[2].cell_number == 3

    def test_track_cell_with_confidence(self):
        tracker = NotebookTracker()
        rec = tracker.track_cell("src", "out", confidence=0.95)
        assert rec.confidence == 0.95

    def test_track_cell_with_metadata(self):
        tracker = NotebookTracker()
        rec = tracker.track_cell("src", "out", metadata={"key": "val"})
        assert rec.metadata == {"key": "val"}

    def test_different_sources_different_hashes(self):
        tracker = NotebookTracker()
        r1 = tracker.track_cell("cell_a", "out")
        r2 = tracker.track_cell("cell_b", "out")
        assert r1.cell_hash != r2.cell_hash

    def test_different_outputs_different_hashes(self):
        tracker = NotebookTracker()
        r1 = tracker.track_cell("cell", "output_1")
        r2 = tracker.track_cell("cell", "output_2")
        assert r1.output_hash != r2.output_hash

    def test_summary(self):
        tracker = NotebookTracker(system_id="my-nb")
        tracker.track_cell("src", "out", model_id="gpt-4")
        summary = tracker.summary()
        assert "my-nb" in summary
        assert "gpt-4" in summary
        assert "Tracked cells: 1" in summary

    def test_complex_output(self):
        tracker = NotebookTracker()
        rec = tracker.track_cell(
            "import pandas as pd\ndf = pd.DataFrame({'a': [1]})",
            {"columns": ["a"], "shape": [1, 1]},
            model_id="analysis",
        )
        assert len(rec.output_hash) == 64

    def test_with_auditor_callback(self):
        from unittest.mock import MagicMock

        mock_auditor = MagicMock()
        tracker = NotebookTracker(auditor=mock_auditor)
        tracker.track_cell("src", "out", model_id="m1")
        mock_auditor.record.assert_called_once()

    def test_auditor_error_does_not_propagate(self):
        from unittest.mock import MagicMock

        mock_auditor = MagicMock()
        mock_auditor.record.side_effect = RuntimeError("boom")
        tracker = NotebookTracker(auditor=mock_auditor)
        rec = tracker.track_cell("src", "out")
        assert rec is not None  # should not raise


class TestNotebookRecord:
    """Tests for NotebookRecord dataclass."""

    def test_to_dict(self):
        r = NotebookRecord(
            record_id="abc",
            model_id="gpt-4",
            cell_hash="a" * 64,
            output_hash="b" * 64,
            confidence=0.9,
            latency_ms=10,
            cell_number=1,
        )
        d = r.to_dict()
        assert d["model_id"] == "gpt-4"
        assert d["confidence"] == 0.9
        assert d["cell_number"] == 1
