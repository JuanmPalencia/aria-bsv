"""Tests for aria.reports — Multi-epoch reporting."""

from __future__ import annotations

import time
import pytest

from aria.core.record import AuditRecord
from aria.reports import MultiReport, MultiEpochReport, EpochSummary
from aria.storage.sqlite import SQLiteStorage


def _storage_with_epochs():
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    for idx, eid in enumerate(["ep-1", "ep-2", "ep-3"]):
        storage.save_epoch_open(
            epoch_id=eid,
            system_id="sys-a" if idx < 2 else "sys-b",
            open_txid=f"tx_{eid}_" + "a" * 55,
            model_hashes={},
            state_hash="sha256:" + "b" * 64,
            opened_at=now - (3 - idx) * 3600,
        )
        if idx < 2:
            storage.save_epoch_close(
                epoch_id=eid,
                close_txid=f"tx_close_{eid}_" + "c" * 50,
                merkle_root="sha256:" + "d" * 64,
                records_count=5,
                closed_at=now - (3 - idx) * 3600 + 600,
            )

        for i in range(5):
            storage.save_record(AuditRecord(
                epoch_id=eid,
                model_id="model-x" if idx < 2 else "model-y",
                input_hash="sha256:" + f"{i + idx * 10:064x}",
                output_hash="sha256:" + f"{i + idx * 10 + 100:064x}",
                sequence=i,
                confidence=0.8 + idx * 0.05,
                latency_ms=100 + idx * 20,
            ))

    return storage


class TestMultiReport:
    def test_all_epochs(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        assert isinstance(report, MultiEpochReport)
        assert report.total_epochs == 3
        assert report.total_records == 15

    def test_system_report(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.system_report("sys-a")
        assert report.total_epochs == 2
        assert report.total_records == 10

    def test_date_range(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.date_range("24h")
        assert report.total_epochs >= 1

    def test_cross_system(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.cross_system(["sys-a", "sys-b"])
        assert report.total_epochs == 3
        assert len(report.models) == 2

    def test_report_summary(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        text = report.summary()
        assert "ARIA" in text
        assert "Epochs" in text

    def test_report_to_json(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        j = report.to_json()
        import json
        data = json.loads(j)
        assert data["total_epochs"] == 3

    def test_report_to_dict(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        d = report.to_dict()
        assert "epochs" in d
        assert len(d["epochs"]) == 3

    def test_report_avg_confidence(self):
        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        assert report.avg_confidence is not None
        assert 0 < report.avg_confidence < 1

    def test_report_save_json(self):
        import tempfile
        from pathlib import Path

        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "report.json")
            report.save(path, fmt="json")
            assert Path(path).exists()

    def test_report_save_html(self):
        import tempfile
        from pathlib import Path

        storage = _storage_with_epochs()
        mr = MultiReport(storage)
        report = mr.all_epochs()
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "report.html")
            report.save(path, fmt="html")
            content = Path(path).read_text()
            assert "<html>" in content
