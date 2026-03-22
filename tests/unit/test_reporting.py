"""Tests for ARIA report generation."""

from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

import pytest

from aria.reporting import ReportGenerator, EpochReport
from aria.storage.sqlite import SQLiteStorage
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> SQLiteStorage:
    return SQLiteStorage("sqlite://")


def _make_record(
    epoch_id: str,
    model_id: str = "model-v1",
    latency_ms: int = 100,
    confidence: float | None = 0.9,
    seq: int = 1,
) -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id=model_id,
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=seq,
        confidence=confidence,
        latency_ms=latency_ms,
        metadata={},
    )


def _seed_closed_epoch(storage, epoch_id, records):
    now = int(time.time() * 1000)
    storage.save_epoch_open(
        epoch_id, "sys-test", "tx-open",
        {"model-v1": "sha256:" + "c" * 64},
        "sha256:" + "d" * 64, now
    )
    for rec in records:
        storage.save_record(rec)
    storage.save_epoch_close(
        epoch_id, "tx-close",
        "sha256:" + "0" * 64,
        len(records), now + 5000
    )


# ---------------------------------------------------------------------------
# EpochReport data model
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_build_report_closed_epoch(self):
        storage = _make_storage()
        recs = [_make_record("ep-rep-1", seq=i) for i in range(1, 4)]
        _seed_closed_epoch(storage, "ep-rep-1", recs)

        gen = ReportGenerator(storage)
        report = gen.build_report("ep-rep-1")

        assert report.epoch_id == "ep-rep-1"
        assert report.system_id == "sys-test"
        assert report.is_closed
        assert report.records_count == 3
        assert bool(report.merkle_root)

    def test_build_report_missing_epoch_raises(self):
        storage = _make_storage()
        gen = ReportGenerator(storage)
        with pytest.raises(ValueError, match="not found"):
            gen.build_report("nonexistent-epoch")

    def test_model_reports_contain_stats(self):
        storage = _make_storage()
        recs = [
            _make_record("ep-mr", model_id="m-A", latency_ms=100, seq=1),
            _make_record("ep-mr", model_id="m-A", latency_ms=200, seq=2),
        ]
        _seed_closed_epoch(storage, "ep-mr", recs)

        gen = ReportGenerator(storage)
        report = gen.build_report("ep-mr")

        assert len(report.model_reports) == 1
        mr = report.model_reports[0]
        assert mr.model_id == "m-A"
        assert mr.record_count == 2
        assert mr.mean_latency_ms == 150.0

    def test_compliance_checks_pass_for_healthy_epoch(self):
        storage = _make_storage()
        recs = [_make_record("ep-comp", seq=1)]
        _seed_closed_epoch(storage, "ep-comp", recs)

        gen = ReportGenerator(storage)
        report = gen.build_report("ep-comp")

        assert report.compliance_pass
        assert all(c.passed for c in report.compliance_checks)

    def test_compliance_checks_fail_for_open_epoch(self):
        storage = _make_storage()
        now = int(time.time() * 1000)
        storage.save_epoch_open("ep-open", "sys", "tx-open", {"m": "h"}, "sh", now)

        gen = ReportGenerator(storage)
        report = gen.build_report("ep-open")

        assert not report.compliance_pass
        close_check = next(c for c in report.compliance_checks if "CLOSE" in c.name)
        assert not close_check.passed


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

class TestRenderText:
    def test_render_text_contains_epoch_id(self):
        storage = _make_storage()
        recs = [_make_record("ep-txt", seq=1)]
        _seed_closed_epoch(storage, "ep-txt", recs)

        gen = ReportGenerator(storage)
        text = gen.render_text("ep-txt")

        assert "ep-txt" in text
        assert "ARIA AUDIT REPORT" in text
        assert "PASS" in text or "FAIL" in text

    def test_render_text_shows_model_stats(self):
        storage = _make_storage()
        recs = [_make_record("ep-txtm", latency_ms=250, seq=i) for i in range(1, 4)]
        _seed_closed_epoch(storage, "ep-txtm", recs)

        gen = ReportGenerator(storage)
        text = gen.render_text("ep-txtm")

        assert "model-v1" in text
        assert "250" in text  # mean latency


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------

class TestRenderJSON:
    def test_render_json_is_valid(self):
        storage = _make_storage()
        recs = [_make_record("ep-json", seq=1)]
        _seed_closed_epoch(storage, "ep-json", recs)

        gen = ReportGenerator(storage)
        raw = gen.render_json("ep-json")
        data = json.loads(raw)

        assert data["epoch"]["epoch_id"] == "ep-json"
        assert data["compliance_pass"] is True
        assert len(data["model_stats"]) == 1

    def test_json_has_all_required_fields(self):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-json2", [_make_record("ep-json2")])

        gen = ReportGenerator(storage)
        data = json.loads(gen.render_json("ep-json2"))

        assert "aria_version" in data
        assert "generated_at" in data
        assert "epoch" in data
        assert "model_stats" in data
        assert "compliance" in data


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

class TestRenderHTML:
    def test_render_html_is_valid(self):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-html", [_make_record("ep-html")])

        gen = ReportGenerator(storage)
        html = gen.render_html("ep-html")

        assert "<!DOCTYPE html>" in html
        assert "ep-html" in html
        assert "ARIA Audit Report" in html

    def test_render_html_includes_compliance_badges(self):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-html2", [_make_record("ep-html2")])

        gen = ReportGenerator(storage)
        html = gen.render_html("ep-html2")

        assert "PASS" in html
        assert "badge" in html


# ---------------------------------------------------------------------------
# Save to file
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_save_text(self, tmp_path):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-save", [_make_record("ep-save")])

        gen = ReportGenerator(storage)
        dest = gen.save("ep-save", path=tmp_path / "report.txt", fmt="text")
        assert dest.exists()
        content = dest.read_text()
        assert "ARIA AUDIT REPORT" in content

    def test_save_json(self, tmp_path):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-savej", [_make_record("ep-savej")])

        gen = ReportGenerator(storage)
        dest = gen.save("ep-savej", path=tmp_path / "report.json", fmt="json")
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["epoch"]["epoch_id"] == "ep-savej"

    def test_save_html(self, tmp_path):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-saveh", [_make_record("ep-saveh")])

        gen = ReportGenerator(storage)
        dest = gen.save("ep-saveh", path=tmp_path / "report.html", fmt="html")
        assert dest.exists()
        assert "<!DOCTYPE html>" in dest.read_text()

    def test_save_unknown_format_raises(self, tmp_path):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-savex", [_make_record("ep-savex")])

        gen = ReportGenerator(storage)
        with pytest.raises(ValueError, match="Unknown format"):
            gen.save("ep-savex", path=tmp_path / "report.xyz", fmt="xyz")  # type: ignore

    def test_save_creates_parent_dirs(self, tmp_path):
        storage = _make_storage()
        _seed_closed_epoch(storage, "ep-dirs", [_make_record("ep-dirs")])

        gen = ReportGenerator(storage)
        nested = tmp_path / "a" / "b" / "c" / "report.txt"
        dest = gen.save("ep-dirs", path=nested, fmt="text")
        assert dest.exists()
