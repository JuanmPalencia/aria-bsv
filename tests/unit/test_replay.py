"""Tests for aria.replay — ReplayEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.replay import ReplayEngine, ReplayRecord, ReplayReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(record_id: str = "r1", input_data: dict | None = None, output_data: str = "result"):
    r = MagicMock()
    r.record_id = record_id
    r.input_data = input_data or {"text": "hello"}
    r.output_data = output_data
    return r


def _storage(epoch_records: dict[str, list]):
    storage = MagicMock()
    storage.list_records_by_epoch.side_effect = lambda eid: epoch_records.get(eid, [])
    return storage


def _identity_fn(input_data: dict):
    return ("replayed", 0.85)


def _failing_fn(input_data: dict):
    raise RuntimeError("model failed")


# ---------------------------------------------------------------------------
# ReplayEngine.replay_epoch
# ---------------------------------------------------------------------------

class TestReplayEngineEpoch:
    def test_empty_epoch(self):
        engine = ReplayEngine(_storage({"ep-1": []}), _identity_fn)
        report = engine.replay_epoch("ep-1")
        assert report.total == 0
        assert report.succeeded == 0

    def test_single_record(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1")
        assert report.total == 1
        assert report.succeeded == 1
        assert report.failed == 0

    def test_multiple_records(self):
        recs = [_record(f"r{i}") for i in range(5)]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1")
        assert report.total == 5
        assert report.succeeded == 5

    def test_failed_records(self):
        recs = [_record("r1"), _record("r2")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _failing_fn)
        report = engine.replay_epoch("ep-1")
        assert report.failed == 2
        assert report.succeeded == 0

    def test_records_stored(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1")
        assert len(report.records) == 1
        assert report.records[0].original_record_id == "r1"

    def test_source_epoch_preserved(self):
        engine = ReplayEngine(_storage({"ep-1": []}), _identity_fn)
        report = engine.replay_epoch("ep-1", target_epoch_label="replay-run-1")
        assert report.source_epoch == "ep-1"
        assert report.target_label == "replay-run-1"

    def test_confidence_extracted(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), lambda d: ("result", 0.95))
        report = engine.replay_epoch("ep-1")
        assert report.records[0].new_confidence == pytest.approx(0.95)

    def test_no_confidence_when_single_return(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), lambda d: "result-only")
        report = engine.replay_epoch("ep-1")
        assert report.records[0].new_confidence is None

    def test_error_record_has_error_string(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _failing_fn)
        report = engine.replay_epoch("ep-1")
        assert "model failed" in report.records[0].error


# ---------------------------------------------------------------------------
# ReplayEngine.replay_epoch with compare=True
# ---------------------------------------------------------------------------

class TestReplayEngineCompare:
    def test_match_rate_100_when_same(self):
        # identity_fn returns ("replayed", 0.85) → output = "replayed"
        recs = [_record("r1", output_data="replayed")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1", compare=True)
        assert report.match_rate == pytest.approx(1.0)

    def test_match_rate_0_when_different(self):
        recs = [_record("r1", output_data="different output")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1", compare=True)
        assert report.match_rate == pytest.approx(0.0)

    def test_match_rate_none_when_no_compare(self):
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        report = engine.replay_epoch("ep-1", compare=False)
        assert report.match_rate is None


# ---------------------------------------------------------------------------
# ReplayEngine.replay_records
# ---------------------------------------------------------------------------

class TestReplayRecords:
    def test_replay_explicit_records(self):
        recs = [_record("r1"), _record("r2")]
        engine = ReplayEngine(_storage({}), _identity_fn)
        report = engine.replay_records(recs)
        assert report.total == 2
        assert report.succeeded == 2

    def test_empty_records(self):
        engine = ReplayEngine(_storage({}), _identity_fn)
        report = engine.replay_records([])
        assert report.total == 0


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestReplayRecording:
    def test_records_to_auditor(self):
        auditor = MagicMock()
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        engine.replay_epoch("ep-1", auditor=auditor, model_id="replay-v2")
        auditor.record.assert_called_once()
        args = auditor.record.call_args[0]
        assert args[0] == "replay-v2"

    def test_records_to_aria(self):
        aria = MagicMock()
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        engine.replay_epoch("ep-1", aria=aria)
        aria.record.assert_called_once()

    def test_no_record_on_failed(self):
        auditor = MagicMock()
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _failing_fn)
        engine.replay_epoch("ep-1", auditor=auditor)
        auditor.record.assert_not_called()

    def test_metadata_has_replay_flag(self):
        auditor = MagicMock()
        recs = [_record("r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        engine.replay_epoch("ep-1", auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["replay"] is True

    def test_source_record_id_in_metadata(self):
        auditor = MagicMock()
        recs = [_record("original-r1")]
        engine = ReplayEngine(_storage({"ep-1": recs}), _identity_fn)
        engine.replay_epoch("ep-1", auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["source_record"] == "original-r1"


# ---------------------------------------------------------------------------
# ReplayReport.__str__
# ---------------------------------------------------------------------------

class TestReplayReportStr:
    def test_str_basic(self):
        report = ReplayReport(source_epoch="ep-1", target_label="replay")
        report.total = 10
        report.succeeded = 9
        report.failed = 1
        s = str(report)
        assert "ep-1" in s
        assert "10" in s

    def test_str_with_match_rate(self):
        report = ReplayReport(source_epoch="ep", target_label="r", match_rate=0.95)
        report.total = 5
        report.succeeded = 5
        s = str(report)
        assert "95.0%" in s
