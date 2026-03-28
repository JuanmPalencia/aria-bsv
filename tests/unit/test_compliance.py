"""Tests for aria.compliance — compliance checker and report generation."""

from __future__ import annotations

import json

import pytest

from aria.compliance import (
    CheckSeverity,
    ComplianceCheck,
    ComplianceChecker,
    ComplianceReport,
    Regulation,
    SystemComplianceReport,
)


# ---------------------------------------------------------------------------
# Helpers / mock storage
# ---------------------------------------------------------------------------

class FakeRecord:
    def __init__(
        self,
        record_id="rec-1",
        model_id="gpt-4",
        confidence=0.95,
        latency_ms=120.0,
        metadata=None,
    ):
        self.record_id = record_id
        self.model_id = model_id
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.metadata = metadata or {}


class FakeEpoch:
    def __init__(
        self,
        epoch_id="ep-1",
        system_id="sys-1",
        open_txid="open-tx-abc",
        close_txid="close-tx-xyz",
        merkle_root="deadbeef" * 8,
        records_count=10,
        model_hashes=None,
    ):
        self.epoch_id = epoch_id
        self.system_id = system_id
        self.open_txid = open_txid
        self.close_txid = close_txid
        self.merkle_root = merkle_root
        self.records_count = records_count
        self.model_hashes = model_hashes if model_hashes is not None else {"gpt-4": "abc"}


class MockStorage:
    def __init__(self, epochs=None, records=None):
        self._epochs = epochs or []
        self._records = records or {}

    def list_epochs(self, system_id=None, limit=10000):
        result = self._epochs
        if system_id:
            result = [e for e in result if e.system_id == system_id]
        return result[:limit]

    def list_records_by_epoch(self, epoch_id):
        return self._records.get(epoch_id, [])


def _make_full_storage(n_records=15):
    records = [
        FakeRecord(
            record_id=f"rec-{i}",
            model_id="gpt-4",
            confidence=0.9,
            latency_ms=100.0,
        )
        for i in range(n_records)
    ]
    epoch = FakeEpoch(records_count=n_records)
    storage = MockStorage(
        epochs=[epoch],
        records={epoch.epoch_id: records},
    )
    return storage, epoch, records


# ---------------------------------------------------------------------------
# ComplianceCheck
# ---------------------------------------------------------------------------

class TestComplianceCheck:
    def test_to_dict_keys(self):
        c = ComplianceCheck(
            rule_id="BRC121-4.1",
            regulation=Regulation.BRC121,
            description="Epoch must be anchored",
            passed=True,
        )
        d = c.to_dict()
        assert d["rule_id"] == "BRC121-4.1"
        assert d["regulation"] == "BRC-121"
        assert d["passed"] is True


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def test_passed_when_all_critical_pass(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "desc", True, CheckSeverity.CRITICAL),
                ComplianceCheck("r2", Regulation.EU_AI, "desc", False, CheckSeverity.WARNING),
            ],
        )
        assert report.passed is True  # Warning doesn't fail overall

    def test_failed_when_critical_fails(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "desc", False, CheckSeverity.CRITICAL),
            ],
        )
        assert report.passed is False

    def test_violations_list(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "Missing txid", False, detail="no tx"),
                ComplianceCheck("r2", Regulation.EU_AI, "No model id", True),
            ],
        )
        assert len(report.violations) == 1
        assert "r1" in report.violations[0]
        assert "Missing txid" in report.violations[0]

    def test_critical_failures_count(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "x", False, CheckSeverity.CRITICAL),
                ComplianceCheck("r2", Regulation.EU_AI,  "y", False, CheckSeverity.WARNING),
                ComplianceCheck("r3", Regulation.GDPR,   "z", True,  CheckSeverity.CRITICAL),
            ],
        )
        assert report.critical_failures == 1
        assert report.warnings == 1

    def test_to_dict_has_violations(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "Bad", False, CheckSeverity.CRITICAL),
            ],
        )
        d = report.to_dict()
        assert "violations" in d
        assert len(d["violations"]) == 1

    def test_to_json_valid(self):
        report = ComplianceReport(epoch_id="ep-1", system_id="sys-1")
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["epoch_id"] == "ep-1"

    def test_to_text_contains_epoch_id(self):
        report = ComplianceReport(epoch_id="ep-1", system_id="sys-1")
        text = report.to_text()
        assert "ep-1" in text

    def test_to_text_shows_pass_fail(self):
        report = ComplianceReport(
            epoch_id="ep-1",
            system_id="sys-1",
            checks=[
                ComplianceCheck("r1", Regulation.BRC121, "x", True),
                ComplianceCheck("r2", Regulation.EU_AI, "y", False, CheckSeverity.CRITICAL),
            ],
        )
        text = report.to_text()
        assert "FAIL" in text


# ---------------------------------------------------------------------------
# SystemComplianceReport
# ---------------------------------------------------------------------------

class TestSystemComplianceReport:
    def _make_passing(self):
        return ComplianceReport(epoch_id="pass", system_id="s")

    def _make_failing(self):
        r = ComplianceReport(
            epoch_id="fail", system_id="s",
            checks=[ComplianceCheck("x", Regulation.BRC121, "x", False, CheckSeverity.CRITICAL)]
        )
        return r

    def test_compliance_rate(self):
        summary = SystemComplianceReport(
            system_id="s",
            epoch_reports=[self._make_passing(), self._make_failing()],
        )
        assert summary.compliance_rate == 0.5

    def test_compliance_rate_all_pass(self):
        summary = SystemComplianceReport(
            system_id="s",
            epoch_reports=[self._make_passing(), self._make_passing()],
        )
        assert summary.compliance_rate == 1.0

    def test_empty_reports_rate_is_one(self):
        summary = SystemComplianceReport(system_id="s")
        assert summary.compliance_rate == 1.0

    def test_all_violations_aggregated(self):
        summary = SystemComplianceReport(
            system_id="s",
            epoch_reports=[self._make_passing(), self._make_failing()],
        )
        assert len(summary.all_violations) == 1
        assert "fail" in summary.all_violations[0]


# ---------------------------------------------------------------------------
# ComplianceChecker — full integration
# ---------------------------------------------------------------------------

class TestComplianceChecker:
    def test_fully_compliant_epoch(self):
        storage, epoch, _ = _make_full_storage(n_records=15)
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        assert report.passed is True
        assert report.critical_failures == 0

    def test_missing_open_txid_fails(self):
        epoch = FakeEpoch(open_txid=None)
        records = [FakeRecord()] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        assert report.passed is False
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-4.1" in rule_ids

    def test_pending_open_txid_fails(self):
        epoch = FakeEpoch(open_txid="pending")
        records = [FakeRecord()] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-4.1" in rule_ids

    def test_missing_close_txid_fails(self):
        epoch = FakeEpoch(close_txid=None)
        records = [FakeRecord()] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-4.2" in rule_ids

    def test_missing_merkle_root_fails(self):
        epoch = FakeEpoch(merkle_root="")
        records = [FakeRecord()] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-4.3" in rule_ids

    def test_record_count_mismatch_fails(self):
        epoch = FakeEpoch(records_count=99)   # Declared 99
        records = [FakeRecord()] * 15           # But only 15 stored
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-4.4" in rule_ids

    def test_missing_model_hashes_warning(self):
        epoch = FakeEpoch(model_hashes={})
        records = [FakeRecord()] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        warn_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "BRC121-6.1" in warn_ids
        # Warnings should not fail overall if all critical pass
        # (other criticals may pass)

    def test_records_missing_model_id_warns(self):
        epoch = FakeEpoch()
        records = [FakeRecord(model_id="")] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "EUAI-13.1" in rule_ids

    def test_low_confidence_coverage_info(self):
        epoch = FakeEpoch()
        records = [FakeRecord(confidence=None)] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "EUAI-13.2" in rule_ids

    def test_missing_latency_warns(self):
        epoch = FakeEpoch()
        records = [FakeRecord(latency_ms=0)] * 15
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "EUAI-17.1" in rule_ids

    def test_pii_email_in_record_id_fails(self):
        epoch = FakeEpoch()
        records = [FakeRecord(record_id="user@example.com-rec-1")] * 1 + [FakeRecord()] * 14
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "GDPR-5.1c" in rule_ids

    def test_pii_metadata_key_warns(self):
        epoch = FakeEpoch()
        records = [FakeRecord(metadata={"email": "x@y.com"})] + [FakeRecord()] * 14
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "GDPR-5.1c-meta" in rule_ids

    def test_low_record_count_info(self):
        epoch = FakeEpoch(records_count=3)
        records = [FakeRecord()] * 3
        storage = MockStorage(epochs=[epoch], records={epoch.epoch_id: records})
        checker = ComplianceChecker(storage, min_records=10)
        report = checker.check_epoch(epoch.epoch_id)
        rule_ids = [c.rule_id for c in report.checks if not c.passed]
        assert "INT-MIN-REC" in rule_ids

    def test_epoch_not_found_returns_failed_report(self):
        storage = MockStorage()
        checker = ComplianceChecker(storage)
        report = checker.check_epoch("nonexistent")
        assert report.passed is False
        assert "not found" in report.violations[0].lower()

    def test_check_system_aggregates(self):
        epoch1 = FakeEpoch(epoch_id="ep-1", records_count=15)
        epoch2 = FakeEpoch(epoch_id="ep-2", records_count=15)
        records = [FakeRecord()] * 15
        storage = MockStorage(
            epochs=[epoch1, epoch2],
            records={
                "ep-1": records,
                "ep-2": records,
            },
        )
        checker = ComplianceChecker(storage)
        summary = checker.check_system(last_n=5)
        assert summary.total_epochs == 2
        assert summary.compliance_rate == 1.0

    def test_check_system_filters_by_system_id(self):
        ep_a = FakeEpoch(epoch_id="ep-a", system_id="sys-A")
        ep_b = FakeEpoch(epoch_id="ep-b", system_id="sys-B")
        records = [FakeRecord()] * 15
        storage = MockStorage(
            epochs=[ep_a, ep_b],
            records={"ep-a": records, "ep-b": records},
        )
        checker = ComplianceChecker(storage)
        summary_a = checker.check_system(system_id="sys-A")
        assert all(r.system_id == "sys-A" for r in summary_a.epoch_reports)

    def test_quick_check_returns_bool(self):
        storage, epoch, _ = _make_full_storage()
        checker = ComplianceChecker(storage)
        result = checker.quick_check(epoch.epoch_id)
        assert isinstance(result, bool)

    def test_framework_filter_brc120_only(self):
        storage, epoch, _ = _make_full_storage()
        checker = ComplianceChecker(storage, frameworks=[Regulation.BRC121])
        report = checker.check_epoch(epoch.epoch_id)
        regulations = {c.regulation for c in report.checks}
        assert regulations == {Regulation.BRC121}
