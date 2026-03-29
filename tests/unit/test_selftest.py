"""Tests for aria.selftest — End-to-end health check."""

from __future__ import annotations

import pytest

from aria.selftest import (
    selftest,
    SelftestReport,
    CheckResult,
    _check_hasher,
    _check_merkle,
    _check_record,
    _check_storage,
    _check_epoch,
)


class TestCheckResult:
    def test_pass_result(self):
        r = CheckResult(name="test", ok=True, message="OK")
        assert r.ok
        assert r.name == "test"

    def test_fail_result(self):
        r = CheckResult(name="test", ok=False, message="bad")
        assert not r.ok


class TestIndividualChecks:
    def test_hasher(self):
        result = _check_hasher()
        assert result.ok
        assert result.name == "hasher"

    def test_merkle(self):
        result = _check_merkle()
        assert result.ok
        assert result.name == "merkle"
        assert result.details.get("leaves") == 3

    def test_record(self):
        result = _check_record()
        assert result.ok
        assert result.name == "record"

    def test_storage(self):
        result = _check_storage()
        assert result.ok
        assert result.name == "storage"

    def test_epoch(self):
        result = _check_epoch()
        assert result.ok
        assert result.name == "epoch"


class TestSelftest:
    def test_selftest_passes_locally(self):
        report = selftest(bsv=False)
        assert isinstance(report, SelftestReport)
        assert report.passed
        assert len(report.checks) == 5

    def test_selftest_has_timing(self):
        report = selftest(bsv=False)
        assert report.total_duration_ms >= 0
        for check in report.checks:
            assert check.duration_ms >= 0

    def test_selftest_to_dict(self):
        report = selftest(bsv=False)
        d = report.to_dict()
        assert d["passed"] is True
        assert "checks" in d
        assert len(d["checks"]) == 5

    def test_selftest_summary(self):
        report = selftest(bsv=False)
        text = report.summary()
        assert "ARIA Self-Test Report" in text
        assert "ALL PASSED" in text
        assert "[PASS]" in text

    def test_selftest_all_check_names(self):
        report = selftest(bsv=False)
        names = {c.name for c in report.checks}
        assert names == {"hasher", "merkle", "record", "storage", "epoch"}
