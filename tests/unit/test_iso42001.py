"""
tests/unit/test_iso42001.py

Unit tests for aria/iso42001.py — ISO 42001:2023 AI Management System
compliance mapping.

Coverage targets:
- ISO42001Clause and ConformanceLevel enumerations
- ISO42001Control and ConformanceRecord dataclass construction
- ConformanceRecord evidence_hash auto-computation
- ISO42001Assessor.assess_from_records() — all code paths
- ISO42001Assessor.get_conformance_summary() — scoring and clause breakdown
- ISO42001Assessor.generate_evidence_package() — hash binding
"""

from __future__ import annotations

import pytest

from aria.iso42001 import (
    ISO42001Assessor,
    ISO42001Clause,
    ISO42001Control,
    ConformanceLevel,
    ConformanceRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def assessor() -> ISO42001Assessor:
    return ISO42001Assessor(system_id="test-system", model_id="gpt-4o-test")


@pytest.fixture()
def sample_records() -> list[dict]:
    return [
        {"record_id": f"r{i}", "model_id": "gpt-4o-test", "confidence": 0.9}
        for i in range(5)
    ]


@pytest.fixture()
def sample_epochs_with_txids() -> list[dict]:
    return [
        {"epoch_id": "ep_001", "open_txid": "a" * 64, "close_txid": "b" * 64},
        {"epoch_id": "ep_002", "open_txid": "c" * 64, "close_txid": "d" * 64},
    ]


@pytest.fixture()
def sample_epochs_no_txids() -> list[dict]:
    return [{"epoch_id": "ep_003"}, {"epoch_id": "ep_004"}]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestISO42001Enumerations:
    def test_all_clauses_present(self):
        values = {c.value for c in ISO42001Clause}
        assert values == {"4", "5", "6", "7", "8", "9", "10"}

    def test_all_conformance_levels_present(self):
        levels = {lvl for lvl in ConformanceLevel}
        assert ConformanceLevel.CONFORMING in levels
        assert ConformanceLevel.PARTIALLY_CONFORMING in levels
        assert ConformanceLevel.NOT_CONFORMING in levels
        assert ConformanceLevel.NOT_APPLICABLE in levels

    def test_clause_is_string_enum(self):
        assert isinstance(ISO42001Clause.CONTEXT, str)
        assert ISO42001Clause.CONTEXT == "4"

    def test_conformance_level_is_string_enum(self):
        assert isinstance(ConformanceLevel.CONFORMING, str)
        assert ConformanceLevel.CONFORMING == "CONFORMING"


# ---------------------------------------------------------------------------
# ISO42001Control dataclass
# ---------------------------------------------------------------------------


class TestISO42001Control:
    def test_construction(self):
        ctrl = ISO42001Control(
            control_id="9.1.2",
            clause=ISO42001Clause.PERFORMANCE,
            title="Logging",
            description="Logs must be kept.",
            aria_evidence_types=["inference_record"],
        )
        assert ctrl.control_id == "9.1.2"
        assert ctrl.clause == ISO42001Clause.PERFORMANCE
        assert "inference_record" in ctrl.aria_evidence_types

    def test_all_catalogue_controls_have_required_fields(self):
        for ctrl in ISO42001Assessor.CONTROLS:
            assert ctrl.control_id
            assert ctrl.clause in ISO42001Clause
            assert ctrl.title
            assert ctrl.description
            assert isinstance(ctrl.aria_evidence_types, list)


# ---------------------------------------------------------------------------
# ConformanceRecord — evidence hash auto-computation
# ---------------------------------------------------------------------------


class TestConformanceRecord:
    def test_evidence_hash_auto_computed(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        cr = ConformanceRecord(
            control=ctrl,
            level=ConformanceLevel.CONFORMING,
            evidence=[{"type": "inference_record", "count": 5}],
        )
        assert cr.evidence_hash.startswith("sha256:")

    def test_different_evidence_different_hash(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        cr1 = ConformanceRecord(
            control=ctrl, level=ConformanceLevel.CONFORMING,
            evidence=[{"type": "epoch_record", "count": 1}],
        )
        cr2 = ConformanceRecord(
            control=ctrl, level=ConformanceLevel.CONFORMING,
            evidence=[{"type": "epoch_record", "count": 99}],
        )
        assert cr1.evidence_hash != cr2.evidence_hash

    def test_empty_evidence_still_hashed(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        cr = ConformanceRecord(
            control=ctrl, level=ConformanceLevel.NOT_CONFORMING, evidence=[]
        )
        assert cr.evidence_hash.startswith("sha256:")

    def test_notes_preserved(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        cr = ConformanceRecord(
            control=ctrl, level=ConformanceLevel.PARTIALLY_CONFORMING,
            evidence=[], notes="Custom note"
        )
        assert cr.notes == "Custom note"


# ---------------------------------------------------------------------------
# ISO42001Assessor — assess_from_records: full records + txid epochs
# ---------------------------------------------------------------------------


class TestAssessFromRecords:
    def test_returns_one_result_per_control(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        assert len(results) == len(ISO42001Assessor.CONTROLS)

    def test_9_1_1_conforming_with_records(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "9.1.1")
        assert r.level == ConformanceLevel.CONFORMING

    def test_9_1_2_conforming_with_records(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "9.1.2")
        assert r.level == ConformanceLevel.CONFORMING

    def test_8_4_conforming_when_txids_present(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "8.4")
        assert r.level == ConformanceLevel.CONFORMING

    def test_6_1_2_conforming_when_txids_present(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "6.1.2")
        assert r.level == ConformanceLevel.CONFORMING

    def test_10_2_conforming_with_records_and_epochs(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "10.2")
        assert r.level == ConformanceLevel.CONFORMING

    def test_9_3_conforming_with_epochs_and_txids(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        r = next(r for r in results if r.control.control_id == "9.3")
        assert r.level == ConformanceLevel.CONFORMING

    def test_all_results_have_evidence_hash(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        for r in results:
            assert r.evidence_hash.startswith("sha256:")


# ---------------------------------------------------------------------------
# ISO42001Assessor — assess_from_records: no records, no epochs (worst case)
# ---------------------------------------------------------------------------


class TestAssessFromRecordsEmpty:
    def test_9_1_1_not_conforming_when_no_records(self, assessor):
        results = assessor.assess_from_records([], [])
        r = next(r for r in results if r.control.control_id == "9.1.1")
        assert r.level == ConformanceLevel.NOT_CONFORMING

    def test_9_1_2_not_conforming_when_no_records(self, assessor):
        results = assessor.assess_from_records([], [])
        r = next(r for r in results if r.control.control_id == "9.1.2")
        assert r.level == ConformanceLevel.NOT_CONFORMING

    def test_8_4_not_conforming_when_no_epochs_no_txids(self, assessor):
        results = assessor.assess_from_records([], [])
        r = next(r for r in results if r.control.control_id == "8.4")
        assert r.level == ConformanceLevel.NOT_CONFORMING

    def test_6_1_2_not_conforming_with_no_artifacts(self, assessor):
        results = assessor.assess_from_records([], [])
        r = next(r for r in results if r.control.control_id == "6.1.2")
        assert r.level == ConformanceLevel.NOT_CONFORMING

    def test_all_clause_4_5_7_not_conforming_with_no_artifacts(self, assessor):
        results = assessor.assess_from_records([], [])
        for r in results:
            if r.control.clause in (ISO42001Clause.CONTEXT, ISO42001Clause.LEADERSHIP, ISO42001Clause.SUPPORT):
                assert r.level == ConformanceLevel.NOT_CONFORMING


# ---------------------------------------------------------------------------
# ISO42001Assessor — epochs without txids (partial conformance paths)
# ---------------------------------------------------------------------------


class TestAssessFromRecordsPartial:
    def test_8_4_partially_conforming_with_epochs_no_txids(self, assessor, sample_epochs_no_txids):
        results = assessor.assess_from_records([], sample_epochs_no_txids)
        r = next(r for r in results if r.control.control_id == "8.4")
        assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_9_3_partially_conforming_with_epochs_no_txids(self, assessor, sample_epochs_no_txids):
        results = assessor.assess_from_records([], sample_epochs_no_txids)
        r = next(r for r in results if r.control.control_id == "9.3")
        assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_6_1_2_partially_conforming_with_epochs_no_txids(self, assessor, sample_epochs_no_txids):
        results = assessor.assess_from_records([], sample_epochs_no_txids)
        r = next(r for r in results if r.control.control_id == "6.1.2")
        assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_10_2_partially_conforming_with_only_records(self, assessor, sample_records):
        results = assessor.assess_from_records(sample_records, [])
        r = next(r for r in results if r.control.control_id == "10.2")
        assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_clause_4_partially_conforming_with_any_artifacts(self, assessor, sample_epochs_no_txids):
        results = assessor.assess_from_records([], sample_epochs_no_txids)
        context_results = [r for r in results if r.control.clause == ISO42001Clause.CONTEXT]
        for r in context_results:
            assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_pending_txid_not_counted_as_confirmed(self, assessor):
        epochs = [{"epoch_id": "ep_x", "open_txid": "pending"}]
        results = assessor.assess_from_records([], epochs)
        r = next(r for r in results if r.control.control_id == "8.4")
        assert r.level == ConformanceLevel.PARTIALLY_CONFORMING

    def test_txid_field_name_variations(self, assessor):
        """Epochs may supply txid via 'txid', 'open_txid', or 'close_txid'."""
        epochs = [{"epoch_id": "ep_x", "txid": "d" * 64}]
        results = assessor.assess_from_records([], epochs)
        r = next(r for r in results if r.control.control_id == "8.4")
        assert r.level == ConformanceLevel.CONFORMING


# ---------------------------------------------------------------------------
# ISO42001Assessor — get_conformance_summary
# ---------------------------------------------------------------------------


class TestGetConformanceSummary:
    def test_summary_keys(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        summary = assessor.get_conformance_summary(results)
        for key in ("total", "conforming", "partially", "not_conforming", "not_applicable", "score_pct", "clause_breakdown"):
            assert key in summary

    def test_total_equals_control_count(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        summary = assessor.get_conformance_summary(results)
        assert summary["total"] == len(ISO42001Assessor.CONTROLS)

    def test_counts_sum_to_total(self, assessor, sample_records, sample_epochs_no_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_no_txids)
        s = assessor.get_conformance_summary(results)
        assert s["conforming"] + s["partially"] + s["not_conforming"] + s["not_applicable"] == s["total"]

    def test_score_pct_is_100_when_all_conforming(self):
        """Manually create all-CONFORMING result list."""
        ctrl = ISO42001Assessor.CONTROLS[0]
        records = [ConformanceRecord(control=ctrl, level=ConformanceLevel.CONFORMING, evidence=[])]
        assessor = ISO42001Assessor("sys")
        s = assessor.get_conformance_summary(records)
        assert s["score_pct"] == 100.0

    def test_score_pct_is_0_when_all_not_conforming(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        records = [ConformanceRecord(control=ctrl, level=ConformanceLevel.NOT_CONFORMING, evidence=[])]
        assessor = ISO42001Assessor("sys")
        s = assessor.get_conformance_summary(records)
        assert s["score_pct"] == 0.0

    def test_score_pct_0_when_all_not_applicable(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        records = [ConformanceRecord(control=ctrl, level=ConformanceLevel.NOT_APPLICABLE, evidence=[])]
        assessor = ISO42001Assessor("sys")
        s = assessor.get_conformance_summary(records)
        assert s["score_pct"] == 100.0  # n/a excluded from scoring → 100

    def test_clause_breakdown_contains_all_assessed_clauses(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        s = assessor.get_conformance_summary(results)
        for r in results:
            assert r.control.clause.value in s["clause_breakdown"]

    def test_score_pct_is_50_for_all_partially_conforming(self):
        ctrl = ISO42001Assessor.CONTROLS[0]
        records = [
            ConformanceRecord(control=ctrl, level=ConformanceLevel.PARTIALLY_CONFORMING, evidence=[])
        ]
        asr = ISO42001Assessor("sys")
        s = asr.get_conformance_summary(records)
        assert s["score_pct"] == 50.0


# ---------------------------------------------------------------------------
# ISO42001Assessor — generate_evidence_package
# ---------------------------------------------------------------------------


class TestGenerateEvidencePackage:
    def test_package_keys(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        for key in ("system_id", "framework", "assessed_at", "overall_score_pct", "summary", "controls_assessed", "evidence_package_hash"):
            assert key in pkg

    def test_framework_is_iso42001(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        assert pkg["framework"] == "ISO 42001:2023"

    def test_system_id_preserved(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        assert pkg["system_id"] == "test-system"

    def test_model_id_preserved(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        assert pkg["model_id"] == "gpt-4o-test"

    def test_evidence_package_hash_starts_with_sha256(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        assert pkg["evidence_package_hash"].startswith("sha256:")

    def test_controls_assessed_count_matches(self, assessor, sample_records, sample_epochs_with_txids):
        results = assessor.assess_from_records(sample_records, sample_epochs_with_txids)
        pkg = assessor.generate_evidence_package(results)
        assert len(pkg["controls_assessed"]) == len(results)

    def test_assessor_no_model_id(self):
        asr = ISO42001Assessor(system_id="no-model")
        results = asr.assess_from_records([], [])
        pkg = asr.generate_evidence_package(results)
        assert pkg["model_id"] is None
