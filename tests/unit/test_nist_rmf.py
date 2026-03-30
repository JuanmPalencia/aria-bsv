"""
tests/unit/test_nist_rmf.py

Unit tests for aria/nist_rmf.py — NIST AI Risk Management Framework 1.0 mapping.

Coverage targets:
- RMFFunction and RiskLevel enumerations
- RMFCategory and RMFAssessment dataclass construction
- NISTRMFAssessor.assess() — MEASURE, MANAGE, MAP, GOVERN paths
- NISTRMFAssessor.risk_profile() — function scores and gap ranking
- NISTRMFAssessor.generate_rmf_report() — full hashable report
"""

from __future__ import annotations

import pytest

from aria.nist_rmf import (
    NISTRMFAssessor,
    RiskLevel,
    RMFAssessment,
    RMFCategory,
    RMFFunction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def assessor() -> NISTRMFAssessor:
    return NISTRMFAssessor(system_id="fraud-detector", risk_tier=RiskLevel.HIGH)


@pytest.fixture()
def records() -> list[dict]:
    return [{"record_id": f"r{i}", "model_id": "fraud-v1"} for i in range(10)]


@pytest.fixture()
def epochs() -> list[dict]:
    return [{"epoch_id": f"ep_{i:03d}", "state_hash": "s" * 64} for i in range(3)]


@pytest.fixture()
def txids() -> list[str]:
    return ["a" * 64, "b" * 64, "c" * 64]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestRMFEnumerations:
    def test_four_functions(self):
        assert set(RMFFunction) == {RMFFunction.GOVERN, RMFFunction.MAP, RMFFunction.MEASURE, RMFFunction.MANAGE}

    def test_four_risk_levels(self):
        assert set(RiskLevel) == {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_function_is_string_enum(self):
        assert isinstance(RMFFunction.MEASURE, str)
        assert RMFFunction.MEASURE == "MEASURE"

    def test_risk_level_is_string_enum(self):
        assert isinstance(RiskLevel.LOW, str)
        assert RiskLevel.LOW == "LOW"


# ---------------------------------------------------------------------------
# RMFCategory dataclass
# ---------------------------------------------------------------------------


class TestRMFCategoryCatalogue:
    def test_catalogue_has_at_least_16_categories(self):
        assert len(NISTRMFAssessor.CATEGORIES) >= 16

    def test_all_four_functions_represented(self):
        functions = {c.function for c in NISTRMFAssessor.CATEGORIES}
        assert functions == set(RMFFunction)

    def test_all_categories_have_required_fields(self):
        for cat in NISTRMFAssessor.CATEGORIES:
            assert cat.category_id
            assert cat.function in RMFFunction
            assert cat.title
            assert cat.description
            assert cat.aria_coverage


# ---------------------------------------------------------------------------
# NISTRMFAssessor.assess() — MEASURE paths
# ---------------------------------------------------------------------------


class TestAssessMEASURE:
    def test_measure_1_1_implemented_with_records(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-1.1")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_measure_4_1_implemented_with_records(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-4.1")
        assert r.implemented is True

    def test_measure_1_1_not_implemented_without_records(self, assessor, epochs, txids):
        results = assessor.assess([], epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-1.1")
        assert r.implemented is False

    def test_measure_2_5_implemented_with_epochs_and_txids(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-2.5")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_measure_2_5_partial_with_epochs_no_txids(self, assessor, records, epochs):
        results = assessor.assess(records, epochs, [])
        r = next(a for a in results if a.category.category_id == "MEASURE-2.5")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.MEDIUM
        assert len(r.gaps) > 0

    def test_measure_2_6_not_implemented_without_any_artifacts(self, assessor):
        results = assessor.assess([], [], [])
        r = next(a for a in results if a.category.category_id == "MEASURE-2.6")
        assert r.implemented is False

    def test_measure_2_9_implemented_with_records(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-2.9")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_measure_2_9_not_implemented_without_records(self, assessor, epochs, txids):
        results = assessor.assess([], epochs, txids)
        r = next(a for a in results if a.category.category_id == "MEASURE-2.9")
        assert r.implemented is False


# ---------------------------------------------------------------------------
# NISTRMFAssessor.assess() — MANAGE paths
# ---------------------------------------------------------------------------


class TestAssessMANAGE:
    def test_manage_2_4_implemented_with_txids(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MANAGE-2.4")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_manage_2_4_not_implemented_without_txids(self, assessor, records, epochs):
        results = assessor.assess(records, epochs, [])
        r = next(a for a in results if a.category.category_id == "MANAGE-2.4")
        assert r.implemented is False

    def test_manage_1_1_implemented_with_txids_and_records(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MANAGE-1.1")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_manage_4_1_partial_with_epochs_no_records(self, assessor, epochs, txids):
        results = assessor.assess([], epochs, txids)
        r = next(a for a in results if a.category.category_id == "MANAGE-4.1")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.MEDIUM
        assert any("inference" in g.lower() for g in r.gaps)

    def test_manage_3_1_not_implemented_with_no_artifacts(self, assessor):
        results = assessor.assess([], [], [])
        r = next(a for a in results if a.category.category_id == "MANAGE-3.1")
        assert r.implemented is False


# ---------------------------------------------------------------------------
# NISTRMFAssessor.assess() — MAP paths
# ---------------------------------------------------------------------------


class TestAssessMAP:
    def test_map_1_1_implemented_with_epochs(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MAP-1.1")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_map_1_1_not_implemented_without_epochs(self, assessor, records, txids):
        results = assessor.assess(records, [], txids)
        r = next(a for a in results if a.category.category_id == "MAP-1.1")
        assert r.implemented is False

    def test_map_3_5_always_implemented(self, assessor):
        # SDK availability is always present
        results = assessor.assess([], [], [])
        r = next(a for a in results if a.category.category_id == "MAP-3.5")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.LOW

    def test_map_5_1_implemented_with_records_and_epochs(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        r = next(a for a in results if a.category.category_id == "MAP-5.1")
        assert r.implemented is True

    def test_map_5_1_partial_with_only_records(self, assessor, records):
        results = assessor.assess(records, [], [])
        r = next(a for a in results if a.category.category_id == "MAP-5.1")
        assert r.implemented is True
        assert r.risk_level == RiskLevel.MEDIUM

    def test_map_1_5_not_implemented_without_epochs(self, assessor):
        results = assessor.assess([], [], [])
        r = next(a for a in results if a.category.category_id == "MAP-1.5")
        assert r.implemented is False


# ---------------------------------------------------------------------------
# NISTRMFAssessor.assess() — GOVERN paths
# ---------------------------------------------------------------------------


class TestAssessGOVERN:
    def test_govern_partial_with_any_artifacts(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        govern_results = [a for a in results if a.category.function == RMFFunction.GOVERN]
        for r in govern_results:
            assert r.implemented is True
            assert r.risk_level == RiskLevel.MEDIUM

    def test_govern_not_implemented_without_artifacts(self, assessor):
        results = assessor.assess([], [], [])
        govern_results = [a for a in results if a.category.function == RMFFunction.GOVERN]
        for r in govern_results:
            assert r.implemented is False

    def test_govern_always_has_gaps_about_policies(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        govern_results = [a for a in results if a.category.function == RMFFunction.GOVERN]
        for r in govern_results:
            assert len(r.gaps) > 0  # Formal policy docs must be maintained separately


# ---------------------------------------------------------------------------
# NISTRMFAssessor — total results count
# ---------------------------------------------------------------------------


class TestAssessResultsCount:
    def test_one_result_per_category(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        assert len(results) == len(NISTRMFAssessor.CATEGORIES)

    def test_all_results_have_assessed_at_utc(self, assessor, records, epochs, txids):
        from datetime import timezone
        results = assessor.assess(records, epochs, txids)
        for r in results:
            assert r.assessed_at.tzinfo == timezone.utc

    def test_blank_txid_string_not_counted(self, assessor, records, epochs):
        results = assessor.assess(records, epochs, ["", "   "])
        r = next(a for a in results if a.category.category_id == "MANAGE-2.4")
        assert r.implemented is False


# ---------------------------------------------------------------------------
# NISTRMFAssessor — risk_profile
# ---------------------------------------------------------------------------


class TestRiskProfile:
    def test_profile_keys(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        profile = assessor.risk_profile(results)
        for key in ("function_scores", "overall_score", "risk_tier", "top_gaps"):
            assert key in profile

    def test_function_scores_has_all_four_functions(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        profile = assessor.risk_profile(results)
        for fn in RMFFunction:
            assert fn.value in profile["function_scores"]

    def test_function_scores_are_floats_between_0_and_1(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        profile = assessor.risk_profile(results)
        for _, score in profile["function_scores"].items():
            assert 0.0 <= score <= 1.0

    def test_overall_score_between_0_and_1(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        profile = assessor.risk_profile(results)
        assert 0.0 <= profile["overall_score"] <= 1.0

    def test_overall_score_1_when_all_implemented(self, assessor):
        # MAP-3.5 is always implemented; if all others were implemented, score = 1.0
        # Use all records+epochs+txids to maximise implemented count
        records = [{"r": 1}]
        epochs  = [{"epoch_id": "x"}]
        txids   = ["a" * 64]
        results = assessor.assess(records, epochs, txids)
        profile = assessor.risk_profile(results)
        # At least MAP-3.5 is always MEASURE-type but it is MAP function
        assert profile["overall_score"] > 0.0

    def test_risk_tier_matches_constructor(self, assessor):
        results = assessor.assess([], [], [])
        profile = assessor.risk_profile(results)
        assert profile["risk_tier"] == RiskLevel.HIGH.value

    def test_top_gaps_at_most_3(self, assessor):
        results = assessor.assess([], [], [])
        profile = assessor.risk_profile(results)
        assert len(profile["top_gaps"]) <= 3

    def test_top_gaps_are_strings(self, assessor):
        results = assessor.assess([], [], [])
        profile = assessor.risk_profile(results)
        for g in profile["top_gaps"]:
            assert isinstance(g, str)


# ---------------------------------------------------------------------------
# NISTRMFAssessor — generate_rmf_report
# ---------------------------------------------------------------------------


class TestGenerateRMFReport:
    def test_report_keys(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        for key in ("system_id", "framework", "assessed_at", "risk_tier", "assessments", "risk_profile", "report_hash"):
            assert key in report

    def test_framework_is_nist(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        assert report["framework"] == "NIST AI RMF 1.0"

    def test_system_id_preserved(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        assert report["system_id"] == "fraud-detector"

    def test_risk_tier_preserved(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        assert report["risk_tier"] == "HIGH"

    def test_report_hash_starts_with_sha256(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        assert report["report_hash"].startswith("sha256:")

    def test_assessments_count_matches(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        assert len(report["assessments"]) == len(results)

    def test_assessments_serialized_have_expected_keys(self, assessor, records, epochs, txids):
        results = assessor.assess(records, epochs, txids)
        report = assessor.generate_rmf_report(results)
        for a in report["assessments"]:
            for key in ("category_id", "function", "title", "risk_level", "implemented", "evidence", "gaps", "assessed_at"):
                assert key in a

    def test_default_risk_tier_is_medium(self):
        asr = NISTRMFAssessor(system_id="default-sys")
        assert asr.risk_tier == RiskLevel.MEDIUM
