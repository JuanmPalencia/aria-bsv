"""Tests for aria.zk.statement — EpochStatement + regulatory report."""

from __future__ import annotations

import pytest

from aria.zk.aggregate import AggregateProof, MerkleAggregator
from aria.zk.base import ZKProof
from aria.zk.claims import ClaimResult, ConfidencePercentile, ModelUnchanged
from aria.zk.statement import EpochStatement
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _satisfied_claim(claim_type: str = "test_claim") -> ClaimResult:
    return ClaimResult(
        claim_type=claim_type,
        params={},
        satisfied=True,
        evidence_hash="sha256:" + "a" * 64,
        human_description="Test claim satisfied.",
        eu_ai_act_reference="Art. 9",
    )


def _failed_claim(claim_type: str = "test_claim") -> ClaimResult:
    return ClaimResult(
        claim_type=claim_type,
        params={},
        satisfied=False,
        evidence_hash="sha256:" + "b" * 64,
        human_description="Test claim NOT satisfied.",
        eu_ai_act_reference="Art. 9",
        detail="Something went wrong.",
    )


def _fake_proof(record_id: str) -> ZKProof:
    import hashlib
    return ZKProof(
        proof_bytes=hashlib.sha256(record_id.encode()).digest(),
        public_inputs=["sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64],
        proving_system="mock",
        tier="full_zk",
        model_hash="sha256:" + "d" * 64,
        prover_version="mock-1.0",
        epoch_id="ep_test",
        record_id=record_id,
    )


# ---------------------------------------------------------------------------
# EpochStatement construction
# ---------------------------------------------------------------------------

class TestEpochStatementConstruction:
    def test_statement_hash_computed_on_init(self):
        stmt = EpochStatement(
            epoch_id="ep_1_0001",
            system_id="test-system",
            claims=[_satisfied_claim()],
            n_records=5,
        )
        assert stmt.statement_hash.startswith("sha256:")

    def test_statement_hash_is_deterministic(self):
        claims = [_satisfied_claim()]
        stmt1 = EpochStatement(epoch_id="ep_1", system_id="sys", claims=claims, n_records=3)
        stmt2 = EpochStatement(epoch_id="ep_1", system_id="sys", claims=claims, n_records=3)
        assert stmt1.statement_hash == stmt2.statement_hash

    def test_statement_hash_changes_with_different_claims(self):
        stmt1 = EpochStatement(epoch_id="ep_1", system_id="sys",
                               claims=[_satisfied_claim()], n_records=3)
        stmt2 = EpochStatement(epoch_id="ep_1", system_id="sys",
                               claims=[_failed_claim()], n_records=3)
        assert stmt1.statement_hash != stmt2.statement_hash

    def test_statement_hash_changes_with_epoch_id(self):
        claims = [_satisfied_claim()]
        stmt1 = EpochStatement(epoch_id="ep_1", system_id="sys", claims=claims, n_records=3)
        stmt2 = EpochStatement(epoch_id="ep_2", system_id="sys", claims=claims, n_records=3)
        assert stmt1.statement_hash != stmt2.statement_hash

    def test_no_claims_allowed(self):
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=[], n_records=0)
        assert stmt.statement_hash.startswith("sha256:")


# ---------------------------------------------------------------------------
# all_satisfied
# ---------------------------------------------------------------------------

class TestAllSatisfied:
    def test_all_satisfied_true(self):
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim(), _satisfied_claim("c2")],
            n_records=5,
        )
        assert stmt.all_satisfied() is True

    def test_all_satisfied_false_if_any_fails(self):
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim(), _failed_claim()],
            n_records=5,
        )
        assert stmt.all_satisfied() is False

    def test_no_claims_is_all_satisfied(self):
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=[], n_records=0)
        assert stmt.all_satisfied() is True


# ---------------------------------------------------------------------------
# to_bsv_payload
# ---------------------------------------------------------------------------

class TestToBSVPayload:
    def test_payload_has_required_keys(self):
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim()], n_records=5,
        )
        payload = stmt.to_bsv_payload()
        assert "zk_enabled" in payload
        assert "claims_count" in payload
        assert "all_claims_satisfied" in payload
        assert "claims" in payload
        assert "statement_hash" in payload

    def test_payload_zk_disabled_without_aggregate(self):
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=[], n_records=0)
        assert stmt.to_bsv_payload()["zk_enabled"] is False

    def test_payload_zk_enabled_with_aggregate(self):
        proofs = [_fake_proof(f"rec_{i}") for i in range(3)]
        agg = MerkleAggregator().aggregate(proofs, "ep_1")
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim()],
            aggregate_proof=agg,
            n_records=3,
        )
        payload = stmt.to_bsv_payload()
        assert payload["zk_enabled"] is True
        assert "aggregate_proof" in payload

    def test_payload_claims_count_matches(self):
        claims = [_satisfied_claim(f"c{i}") for i in range(4)]
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=claims, n_records=10)
        assert stmt.to_bsv_payload()["claims_count"] == 4


# ---------------------------------------------------------------------------
# to_regulatory_report
# ---------------------------------------------------------------------------

class TestRegulatoryReport:
    def test_report_contains_epoch_id(self):
        stmt = EpochStatement(
            epoch_id="ep_1774168_0001", system_id="kairos-v2",
            claims=[_satisfied_claim()], n_records=500,
        )
        report = stmt.to_regulatory_report()
        assert "ep_1774168_0001" in report

    def test_report_contains_system_id(self):
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="kairos-v2",
            claims=[_satisfied_claim()], n_records=100,
        )
        assert "kairos-v2" in stmt.to_regulatory_report()

    def test_report_contains_claim_symbols(self):
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim(), _failed_claim()],
            n_records=10,
        )
        report = stmt.to_regulatory_report()
        assert "✓" in report
        assert "✗" in report

    def test_report_contains_overall_result(self):
        stmt_ok = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_satisfied_claim()], n_records=5,
        )
        assert "ALL CLAIMS SATISFIED" in stmt_ok.to_regulatory_report()

        stmt_fail = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[_failed_claim()], n_records=5,
        )
        assert "ONE OR MORE CLAIMS NOT SATISFIED" in stmt_fail.to_regulatory_report()

    def test_report_contains_statement_hash(self):
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=[], n_records=0)
        assert stmt.statement_hash in stmt.to_regulatory_report()

    def test_report_contains_zk_proof_section_when_aggregate_present(self):
        proofs = [_fake_proof(f"rec_{i}") for i in range(2)]
        agg = MerkleAggregator().aggregate(proofs, "ep_1")
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=[], aggregate_proof=agg, n_records=2,
        )
        report = stmt.to_regulatory_report()
        assert "ZERO-KNOWLEDGE" in report
        assert "merkle" in report

    def test_report_is_string(self):
        stmt = EpochStatement(epoch_id="ep_1", system_id="sys", claims=[], n_records=0)
        assert isinstance(stmt.to_regulatory_report(), str)

    def test_report_contains_eu_ai_act_references(self):
        records = [AuditRecord(
            epoch_id="ep_1", model_id="m", input_hash="sha256:" + "a" * 64,
            output_hash="sha256:" + "b" * 64, sequence=i,
            confidence=0.9, latency_ms=50,
        ) for i in range(10)]
        claims_evaluated = [
            ConfidencePercentile(p=99, threshold=0.85).evaluate(records),
            ModelUnchanged().evaluate(records),
        ]
        stmt = EpochStatement(
            epoch_id="ep_1", system_id="sys",
            claims=claims_evaluated, n_records=10,
        )
        report = stmt.to_regulatory_report()
        assert "Art. 9" in report
