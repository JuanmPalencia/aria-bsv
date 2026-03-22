"""Tests for aria.zk.claims — EU AI Act regulatory statement DSL."""

from __future__ import annotations

import pytest

from aria.core.record import AuditRecord
from aria.zk.claims import (
    AllModelsRegistered,
    ConfidencePercentile,
    LatencyBound,
    ModelUnchanged,
    NoPIIInInputs,
    OutputDistribution,
    RecordCountRange,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    model_id: str = "model-a",
    confidence: float | None = 0.9,
    latency_ms: int = 50,
    metadata: dict | None = None,
    sequence: int = 0,
) -> AuditRecord:
    return AuditRecord(
        epoch_id="ep_test_0001",
        model_id=model_id,
        input_hash="sha256:" + "a" * 64,
        output_hash="sha256:" + "b" * 64,
        sequence=sequence,
        confidence=confidence,
        latency_ms=latency_ms,
        metadata=metadata or {},
    )


def _records(n: int, **kwargs) -> list[AuditRecord]:
    return [_record(sequence=i, **kwargs) for i in range(n)]


# ---------------------------------------------------------------------------
# ConfidencePercentile
# ---------------------------------------------------------------------------

class TestConfidencePercentile:
    def test_satisfied_when_p99_above_threshold(self):
        records = [_record(confidence=0.9 + i * 0.001, sequence=i) for i in range(100)]
        claim = ConfidencePercentile(p=99, threshold=0.85)
        result = claim.evaluate(records)
        assert result.satisfied is True
        assert result.eu_ai_act_reference.startswith("Art. 9")

    def test_not_satisfied_when_p99_below_threshold(self):
        records = [_record(confidence=0.5, sequence=i) for i in range(100)]
        claim = ConfidencePercentile(p=99, threshold=0.85)
        result = claim.evaluate(records)
        assert result.satisfied is False
        assert result.detail != ""

    def test_empty_records_not_satisfied(self):
        claim = ConfidencePercentile(p=99, threshold=0.85)
        result = claim.evaluate([])
        assert result.satisfied is False
        assert "No confidence" in result.detail

    def test_records_without_confidence_ignored(self):
        records = [_record(confidence=None, sequence=i) for i in range(5)]
        claim = ConfidencePercentile(p=50, threshold=0.5)
        result = claim.evaluate(records)
        assert result.satisfied is False  # no data = not satisfied

    def test_evidence_hash_is_sha256(self):
        records = _records(10)
        result = ConfidencePercentile(p=50, threshold=0.5).evaluate(records)
        assert result.evidence_hash.startswith("sha256:")

    def test_evidence_hash_is_reproducible(self):
        records = _records(10)
        r1 = ConfidencePercentile(p=99, threshold=0.85).evaluate(records)
        r2 = ConfidencePercentile(p=99, threshold=0.85).evaluate(records)
        assert r1.evidence_hash == r2.evidence_hash

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            ConfidencePercentile(p=101, threshold=0.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ConfidencePercentile(p=99, threshold=1.5)

    def test_claim_type(self):
        assert ConfidencePercentile(p=99, threshold=0.85).claim_type == "confidence_percentile"


# ---------------------------------------------------------------------------
# ModelUnchanged
# ---------------------------------------------------------------------------

class TestModelUnchanged:
    def test_satisfied_single_model(self):
        records = _records(5, model_id="model-a")
        result = ModelUnchanged().evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_multiple_models(self):
        records = [
            _record(model_id="model-a", sequence=0),
            _record(model_id="model-b", sequence=1),
        ]
        result = ModelUnchanged().evaluate(records)
        assert result.satisfied is False
        assert "model-b" in result.detail

    def test_satisfied_expected_models_match(self):
        records = [
            _record(model_id="triage", sequence=0),
            _record(model_id="dispatch", sequence=1),
        ]
        result = ModelUnchanged(expected_model_ids={"triage", "dispatch"}).evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_unexpected_model(self):
        records = [_record(model_id="unknown-model", sequence=0)]
        result = ModelUnchanged(expected_model_ids={"triage"}).evaluate(records)
        assert result.satisfied is False
        assert "unknown-model" in result.detail

    def test_empty_records_satisfied(self):
        result = ModelUnchanged().evaluate([])
        assert result.satisfied is True

    def test_claim_type(self):
        assert ModelUnchanged().claim_type == "model_unchanged"


# ---------------------------------------------------------------------------
# NoPIIInInputs
# ---------------------------------------------------------------------------

class TestNoPIIInInputs:
    def test_satisfied_no_pii_in_metadata(self):
        records = _records(5, metadata={"incident_id": "INC001"})
        result = NoPIIInInputs(pii_fields=["patient_id", "ssn"]).evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_pii_in_metadata(self):
        records = [_record(metadata={"patient_id": "P123", "symptoms": "chest pain"}, sequence=0)]
        result = NoPIIInInputs(pii_fields=["patient_id", "ssn"]).evaluate(records)
        assert result.satisfied is False
        assert "patient_id" in result.detail

    def test_evidence_includes_input_hashes(self):
        records = _records(3)
        result = NoPIIInInputs(pii_fields=["ssn"]).evaluate(records)
        assert result.evidence_hash.startswith("sha256:")

    def test_claim_type(self):
        assert NoPIIInInputs(["ssn"]).claim_type == "no_pii_in_inputs"

    def test_eu_ai_act_reference(self):
        ref = NoPIIInInputs(["ssn"]).eu_ai_act_reference
        assert "Art. 10" in ref


# ---------------------------------------------------------------------------
# OutputDistribution
# ---------------------------------------------------------------------------

class TestOutputDistribution:
    def test_satisfied_diverse_outputs(self):
        import hashlib
        records = []
        hashes = ["sha256:" + hashlib.sha256(f"out_{i}".encode()).hexdigest() for i in range(10)]
        for i in range(10):
            r = AuditRecord(
                epoch_id="ep_test_0001",
                model_id="model-a",
                input_hash="sha256:" + "a" * 64,
                output_hash=hashes[i % len(hashes)],
                sequence=i,
            )
            records.append(r)
        result = OutputDistribution("decision", max_single_fraction=0.95).evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_single_dominant_output(self):
        import hashlib
        same_hash = "sha256:" + hashlib.sha256(b"same").hexdigest()
        diff_hash = "sha256:" + hashlib.sha256(b"diff").hexdigest()
        records = []
        for i in range(100):
            r = AuditRecord(
                epoch_id="ep_test_0001",
                model_id="model-a",
                input_hash="sha256:" + "a" * 64,
                output_hash=same_hash if i < 98 else diff_hash,
                sequence=i,
            )
            records.append(r)
        result = OutputDistribution("decision", max_single_fraction=0.90).evaluate(records)
        assert result.satisfied is False

    def test_empty_records_satisfied(self):
        result = OutputDistribution("decision").evaluate([])
        assert result.satisfied is True

    def test_claim_type(self):
        assert OutputDistribution("decision").claim_type == "output_distribution"


# ---------------------------------------------------------------------------
# LatencyBound
# ---------------------------------------------------------------------------

class TestLatencyBound:
    def test_satisfied_all_fast(self):
        records = _records(10, latency_ms=50)
        result = LatencyBound(p=99, max_ms=200).evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_slow_p99(self):
        records = [_record(latency_ms=300 if i == 9 else 50, sequence=i) for i in range(10)]
        result = LatencyBound(p=99, max_ms=200).evaluate(records)
        assert result.satisfied is False
        assert "300" in result.detail

    def test_no_latency_data_satisfied(self):
        records = _records(5, latency_ms=0)
        result = LatencyBound(p=99, max_ms=200).evaluate(records)
        assert result.satisfied is True  # no data = vacuously satisfied

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            LatencyBound(p=101, max_ms=200)

    def test_invalid_max_ms_raises(self):
        with pytest.raises(ValueError):
            LatencyBound(p=99, max_ms=0)

    def test_claim_type(self):
        assert LatencyBound(p=99, max_ms=200).claim_type == "latency_bound"


# ---------------------------------------------------------------------------
# RecordCountRange
# ---------------------------------------------------------------------------

class TestRecordCountRange:
    def test_satisfied_within_range(self):
        result = RecordCountRange(min_count=1, max_count=100).evaluate(_records(50))
        assert result.satisfied is True

    def test_not_satisfied_too_few(self):
        result = RecordCountRange(min_count=10).evaluate(_records(5))
        assert result.satisfied is False
        assert "5" in result.detail

    def test_not_satisfied_too_many(self):
        result = RecordCountRange(min_count=0, max_count=10).evaluate(_records(50))
        assert result.satisfied is False

    def test_no_max_satisfied(self):
        result = RecordCountRange(min_count=1).evaluate(_records(1_000))
        assert result.satisfied is True

    def test_claim_type(self):
        assert RecordCountRange().claim_type == "record_count_range"


# ---------------------------------------------------------------------------
# AllModelsRegistered
# ---------------------------------------------------------------------------

class TestAllModelsRegistered:
    def test_satisfied_all_registered(self):
        records = [
            _record(model_id="triage", sequence=0),
            _record(model_id="dispatch", sequence=1),
        ]
        result = AllModelsRegistered({"triage", "dispatch"}).evaluate(records)
        assert result.satisfied is True

    def test_not_satisfied_unregistered_model(self):
        records = [_record(model_id="shadow-model", sequence=0)]
        result = AllModelsRegistered({"triage"}).evaluate(records)
        assert result.satisfied is False
        assert "shadow-model" in result.detail

    def test_empty_records_satisfied(self):
        result = AllModelsRegistered({"triage"}).evaluate([])
        assert result.satisfied is True

    def test_claim_type(self):
        assert AllModelsRegistered(set()).claim_type == "all_models_registered"

    def test_eu_ai_act_reference(self):
        ref = AllModelsRegistered(set()).eu_ai_act_reference
        assert "Art. 11" in ref
