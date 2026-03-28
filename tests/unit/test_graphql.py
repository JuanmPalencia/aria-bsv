"""Tests for portal.backend.graphql — GraphQL API engine."""

from __future__ import annotations

import json

import pytest

from portal.backend.graphql import (
    ARIAGraphQLEngine,
    AuditRecordResult,
    EpochSummary,
    GraphQLError,
    GraphQLResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epoch(epoch_id="ep-1", system_id="acme", model_id="gpt-4", records_count=100):
    return EpochSummary(
        epoch_id=epoch_id,
        system_id=system_id,
        model_id=model_id,
        records_count=records_count,
        commitment_hash="0x" + "ab" * 32,
        bsv_tx_open="tx-open-abc",
        bsv_tx_close="tx-close-xyz",
        opened_at="2025-01-01T00:00:00",
        closed_at="2025-01-01T01:00:00",
    )


def _record(record_id="rec-1", epoch_id="ep-1", model_id="gpt-4", confidence=0.95):
    return AuditRecordResult(
        record_id=record_id,
        epoch_id=epoch_id,
        model_id=model_id,
        confidence=confidence,
        latency_ms=120,
        timestamp="2025-01-01T00:01:00",
    )


# ---------------------------------------------------------------------------
# EpochSummary
# ---------------------------------------------------------------------------

class TestEpochSummary:
    def test_to_dict(self):
        ep = _epoch()
        d = ep.to_dict()
        assert d["epoch_id"] == "ep-1"
        assert d["system_id"] == "acme"
        assert d["records_count"] == 100

    def test_from_dict(self):
        ep = _epoch()
        ep2 = EpochSummary.from_dict(ep.to_dict())
        assert ep2.epoch_id == ep.epoch_id
        assert ep2.system_id == ep.system_id

    def test_roundtrip(self):
        ep = _epoch()
        ep2 = EpochSummary.from_dict(ep.to_dict())
        assert ep2.to_dict() == ep.to_dict()

    def test_from_dict_defaults(self):
        ep = EpochSummary.from_dict({"epoch_id": "e", "system_id": "s"})
        assert ep.records_count == 0
        assert ep.model_id == ""

    def test_from_dict_unknown_keys_ignored(self):
        ep = EpochSummary.from_dict({"epoch_id": "e", "system_id": "s", "future_field": "x"})
        assert ep.epoch_id == "e"


# ---------------------------------------------------------------------------
# AuditRecordResult
# ---------------------------------------------------------------------------

class TestAuditRecordResult:
    def test_to_dict(self):
        r = _record()
        d = r.to_dict()
        assert d["record_id"] == "rec-1"
        assert d["confidence"] == 0.95
        assert d["latency_ms"] == 120

    def test_from_dict(self):
        r = _record()
        r2 = AuditRecordResult.from_dict(r.to_dict())
        assert r2.record_id == r.record_id
        assert r2.confidence == r.confidence

    def test_roundtrip(self):
        r = _record()
        assert AuditRecordResult.from_dict(r.to_dict()).to_dict() == r.to_dict()

    def test_confidence_coerced_float(self):
        r = AuditRecordResult.from_dict({"record_id": "r", "epoch_id": "e", "confidence": "0.75"})
        assert r.confidence == 0.75

    def test_latency_coerced_int(self):
        r = AuditRecordResult.from_dict({"record_id": "r", "epoch_id": "e", "latency_ms": "200"})
        assert r.latency_ms == 200


# ---------------------------------------------------------------------------
# GraphQLResult
# ---------------------------------------------------------------------------

class TestGraphQLResult:
    def test_ok_when_no_errors(self):
        r = GraphQLResult(data={"epochs": []})
        assert r.ok is True

    def test_not_ok_when_errors(self):
        r = GraphQLResult(errors=[GraphQLError("fail")])
        assert r.ok is False

    def test_to_dict_with_data(self):
        r = GraphQLResult(data={"epoch": None})
        d = r.to_dict()
        assert "data" in d
        assert "errors" not in d

    def test_to_dict_with_errors(self):
        r = GraphQLResult(errors=[GraphQLError("oops")])
        d = r.to_dict()
        assert "errors" in d
        assert d["errors"][0]["message"] == "oops"

    def test_to_json(self):
        r = GraphQLResult(data={"x": 1})
        j = r.to_json()
        assert json.loads(j)["data"]["x"] == 1


# ---------------------------------------------------------------------------
# ARIAGraphQLEngine — data ingestion
# ---------------------------------------------------------------------------

class TestARIAGraphQLEngineIngestion:
    def test_add_epoch_stores(self):
        engine = ARIAGraphQLEngine()
        ep = engine.add_epoch(_epoch())
        assert engine.resolve_epoch("ep-1") is ep

    def test_add_epoch_from_dict(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch({"epoch_id": "ep-2", "system_id": "test"})
        assert engine.resolve_epoch("ep-2") is not None

    def test_add_record_stores(self):
        engine = ARIAGraphQLEngine()
        rec = engine.add_record(_record())
        assert engine.resolve_record("rec-1") is rec

    def test_add_record_from_dict(self):
        engine = ARIAGraphQLEngine()
        engine.add_record({"record_id": "r2", "epoch_id": "ep-1"})
        assert engine.resolve_record("r2") is not None

    def test_replace_epoch(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch(records_count=10))
        engine.add_epoch(_epoch(records_count=20))
        assert engine.resolve_epoch("ep-1").records_count == 20


# ---------------------------------------------------------------------------
# ARIAGraphQLEngine — resolvers
# ---------------------------------------------------------------------------

class TestARIAGraphQLEngineResolvers:
    def test_resolve_epoch_none(self):
        engine = ARIAGraphQLEngine()
        assert engine.resolve_epoch("ghost") is None

    def test_resolve_epochs_all(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1", system_id="a"))
        engine.add_epoch(_epoch("ep-2", system_id="b"))
        assert len(engine.resolve_epochs()) == 2

    def test_resolve_epochs_filter_system(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1", system_id="acme"))
        engine.add_epoch(_epoch("ep-2", system_id="globex"))
        result = engine.resolve_epochs(system_id="acme")
        assert len(result) == 1
        assert result[0].system_id == "acme"

    def test_resolve_epochs_limit(self):
        engine = ARIAGraphQLEngine()
        for i in range(10):
            engine.add_epoch(_epoch(f"ep-{i}"))
        result = engine.resolve_epochs(limit=3)
        assert len(result) == 3

    def test_resolve_epochs_offset(self):
        engine = ARIAGraphQLEngine()
        for i in range(5):
            engine.add_epoch(_epoch(f"ep-{i}"))
        result = engine.resolve_epochs(offset=3)
        assert len(result) == 2

    def test_resolve_records_filter_by_epoch(self):
        engine = ARIAGraphQLEngine()
        engine.add_record(_record("r1", epoch_id="ep-1"))
        engine.add_record(_record("r2", epoch_id="ep-2"))
        result = engine.resolve_records("ep-1")
        assert len(result) == 1
        assert result[0].epoch_id == "ep-1"

    def test_resolve_records_limit(self):
        engine = ARIAGraphQLEngine()
        for i in range(10):
            engine.add_record(_record(f"r-{i}", epoch_id="ep-1"))
        result = engine.resolve_records("ep-1", limit=4)
        assert len(result) == 4

    def test_resolve_record_none(self):
        engine = ARIAGraphQLEngine()
        assert engine.resolve_record("ghost") is None


# ---------------------------------------------------------------------------
# ARIAGraphQLEngine — execute (query parsing)
# ---------------------------------------------------------------------------

class TestARIAGraphQLEngineExecute:
    def test_execute_epochs_all(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1"))
        engine.add_epoch(_epoch("ep-2"))
        result = engine.execute('{ epochs { epoch_id system_id } }')
        assert result.ok
        assert len(result.data["epochs"]) == 2

    def test_execute_epochs_filter_system(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1", system_id="acme"))
        engine.add_epoch(_epoch("ep-2", system_id="globex"))
        result = engine.execute('{ epochs(system_id: "acme") { epoch_id } }')
        assert result.ok
        epochs = result.data["epochs"]
        assert len(epochs) == 1
        assert epochs[0]["epoch_id"] == "ep-1"

    def test_execute_epoch_single(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1"))
        result = engine.execute('{ epoch(epoch_id: "ep-1") { epoch_id records_count } }')
        assert result.ok
        ep = result.data["epoch"]
        assert ep is not None
        assert ep["epoch_id"] == "ep-1"
        assert ep["records_count"] == 100

    def test_execute_epoch_not_found(self):
        engine = ARIAGraphQLEngine()
        result = engine.execute('{ epoch(epoch_id: "ghost") { epoch_id } }')
        assert result.ok
        assert result.data["epoch"] is None

    def test_execute_records(self):
        engine = ARIAGraphQLEngine()
        engine.add_record(_record("r1", epoch_id="ep-1"))
        engine.add_record(_record("r2", epoch_id="ep-1"))
        result = engine.execute('{ records(epoch_id: "ep-1") { record_id confidence } }')
        assert result.ok
        assert len(result.data["records"]) == 2

    def test_execute_record_single(self):
        engine = ARIAGraphQLEngine()
        engine.add_record(_record("r1", confidence=0.77))
        result = engine.execute('{ record(record_id: "r1") { record_id confidence } }')
        assert result.ok
        rec = result.data["record"]
        assert rec["confidence"] == 0.77

    def test_execute_record_not_found(self):
        engine = ARIAGraphQLEngine()
        result = engine.execute('{ record(record_id: "ghost") { record_id } }')
        assert result.ok
        assert result.data["record"] is None

    def test_execute_projection(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1"))
        result = engine.execute('{ epochs { epoch_id } }')
        assert result.ok
        ep = result.data["epochs"][0]
        assert "epoch_id" in ep
        assert "system_id" not in ep

    def test_execute_limit_in_query(self):
        engine = ARIAGraphQLEngine()
        for i in range(10):
            engine.add_epoch(_epoch(f"ep-{i}"))
        result = engine.execute('{ epochs(limit: 3) { epoch_id } }')
        assert result.ok
        assert len(result.data["epochs"]) == 3

    def test_execute_variable_substitution(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-var", system_id="test"))
        result = engine.execute(
            'query GetEpoch { epoch(epoch_id: $id) { epoch_id } }',
            variables={"id": "ep-var"},
        )
        assert result.ok
        assert result.data["epoch"]["epoch_id"] == "ep-var"

    def test_execute_invalid_query_returns_error(self):
        engine = ARIAGraphQLEngine()
        result = engine.execute(None)  # type: ignore[arg-type]
        assert not result.ok
        assert len(result.errors) > 0

    def test_execute_empty_engine(self):
        engine = ARIAGraphQLEngine()
        result = engine.execute('{ epochs { epoch_id } }')
        assert result.ok
        assert result.data["epochs"] == []

    def test_execute_records_limit(self):
        engine = ARIAGraphQLEngine()
        for i in range(5):
            engine.add_record(_record(f"r-{i}", epoch_id="ep-1"))
        result = engine.execute('{ records(epoch_id: "ep-1", limit: 2) { record_id } }')
        assert result.ok
        assert len(result.data["records"]) == 2

    def test_execute_confidence_in_record(self):
        engine = ARIAGraphQLEngine()
        engine.add_record(_record("r1", confidence=0.88))
        result = engine.execute('{ records(epoch_id: "ep-1") { confidence } }')
        assert result.ok
        assert result.data["records"][0]["confidence"] == 0.88

    def test_execute_model_id_field(self):
        engine = ARIAGraphQLEngine()
        engine.add_epoch(_epoch("ep-1", model_id="claude-3"))
        result = engine.execute('{ epochs { epoch_id model_id } }')
        assert result.ok
        assert result.data["epochs"][0]["model_id"] == "claude-3"


# ---------------------------------------------------------------------------
# GraphQLError
# ---------------------------------------------------------------------------

class TestGraphQLError:
    def test_to_dict(self):
        e = GraphQLError(message="bad query", path=["epochs", 0])
        d = e.to_dict()
        assert d["message"] == "bad query"
        assert d["path"] == ["epochs", 0]

    def test_default_path_empty(self):
        e = GraphQLError("x")
        assert e.path == []
