"""
portal.backend.graphql — GraphQL API layer for ARIA audit data.

Provides a GraphQL schema over ARIA's storage backend, enabling flexible
epoch and record queries for dashboards, compliance auditors, and
third-party integrations.

Schema overview::

    type EpochSummary {
        epoch_id: String!
        system_id: String!
        model_id: String
        records_count: Int!
        commitment_hash: String
        bsv_tx_open: String
        bsv_tx_close: String
        opened_at: String
        closed_at: String
    }

    type AuditRecordResult {
        record_id: String!
        epoch_id: String!
        model_id: String
        confidence: Float
        latency_ms: Int
        timestamp: String
    }

    type Query {
        epoch(epoch_id: String!): EpochSummary
        epochs(system_id: String, limit: Int, offset: Int): [EpochSummary!]!
        records(epoch_id: String!, limit: Int, offset: Int): [AuditRecordResult!]!
        record(record_id: String!): AuditRecordResult
    }

This module can operate in two modes:

1. **Standalone** — using the built-in :class:`ARIAGraphQLEngine` with a
   simple dict-based in-memory resolver (no external dependencies).
2. **Strawberry integration** — when ``strawberry-graphql`` is installed,
   :func:`make_strawberry_app` builds a full FastAPI-compatible ASGI app.

Usage (standalone)::

    from portal.backend.graphql import ARIAGraphQLEngine

    engine = ARIAGraphQLEngine()
    engine.add_epoch({"epoch_id": "ep-1", "system_id": "acme", "records_count": 100})

    result = engine.execute('{ epochs(system_id: "acme") { epoch_id records_count } }')
    print(result)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models (returned by resolvers)
# ---------------------------------------------------------------------------


@dataclass
class EpochSummary:
    epoch_id:        str
    system_id:       str
    model_id:        str = ""
    records_count:   int = 0
    commitment_hash: str = ""
    bsv_tx_open:     str = ""
    bsv_tx_close:    str = ""
    opened_at:       str = ""
    closed_at:       str = ""
    metadata:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epoch_id":        self.epoch_id,
            "system_id":       self.system_id,
            "model_id":        self.model_id,
            "records_count":   self.records_count,
            "commitment_hash": self.commitment_hash,
            "bsv_tx_open":     self.bsv_tx_open,
            "bsv_tx_close":    self.bsv_tx_close,
            "opened_at":       self.opened_at,
            "closed_at":       self.closed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpochSummary":
        return cls(
            epoch_id=d.get("epoch_id", ""),
            system_id=d.get("system_id", ""),
            model_id=d.get("model_id", ""),
            records_count=int(d.get("records_count", 0)),
            commitment_hash=d.get("commitment_hash", ""),
            bsv_tx_open=d.get("bsv_tx_open", ""),
            bsv_tx_close=d.get("bsv_tx_close", ""),
            opened_at=d.get("opened_at", ""),
            closed_at=d.get("closed_at", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class AuditRecordResult:
    record_id:   str
    epoch_id:    str
    model_id:    str   = ""
    confidence:  float = 0.0
    latency_ms:  int   = 0
    timestamp:   str   = ""
    input_hash:  str   = ""
    output_hash: str   = ""

    def to_dict(self) -> dict:
        return {
            "record_id":  self.record_id,
            "epoch_id":   self.epoch_id,
            "model_id":   self.model_id,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp":  self.timestamp,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRecordResult":
        return cls(
            record_id=d.get("record_id", ""),
            epoch_id=d.get("epoch_id", ""),
            model_id=d.get("model_id", ""),
            confidence=float(d.get("confidence", 0.0)),
            latency_ms=int(d.get("latency_ms", 0)),
            timestamp=d.get("timestamp", ""),
            input_hash=d.get("input_hash", ""),
            output_hash=d.get("output_hash", ""),
        )


# ---------------------------------------------------------------------------
# GraphQL errors
# ---------------------------------------------------------------------------


@dataclass
class GraphQLError:
    message: str
    path:    list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"message": self.message, "path": self.path}


@dataclass
class GraphQLResult:
    data:   dict | None = None
    errors: list[GraphQLError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d: dict = {}
        if self.data is not None:
            d["data"] = self.data
        if self.errors:
            d["errors"] = [e.to_dict() for e in self.errors]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# ARIAGraphQLEngine — minimal built-in resolver (no external deps)
# ---------------------------------------------------------------------------


class ARIAGraphQLEngine:
    """
    Lightweight GraphQL engine backed by in-memory dicts.

    Supports a subset of the ARIA GraphQL schema without requiring
    ``strawberry-graphql`` or ``graphene``. Parses query strings with
    a simple regex-based field extractor; not a full parser.

    For production use, integrate :func:`make_strawberry_app` instead.
    """

    def __init__(self) -> None:
        self._epochs:  dict[str, EpochSummary] = {}
        self._records: dict[str, AuditRecordResult] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_epoch(self, data: dict | EpochSummary) -> EpochSummary:
        """Add or replace an epoch."""
        ep = data if isinstance(data, EpochSummary) else EpochSummary.from_dict(data)
        self._epochs[ep.epoch_id] = ep
        return ep

    def add_record(self, data: dict | AuditRecordResult) -> AuditRecordResult:
        """Add or replace an audit record."""
        rec = data if isinstance(data, AuditRecordResult) else AuditRecordResult.from_dict(data)
        self._records[rec.record_id] = rec
        return rec

    # ------------------------------------------------------------------
    # Resolver methods (public for direct use / testing)
    # ------------------------------------------------------------------

    def resolve_epoch(self, epoch_id: str) -> EpochSummary | None:
        return self._epochs.get(epoch_id)

    def resolve_epochs(
        self,
        system_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EpochSummary]:
        items = list(self._epochs.values())
        if system_id is not None:
            items = [e for e in items if e.system_id == system_id]
        return items[offset: offset + limit]

    def resolve_records(
        self,
        epoch_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditRecordResult]:
        items = [r for r in self._records.values() if r.epoch_id == epoch_id]
        return items[offset: offset + limit]

    def resolve_record(self, record_id: str) -> AuditRecordResult | None:
        return self._records.get(record_id)

    # ------------------------------------------------------------------
    # execute — minimal query dispatcher
    # ------------------------------------------------------------------

    def execute(self, query: str, variables: dict | None = None) -> GraphQLResult:
        """
        Execute a GraphQL query string against the in-memory store.

        Supports:
        - ``{ epochs(system_id: "x", limit: N, offset: N) { fields... } }``
        - ``{ epoch(epoch_id: "x") { fields... } }``
        - ``{ records(epoch_id: "x", limit: N, offset: N) { fields... } }``
        - ``{ record(record_id: "x") { fields... } }``
        """
        try:
            return self._dispatch(query.strip(), variables or {})
        except Exception as exc:
            _log.debug("GraphQL execute error: %s", exc)
            return GraphQLResult(errors=[GraphQLError(str(exc))])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch(self, query: str, variables: dict) -> GraphQLResult:
        # Strip outer braces / query keyword
        query = re.sub(r"^(query\s+\w+\s*)?{", "", query.strip())
        query = re.sub(r"}$", "", query.strip())

        data: dict = {}

        # epochs
        m = re.search(
            r"epochs\s*(\([^)]*\))?\s*\{([^}]+)\}", query, re.DOTALL
        )
        if m:
            args = self._parse_args(m.group(1) or "", variables)
            fields = self._parse_fields(m.group(2))
            items = self.resolve_epochs(
                system_id=args.get("system_id"),
                limit=int(args.get("limit", 100)),
                offset=int(args.get("offset", 0)),
            )
            data["epochs"] = [self._project(e.to_dict(), fields) for e in items]

        # epoch (single)
        m = re.search(
            r"(?<!s)epoch\s*\(([^)]+)\)\s*\{([^}]+)\}", query, re.DOTALL
        )
        if m:
            args = self._parse_args(m.group(1), variables)
            fields = self._parse_fields(m.group(2))
            ep = self.resolve_epoch(args.get("epoch_id", ""))
            data["epoch"] = self._project(ep.to_dict(), fields) if ep else None

        # records
        m = re.search(
            r"records\s*\(([^)]+)\)\s*\{([^}]+)\}", query, re.DOTALL
        )
        if m:
            args = self._parse_args(m.group(1), variables)
            fields = self._parse_fields(m.group(2))
            items = self.resolve_records(
                epoch_id=args.get("epoch_id", ""),
                limit=int(args.get("limit", 100)),
                offset=int(args.get("offset", 0)),
            )
            data["records"] = [self._project(r.to_dict(), fields) for r in items]

        # record (single)
        m = re.search(
            r"(?<!s)record\s*\(([^)]+)\)\s*\{([^}]+)\}", query, re.DOTALL
        )
        if m:
            args = self._parse_args(m.group(1), variables)
            fields = self._parse_fields(m.group(2))
            rec = self.resolve_record(args.get("record_id", ""))
            data["record"] = self._project(rec.to_dict(), fields) if rec else None

        return GraphQLResult(data=data)

    @staticmethod
    def _parse_args(args_str: str, variables: dict) -> dict:
        """Parse key: value or key: $var pairs from GraphQL argument string."""
        result: dict = {}
        if not args_str:
            return result
        for m in re.finditer(r'(\w+)\s*:\s*("([^"]+)"|(\$\w+)|(\d+))', args_str):
            key = m.group(1)
            if m.group(3):
                result[key] = m.group(3)
            elif m.group(4):
                var_name = m.group(4)[1:]
                result[key] = variables.get(var_name, "")
            elif m.group(5):
                result[key] = int(m.group(5))
        return result

    @staticmethod
    def _parse_fields(fields_str: str) -> list[str]:
        """Extract field names from a GraphQL selection set string."""
        return [f.strip() for f in re.findall(r"\b(\w+)\b", fields_str)]

    @staticmethod
    def _project(d: dict, fields: list[str]) -> dict:
        """Return only the requested fields from dict *d*."""
        if not fields:
            return d
        return {k: v for k, v in d.items() if k in fields}


# ---------------------------------------------------------------------------
# Strawberry integration (optional)
# ---------------------------------------------------------------------------


def make_strawberry_app(engine: ARIAGraphQLEngine):  # type: ignore[return]
    """
    Build a Strawberry GraphQL ASGI app backed by *engine*.

    Requires: ``pip install strawberry-graphql[fastapi]``

    Returns a FastAPI ``GraphQLRouter`` that can be included with
    ``app.include_router(make_strawberry_app(engine), prefix="/graphql")``.
    """
    try:
        import strawberry
        from strawberry.fastapi import GraphQLRouter
    except ImportError:
        return None

    @strawberry.type
    class EpochType:
        epoch_id:        str
        system_id:       str
        model_id:        str
        records_count:   int
        commitment_hash: str
        bsv_tx_open:     str
        bsv_tx_close:    str
        opened_at:       str
        closed_at:       str

    @strawberry.type
    class RecordType:
        record_id:  str
        epoch_id:   str
        model_id:   str
        confidence: float
        latency_ms: int
        timestamp:  str

    def _to_epoch_type(e: EpochSummary) -> EpochType:
        return EpochType(**e.to_dict())

    def _to_record_type(r: AuditRecordResult) -> RecordType:
        d = r.to_dict()
        return RecordType(
            record_id=d["record_id"],
            epoch_id=d["epoch_id"],
            model_id=d["model_id"],
            confidence=d["confidence"],
            latency_ms=d["latency_ms"],
            timestamp=d["timestamp"],
        )

    @strawberry.type
    class Query:
        @strawberry.field
        def epoch(self, epoch_id: str) -> EpochType | None:
            e = engine.resolve_epoch(epoch_id)
            return _to_epoch_type(e) if e else None

        @strawberry.field
        def epochs(
            self,
            system_id: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> list[EpochType]:
            return [_to_epoch_type(e) for e in engine.resolve_epochs(system_id, limit, offset)]

        @strawberry.field
        def records(
            self,
            epoch_id: str,
            limit: int = 100,
            offset: int = 0,
        ) -> list[RecordType]:
            return [_to_record_type(r) for r in engine.resolve_records(epoch_id, limit, offset)]

        @strawberry.field
        def record(self, record_id: str) -> RecordType | None:
            r = engine.resolve_record(record_id)
            return _to_record_type(r) if r else None

    schema = strawberry.Schema(query=Query)
    return GraphQLRouter(schema)
