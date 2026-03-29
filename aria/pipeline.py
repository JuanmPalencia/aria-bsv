"""
aria.pipeline — Pipeline/Chain auditing for multi-model workflows.

Traces the full lifecycle of a RAG pipeline or multi-step LLM chain,
linking individual inference records with parent-child relationships.

Usage::

    from aria.pipeline import PipelineAuditor

    pipe = PipelineAuditor("my-rag-pipeline", storage)

    with pipe.trace("user-query-123") as trace:
        # Step 1: retriever
        trace.step("retriever", "bge-large", input_data, retriever_output)

        # Step 2: reranker
        trace.step("reranker", "cohere-rerank", retriever_output, reranked)

        # Step 3: generator
        trace.step("generator", "gpt-4", reranked, final_answer, confidence=0.92)

    summary = trace.summary()
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from .core.hasher import hash_object
from .core.record import AuditRecord
from .storage.base import StorageInterface

_log = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in a pipeline trace."""

    step_name: str
    model_id: str
    input_hash: str
    output_hash: str
    record_id: str
    sequence: int
    confidence: float | None = None
    latency_ms: int = 0
    started_at: float = 0
    finished_at: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "model_id": self.model_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "record_id": self.record_id,
            "sequence": self.sequence,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class PipelineTrace:
    """A complete trace of a pipeline execution."""

    trace_id: str
    pipeline_name: str
    epoch_id: str
    steps: list[PipelineStep] = field(default_factory=list)
    started_at: float = 0
    finished_at: float = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def models_used(self) -> list[str]:
        return list(dict.fromkeys(s.model_id for s in self.steps))

    @property
    def end_to_end_confidence(self) -> float | None:
        """Return confidence of the last step (final output)."""
        return self.steps[-1].confidence if self.steps else None

    def summary(self) -> str:
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"Trace ID: {self.trace_id}",
            f"Steps: {self.step_count}",
            f"Models: {', '.join(self.models_used)}",
            f"Duration: {self.total_duration_ms:.1f}ms",
        ]
        if self.end_to_end_confidence is not None:
            lines.append(f"Final confidence: {self.end_to_end_confidence:.3f}")
        lines.append("")
        for i, step in enumerate(self.steps):
            lines.append(f"  [{i}] {step.step_name} ({step.model_id}) → {step.duration_ms:.1f}ms")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "pipeline_name": self.pipeline_name,
            "epoch_id": self.epoch_id,
            "step_count": self.step_count,
            "models_used": self.models_used,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "end_to_end_confidence": self.end_to_end_confidence,
            "steps": [s.to_dict() for s in self.steps],
        }


class _TraceContext:
    """Context manager for recording pipeline steps."""

    def __init__(
        self,
        trace: PipelineTrace,
        storage: StorageInterface | None,
        start_sequence: int = 0,
    ) -> None:
        self._trace = trace
        self._storage = storage
        self._sequence = start_sequence

    def step(
        self,
        step_name: str,
        model_id: str,
        input_data: Any,
        output_data: Any,
        confidence: float | None = None,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineStep:
        """Record a single pipeline step.

        Args:
            step_name: Human-readable name (e.g. "retriever", "reranker").
            model_id:  Model identifier.
            input_data: Step input (hashed, not stored raw).
            output_data: Step output (hashed, not stored raw).
            confidence: Optional confidence score.
            latency_ms: Optional latency in milliseconds.
            metadata:  Extra metadata for this step.

        Returns:
            The recorded PipelineStep.
        """
        started = time.time()
        input_hash = hash_object(input_data)
        output_hash = hash_object(output_data)

        step_meta = {
            "pipeline": self._trace.pipeline_name,
            "trace_id": self._trace.trace_id,
            "step_name": step_name,
            "step_sequence": self._sequence,
            **(metadata or {}),
        }

        rec = AuditRecord(
            epoch_id=self._trace.epoch_id,
            model_id=model_id,
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=self._sequence,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=step_meta,
        )

        if self._storage is not None:
            self._storage.save_record(rec)

        finished = time.time()
        ps = PipelineStep(
            step_name=step_name,
            model_id=model_id,
            input_hash=input_hash,
            output_hash=output_hash,
            record_id=rec.record_id,
            sequence=self._sequence,
            confidence=confidence,
            latency_ms=latency_ms,
            started_at=started,
            finished_at=finished,
            metadata=step_meta,
        )

        self._trace.steps.append(ps)
        self._sequence += 1
        return ps

    def summary(self) -> str:
        return self._trace.summary()

    @property
    def trace(self) -> PipelineTrace:
        return self._trace


class PipelineAuditor:
    """Audits multi-model pipelines with linked trace records.

    Each ``trace()`` call creates a new pipeline trace that groups
    multiple inference steps under a single trace_id.

    Args:
        pipeline_name: Human-readable pipeline name.
        storage:       Optional storage backend for persisting records.
        epoch_id:      Epoch to record steps in (default: auto-generated).
    """

    def __init__(
        self,
        pipeline_name: str,
        storage: StorageInterface | None = None,
        epoch_id: str | None = None,
    ) -> None:
        self._pipeline_name = pipeline_name
        self._storage = storage
        self._epoch_id = epoch_id or f"pipe-{uuid.uuid4().hex[:12]}"
        self._traces: list[PipelineTrace] = []
        self._global_sequence = 0

    @property
    def traces(self) -> list[PipelineTrace]:
        return list(self._traces)

    @contextmanager
    def trace(
        self,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[_TraceContext, None, None]:
        """Start a new pipeline trace.

        Usage::

            with auditor.trace("query-123") as t:
                t.step("retriever", "bge-large", query, docs)
                t.step("generator", "gpt-4", docs, answer)

        Args:
            trace_id: Optional trace identifier (auto-generated if None).
            metadata: Extra metadata for the entire trace.
        """
        tid = trace_id or f"trace-{uuid.uuid4().hex[:8]}"
        pt = PipelineTrace(
            trace_id=tid,
            pipeline_name=self._pipeline_name,
            epoch_id=self._epoch_id,
            started_at=time.time(),
            metadata=metadata or {},
        )

        ctx = _TraceContext(pt, self._storage, self._global_sequence)
        yield ctx

        self._global_sequence = ctx._sequence
        pt.finished_at = time.time()
        self._traces.append(pt)
        _log.info(
            "Pipeline trace %s completed: %d steps, %.1fms",
            tid, pt.step_count, pt.total_duration_ms,
        )

    def all_traces_summary(self) -> str:
        """Return a summary of all recorded traces."""
        lines = [
            f"Pipeline: {self._pipeline_name}",
            f"Total traces: {len(self._traces)}",
            "",
        ]
        for trace in self._traces:
            lines.append(
                f"  {trace.trace_id}: {trace.step_count} steps, "
                f"{trace.total_duration_ms:.1f}ms, "
                f"models={trace.models_used}"
            )
        return "\n".join(lines)
