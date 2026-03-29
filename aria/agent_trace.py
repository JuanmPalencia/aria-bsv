"""
aria.agent_trace — Multi-step agent trace auditing for ARIA.

Records a sequence of agent steps (LLM calls, tool uses, decisions) as
individual AuditRecords linked by a shared trace_id and parent_record_id chain.

Each step is recorded through the normal ARIA audit path so every step
appears in the BSV-anchored Merkle tree.  The ``trace_id`` and
``step_sequence`` are injected into the record's ``metadata`` dict so that
an external verifier can reconstruct the full execution trace from on-chain
data alone.

Usage::

    from aria.agent_trace import AgentTraceAuditor, AgentStep

    with AgentTraceAuditor(auditor=auditor, trace_id="run-abc123") as trace:
        # Step 1: LLM call
        step1 = trace.record_step(
            model_id="planner",
            step_type="llm_call",
            input_data={"messages": [...]},
            output_data={"content": "..."},
            latency_ms=120,
        )
        # Step 2: Tool use
        step2 = trace.record_step(
            model_id="tool:search",
            step_type="tool_call",
            input_data={"query": "python docs"},
            output_data={"results": [...]},
            latency_ms=340,
            parent_record_id=step1.record_id,
        )
    # On exit: flush() called automatically
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentStep — result returned by record_step()
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """Represents a single recorded step within an agent trace.

    Attributes:
        record_id:        ARIA record ID returned by the underlying auditor.
        step_type:        Category of this step (e.g. ``"llm_call"``,
                          ``"tool_call"``, ``"decision"``).
        model_id:         Model or tool identifier used for this step.
        trace_id:         Shared identifier for the entire agent run.
        parent_record_id: record_id of the preceding step, or ``None`` if this
                          is the first step in the trace.
        sequence:         Zero-based position of this step within the trace.
    """

    record_id: str
    step_type: str
    model_id: str
    trace_id: str
    parent_record_id: str | None
    sequence: int


# ---------------------------------------------------------------------------
# AgentTraceAuditor
# ---------------------------------------------------------------------------


class AgentTraceAuditor:
    """Auditor for multi-step agent traces.

    Wraps an ``InferenceAuditor`` (or ``ARIAQuick`` instance) and records each
    agent step as an individual ARIA audit record.  All steps share a common
    ``trace_id`` that is injected into every record's ``metadata`` dict.

    Args:
        auditor:  An ``InferenceAuditor`` instance.  Mutually exclusive with
                  ``aria``.
        aria:     An ``ARIAQuick`` instance.  Mutually exclusive with
                  ``auditor``.
        trace_id: Unique identifier for this agent run.  Auto-generated
                  (``"trace_{ms_timestamp}_{4-byte hex}"``).  if not supplied.

    Raises:
        ValueError: if both *auditor* and *aria* are supplied.
    """

    def __init__(
        self,
        auditor: Any = None,
        aria: Any = None,
        trace_id: str | None = None,
    ) -> None:
        if auditor is not None and aria is not None:
            raise ValueError("Supply either 'auditor' or 'aria', not both.")

        self._auditor = auditor
        self._aria = aria
        self._trace_id: str = trace_id or (
            f"trace_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
        )
        self._step_count: int = 0
        self._steps: list[AgentStep] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trace_id(self) -> str:
        """Shared trace identifier injected into every step's metadata."""
        return self._trace_id

    @property
    def steps(self) -> list[AgentStep]:
        """All AgentStep objects recorded so far, in order of recording."""
        return list(self._steps)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_step(
        self,
        model_id: str,
        step_type: str,
        input_data: Any,
        output_data: Any,
        *,
        confidence: float | None = None,
        latency_ms: float | None = None,
        parent_record_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStep:
        """Record a single agent step and return an AgentStep descriptor.

        The step's ``trace_id``, ``step_type``, and ``step_sequence`` are
        merged into the ``metadata`` dict before the call is forwarded to the
        underlying auditor so they appear in every ARIA audit record.

        Args:
            model_id:         Model or tool identifier (must be registered in
                              the underlying auditor's ``model_hashes`` if
                              using ``InferenceAuditor``).
            step_type:        Semantic label for this step (e.g.
                              ``"llm_call"``, ``"tool_call"``).
            input_data:       Raw input to the model/tool.
            output_data:      Raw output from the model/tool.
            confidence:       Optional model confidence in [0.0, 1.0].
            latency_ms:       Wall-clock duration of this step in milliseconds.
            parent_record_id: ``record_id`` of the preceding step, enabling
                              reconstruction of the execution DAG.
            metadata:         Additional caller-supplied key-value pairs merged
                              with ARIA trace metadata.

        Returns:
            AgentStep with the assigned ``record_id`` and positional info.
        """
        sequence = self._step_count
        self._step_count += 1

        # Build merged metadata — caller dict takes lower priority so ARIA
        # trace keys are never accidentally overwritten by the caller.
        merged: dict[str, Any] = dict(metadata or {})
        merged["trace_id"] = self._trace_id
        merged["step_type"] = step_type
        merged["step_sequence"] = sequence
        if parent_record_id is not None:
            merged["parent_record_id"] = parent_record_id

        try:
            record_id = self._call_auditor(
                model_id=model_id,
                input_data=input_data,
                output_data=output_data,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata=merged,
            )
        except Exception as exc:
            _log.warning(
                "AgentTraceAuditor: record_step failed for trace=%s seq=%d: %s",
                self._trace_id,
                sequence,
                exc,
            )
            # Return a placeholder step so callers can continue the trace even
            # when the underlying auditor is unavailable.
            record_id = f"err_{self._trace_id}_{sequence:06d}"

        step = AgentStep(
            record_id=record_id,
            step_type=step_type,
            model_id=model_id,
            trace_id=self._trace_id,
            parent_record_id=parent_record_id,
            sequence=sequence,
        )
        self._steps.append(step)
        return step

    def flush(self) -> None:
        """Request an immediate epoch close on the underlying auditor.

        Calls ``auditor.flush()`` or ``aria.close()`` if the method is
        available.  Errors are logged but not re-raised.
        """
        target = self._auditor or self._aria
        if target is None:
            return
        flush_fn = getattr(target, "flush", None)
        if flush_fn is not None:
            try:
                flush_fn()
            except Exception as exc:
                _log.warning("AgentTraceAuditor.flush() error: %s", exc)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AgentTraceAuditor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_auditor(
        self,
        *,
        model_id: str,
        input_data: Any,
        output_data: Any,
        confidence: float | None,
        latency_ms: float | None,
        metadata: dict[str, Any],
    ) -> str:
        """Dispatch to either the InferenceAuditor or ARIAQuick backend.

        Returns the record_id string produced by the backend.
        """
        if self._auditor is not None:
            return self._auditor.record(
                model_id,
                input_data,
                output_data,
                confidence=confidence,
                latency_ms=int(latency_ms or 0),
                metadata=metadata,
            )
        if self._aria is not None:
            return self._aria.record(
                model_id,
                input_data,
                output_data,
                confidence=confidence,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        # No backend configured — return a deterministic placeholder.
        import hashlib
        raw = f"{model_id}:{self._trace_id}:{metadata.get('step_sequence', 0)}".encode()
        return "noop_" + hashlib.sha256(raw).hexdigest()[:16]
