"""
aria.replay — Replay engine for re-evaluating past inference records.

Replays historical inference records through a new model version, comparing
outputs to detect regressions before deployment. Useful for:
- Regression testing: "does the new model produce the same answers?"
- Benchmark validation: run new model on production traffic
- Canary preparation: generate canary epoch from historical input set

Usage::

    from aria.replay import ReplayEngine

    engine = ReplayEngine(storage, model_fn=new_model.predict)
    report = engine.replay_epoch(
        source_epoch="epoch-prod-123",
        target_epoch_label="epoch-v2-replay",
        auditor=auditor,
    )
    print(report)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .auditor import InferenceAuditor
    from .quick import ARIAQuick
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)


@dataclass
class ReplayRecord:
    """Single replayed inference result."""
    original_record_id:  str
    new_output:          Any
    new_confidence:      float | None
    new_latency_ms:      float
    error:               str | None = None
    matched:             bool | None = None  # True if output matches original


@dataclass
class ReplayReport:
    """Summary of a replay run."""
    source_epoch:   str
    target_label:   str
    total:          int = 0
    succeeded:      int = 0
    failed:         int = 0
    match_rate:     float | None = None
    records:        list[ReplayRecord] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ReplayReport: source={self.source_epoch}  "
            f"total={self.total}  "
            f"ok={self.succeeded}  "
            f"err={self.failed}"
            + (f"  match={self.match_rate:.1%}" if self.match_rate is not None else "")
        )


# ---------------------------------------------------------------------------
# ReplayEngine
# ---------------------------------------------------------------------------

class ReplayEngine:
    """Replays historical inferences through a new model function.

    Args:
        storage:     StorageInterface to read historical records.
        model_fn:    Callable accepting input_data dict, returning (output, confidence).
                     If it returns a single value, confidence will be None.
        input_key:   Key in input_data to extract as the model's primary input.
    """

    def __init__(
        self,
        storage: "StorageInterface",
        model_fn: Callable[[dict], Any],
        input_key: str = "text",
    ) -> None:
        self._storage   = storage
        self._model_fn  = model_fn
        self._input_key = input_key

    def replay_epoch(
        self,
        source_epoch: str,
        target_epoch_label: str = "replay",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str = "replay-model",
        compare: bool = False,
    ) -> ReplayReport:
        """Replay all records from source_epoch through the new model.

        Args:
            source_epoch:       Epoch ID to read historical inputs from.
            target_epoch_label: Label for the replay run (used in reporting).
            auditor:            InferenceAuditor to record new inferences.
            aria:               ARIAQuick instance (alternative).
            model_id:           Model ID label for new records.
            compare:            If True, compare new output to original output.

        Returns:
            ReplayReport with statistics.
        """
        records = self._storage.list_records_by_epoch(source_epoch)
        report = ReplayReport(source_epoch=source_epoch, target_label=target_epoch_label)
        matched_count = 0

        for record in records:
            rr = self._replay_one(record, compare)
            report.records.append(rr)
            report.total += 1
            if rr.error:
                report.failed += 1
            else:
                report.succeeded += 1
                if compare and rr.matched is True:
                    matched_count += 1
                self._record(rr, record, model_id, auditor, aria)

        if compare and report.total > 0:
            report.match_rate = matched_count / report.total

        return report

    def replay_records(
        self,
        records: list[Any],
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str = "replay-model",
    ) -> ReplayReport:
        """Replay an explicit list of records."""
        report = ReplayReport(source_epoch="manual", target_label="replay")
        for record in records:
            rr = self._replay_one(record, compare=False)
            report.records.append(rr)
            report.total += 1
            if rr.error:
                report.failed += 1
            else:
                report.succeeded += 1
                self._record(rr, record, model_id, auditor, aria)
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _replay_one(self, record: Any, compare: bool) -> ReplayRecord:
        input_data = getattr(record, "input_data", {}) or {}
        record_id  = str(getattr(record, "record_id", "") or "")

        t0 = time.time()
        try:
            raw = self._model_fn(input_data)
            latency_ms = (time.time() - t0) * 1000

            if isinstance(raw, tuple) and len(raw) == 2:
                output, confidence = raw
            else:
                output, confidence = raw, None

            matched = None
            if compare:
                orig_output = getattr(record, "output_data", None)
                matched = str(output) == str(orig_output)

            return ReplayRecord(
                original_record_id=record_id,
                new_output=output,
                new_confidence=confidence,
                new_latency_ms=latency_ms,
                matched=matched,
            )
        except Exception as exc:
            return ReplayRecord(
                original_record_id=record_id,
                new_output=None,
                new_confidence=None,
                new_latency_ms=(time.time() - t0) * 1000,
                error=str(exc),
            )

    def _record(
        self,
        rr: ReplayRecord,
        original: Any,
        model_id: str,
        auditor: "InferenceAuditor | None",
        aria: "ARIAQuick | None",
    ) -> None:
        try:
            input_data  = getattr(original, "input_data", {}) or {}
            output_data = {"result": str(rr.new_output)[:1000]}
            meta        = {"replay": True, "source_record": rr.original_record_id}

            if auditor is not None:
                auditor.record(
                    model_id, input_data, output_data,
                    confidence=rr.new_confidence,
                    latency_ms=int(rr.new_latency_ms),
                    metadata=meta,
                )
            elif aria is not None:
                aria.record(
                    model_id=model_id,
                    input_data=input_data,
                    output_data=output_data,
                    confidence=rr.new_confidence,
                    latency_ms=rr.new_latency_ms,
                    metadata=meta,
                )
        except Exception as exc:
            _log.warning("ReplayEngine: record error: %s", exc)
