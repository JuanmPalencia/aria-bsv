"""
aria.regulatory_export — Export ARIA audit data in regulatory-compliant formats.

Supports:
- EU AI Act Article 12 (logging): structured JSON export
- EU AI Act Article 13 (transparency): model card export
- NIST AI RMF: risk framework alignment export
- ISO 42001: AI management system evidence package

Usage::

    from aria.regulatory_export import RegulatoryExporter, ExportFormat

    exporter = RegulatoryExporter(storage, tracker=lineage_tracker)

    # EU AI Act compliance package
    package = exporter.export(
        epoch_ids=["ep-1", "ep-2"],
        fmt=ExportFormat.EU_AI_ACT,
        system_id="my-ai-system",
    )

    with open("eu_ai_act_evidence.json", "w") as f:
        f.write(package.to_json())
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface
    from .lineage import LineageTracker

_log = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    EU_AI_ACT  = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    ISO_42001   = "iso_42001"
    GENERIC     = "generic"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EpochSummaryExport:
    """Summarised epoch data for regulatory exports."""
    epoch_id:        str
    n_records:       int
    mean_confidence: float | None
    mean_latency_ms: float | None
    opened_at:       str
    closed_at:       str
    bsv_tx_open:     str
    bsv_tx_close:    str
    model_id:        str


@dataclass
class RegulatoryPackage:
    """Complete regulatory evidence package."""
    format:          ExportFormat
    system_id:       str
    generated_at:    str
    epoch_summaries: list[EpochSummaryExport] = field(default_factory=list)
    lineage_records: list[dict] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)
    metadata:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format":           self.format.value,
            "system_id":        self.system_id,
            "generated_at":     self.generated_at,
            "epoch_summaries":  [asdict(e) for e in self.epoch_summaries],
            "lineage_records":  self.lineage_records,
            "compliance_notes": self.compliance_notes,
            "metadata":         self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __str__(self) -> str:
        return (
            f"RegulatoryPackage [{self.format.value}]: "
            f"system={self.system_id}  "
            f"epochs={len(self.epoch_summaries)}  "
            f"generated={self.generated_at}"
        )


# ---------------------------------------------------------------------------
# RegulatoryExporter
# ---------------------------------------------------------------------------

class RegulatoryExporter:
    """Exports ARIA audit data in regulatory-compliant formats.

    Args:
        storage:  StorageInterface implementation.
        tracker:  Optional LineageTracker for model version lineage.
    """

    _EU_AI_ACT_NOTES = [
        "Art. 12 — Technical documentation: epoch records captured with BSV pre-commitment",
        "Art. 13 — Transparency: model_id, input/output recorded per inference",
        "Art. 17 — Post-market monitoring: latency and confidence tracked per epoch",
        "BRC-121 — Immutable epoch anchoring on Bitcoin SV blockchain",
    ]

    _NIST_NOTES = [
        "GOVERN 1.1 — AI risk management policies documented via epoch records",
        "MAP 1.1    — Intended use and context captured in model metadata",
        "MEASURE 2.1 — Performance metrics: confidence and latency tracked",
        "MANAGE 2.2  — Model versioning via lineage tracking",
    ]

    _ISO_NOTES = [
        "ISO 42001 §6.1 — Risk identification: drift detection and alerts",
        "ISO 42001 §8.4 — AI system documentation: epoch audit trail",
        "ISO 42001 §9.1 — Performance evaluation: A/B testing and canary metrics",
        "ISO 42001 §10   — Improvement: model lineage and version tracking",
    ]

    def __init__(
        self,
        storage: "StorageInterface",
        tracker: "LineageTracker | None" = None,
    ) -> None:
        self._storage = storage
        self._tracker = tracker

    def export(
        self,
        epoch_ids: list[str],
        fmt: ExportFormat = ExportFormat.EU_AI_ACT,
        system_id: str = "",
    ) -> RegulatoryPackage:
        """Generate a regulatory evidence package.

        Args:
            epoch_ids: List of epoch IDs to include.
            fmt:       Target regulatory framework.
            system_id: AI system identifier.

        Returns:
            RegulatoryPackage ready to serialise.
        """
        summaries = [self._epoch_summary(eid) for eid in epoch_ids]
        lineage = self._collect_lineage(epoch_ids)
        notes = self._compliance_notes(fmt)

        return RegulatoryPackage(
            format=fmt,
            system_id=system_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            epoch_summaries=summaries,
            lineage_records=lineage,
            compliance_notes=notes,
            metadata={"epoch_count": len(epoch_ids), "framework": fmt.value},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _epoch_summary(self, epoch_id: str) -> EpochSummaryExport:
        records = []
        opened_at = closed_at = bsv_open = bsv_close = model_id = ""
        try:
            records = self._storage.list_records_by_epoch(epoch_id)
            epoch = self._storage.get_epoch(epoch_id) if hasattr(self._storage, "get_epoch") else None
            if epoch:
                opened_at  = str(getattr(epoch, "opened_at",  "") or "")
                closed_at  = str(getattr(epoch, "closed_at",  "") or "")
                bsv_open   = str(getattr(epoch, "tx_open",    "") or "")
                bsv_close  = str(getattr(epoch, "tx_close",   "") or "")
                model_id   = str(getattr(epoch, "model_id",   "") or "")
        except Exception as exc:
            _log.warning("RegulatoryExport: epoch %s: %s", epoch_id, exc)

        confidences = [float(getattr(r, "confidence", 0) or 0) for r in records
                       if getattr(r, "confidence", None)]
        latencies   = [float(getattr(r, "latency_ms", 0) or 0) for r in records
                       if getattr(r, "latency_ms", None)]

        mean_c = sum(confidences) / len(confidences) if confidences else None
        mean_l = sum(latencies)   / len(latencies)   if latencies   else None

        return EpochSummaryExport(
            epoch_id=epoch_id,
            n_records=len(records),
            mean_confidence=round(mean_c, 4) if mean_c else None,
            mean_latency_ms=round(mean_l, 1) if mean_l else None,
            opened_at=opened_at,
            closed_at=closed_at,
            bsv_tx_open=bsv_open,
            bsv_tx_close=bsv_close,
            model_id=model_id,
        )

    def _collect_lineage(self, epoch_ids: list[str]) -> list[dict]:
        if self._tracker is None:
            return []
        result = []
        for eid in epoch_ids:
            exported = self._tracker.export_lineage(eid)
            if exported:
                result.append(exported)
        return result

    def _compliance_notes(self, fmt: ExportFormat) -> list[str]:
        return {
            ExportFormat.EU_AI_ACT:   self._EU_AI_ACT_NOTES,
            ExportFormat.NIST_AI_RMF: self._NIST_NOTES,
            ExportFormat.ISO_42001:   self._ISO_NOTES,
            ExportFormat.GENERIC:     [],
        }.get(fmt, [])
