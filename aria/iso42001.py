"""
aria.iso42001 — ISO 42001:2023 AI Management System compliance mapping.

ISO 42001:2023 is the international standard for AI Management Systems
(AIMS) — often described as the "ISO 9001 for AI". It defines requirements
for establishing, implementing, maintaining, and continually improving an
AI management system within the context of an organization.

This module maps ARIA audit records and epoch data to ISO 42001 controls
and generates machine-readable compliance evidence packages suitable for
audit submissions, certification bodies, or internal governance reporting.

Key clauses addressed
---------------------
Clause 6.1  Planning — Actions to address risks (risk assessment)
Clause 8.4  Operation — AI system impact assessment
Clause 9.1  Performance — Monitoring, measurement, analysis, evaluation
Clause 10.2 Improvement — Nonconformity and corrective action

Usage::

    from aria.iso42001 import ISO42001Assessor

    assessor = ISO42001Assessor(system_id="my-system", model_id="gpt-4o")
    records = [...]   # list of ARIA inference record dicts
    epochs  = [...]   # list of ARIA epoch dicts

    conformance = assessor.assess_from_records(records, epochs)
    summary     = assessor.get_conformance_summary(conformance)
    package     = assessor.generate_evidence_package(conformance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aria.core.hasher import hash_object


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ISO42001Clause(str, Enum):
    """Top-level clauses of ISO 42001:2023 that carry normative requirements."""

    CONTEXT     = "4"   # Context of the organization
    LEADERSHIP  = "5"   # Leadership
    PLANNING    = "6"   # Planning
    SUPPORT     = "7"   # Support
    OPERATION   = "8"   # Operation
    PERFORMANCE = "9"   # Performance evaluation
    IMPROVEMENT = "10"  # Improvement


class ConformanceLevel(str, Enum):
    """Degree to which an ARIA deployment satisfies an ISO 42001 control."""

    CONFORMING           = "CONFORMING"
    PARTIALLY_CONFORMING = "PARTIALLY_CONFORMING"
    NOT_CONFORMING       = "NOT_CONFORMING"
    NOT_APPLICABLE       = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ISO42001Control:
    """A single ISO 42001:2023 control definition.

    Attributes:
        control_id:          Dot-notation identifier, e.g. "6.1.2", "9.1.1".
        clause:              The top-level clause this control belongs to.
        title:               Short human-readable title.
        description:         Normative description of what the control requires.
        aria_evidence_types: ARIA artifact types that can satisfy this control,
                             e.g. ["epoch_record", "merkle_proof", "txid"].
    """

    control_id:          str
    clause:              ISO42001Clause
    title:               str
    description:         str
    aria_evidence_types: list[str]


@dataclass
class ConformanceRecord:
    """Assessment result for a single ISO 42001 control.

    The ``evidence_hash`` is computed automatically in ``__post_init__``
    using :func:`aria.core.hasher.hash_object` so that the evidence list
    is cryptographically bound to this record.

    Attributes:
        control:        The control that was assessed.
        level:          Conformance outcome.
        evidence:       List of evidence items (each item is a dict) drawn
                        from ARIA artifacts.
        notes:          Free-text assessor notes.
        assessed_at:    UTC timestamp of the assessment.
        evidence_hash:  SHA-256 hash of the evidence list, set in
                        ``__post_init__``.
    """

    control:       ISO42001Control
    level:         ConformanceLevel
    evidence:      list[dict]
    notes:         str = ""
    assessed_at:   datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    evidence_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.evidence_hash = hash_object(self.evidence)


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------

class ISO42001Assessor:
    """Maps ARIA audit data to ISO 42001:2023 controls and produces evidence.

    The assessor holds a catalogue of controls drawn from the standard and
    evaluates each one based on the presence and quality of ARIA artifacts
    (inference records, epoch records, BSV transaction IDs).

    Args:
        system_id:  Identifier of the AI system under assessment.
        model_id:   Optional model identifier for finer-grained evidence.
    """

    # ------------------------------------------------------------------
    # Control catalogue — at least 12 controls spanning all 7 clauses
    # ------------------------------------------------------------------

    CONTROLS: list[ISO42001Control] = [
        # ---- Clause 4: Context ----------------------------------------
        ISO42001Control(
            control_id="4.1",
            clause=ISO42001Clause.CONTEXT,
            title="Understanding the organization and its context",
            description=(
                "The organization shall determine external and internal issues "
                "relevant to its purpose and that affect its ability to achieve "
                "the intended outcomes of the AI management system."
            ),
            aria_evidence_types=["system_config", "epoch_record"],
        ),
        ISO42001Control(
            control_id="4.3",
            clause=ISO42001Clause.CONTEXT,
            title="Determining the scope of the AI management system",
            description=(
                "The organization shall determine the boundaries and applicability "
                "of the AI management system, including the AI systems in scope."
            ),
            aria_evidence_types=["system_config", "model_hash"],
        ),
        # ---- Clause 5: Leadership -------------------------------------
        ISO42001Control(
            control_id="5.2",
            clause=ISO42001Clause.LEADERSHIP,
            title="AI policy",
            description=(
                "Top management shall establish an AI policy that is appropriate "
                "to the purpose of the organization, includes commitments to "
                "satisfy applicable requirements, and provides a framework for "
                "setting AI objectives."
            ),
            aria_evidence_types=["system_config", "epoch_record"],
        ),
        # ---- Clause 6: Planning ---------------------------------------
        ISO42001Control(
            control_id="6.1.1",
            clause=ISO42001Clause.PLANNING,
            title="Actions to address risks and opportunities — general",
            description=(
                "When planning for the AI management system, the organization "
                "shall consider risks and opportunities to ensure the system can "
                "achieve its intended outcomes and prevent or reduce undesired effects."
            ),
            aria_evidence_types=["epoch_record", "merkle_proof"],
        ),
        ISO42001Control(
            control_id="6.1.2",
            clause=ISO42001Clause.PLANNING,
            title="AI system risk identification and treatment",
            description=(
                "The organization shall implement a process to identify, analyse, "
                "evaluate, and treat risks associated with the AI system, including "
                "risks arising from the AI system's outputs."
            ),
            aria_evidence_types=["epoch_record", "merkle_proof", "txid"],
        ),
        ISO42001Control(
            control_id="6.2",
            clause=ISO42001Clause.PLANNING,
            title="AI management system objectives",
            description=(
                "The organization shall establish AI management system objectives "
                "at relevant functions and levels, and shall plan how to achieve them."
            ),
            aria_evidence_types=["system_config", "epoch_record"],
        ),
        # ---- Clause 7: Support ----------------------------------------
        ISO42001Control(
            control_id="7.5",
            clause=ISO42001Clause.SUPPORT,
            title="Documented information",
            description=(
                "The AI management system shall include documented information "
                "required by the standard, as well as documented information "
                "determined by the organization as necessary for its effectiveness."
            ),
            aria_evidence_types=["epoch_record", "inference_record", "merkle_proof"],
        ),
        # ---- Clause 8: Operation --------------------------------------
        ISO42001Control(
            control_id="8.4",
            clause=ISO42001Clause.OPERATION,
            title="AI system impact assessment",
            description=(
                "The organization shall conduct an assessment of the impact of "
                "the AI system on individuals and groups, taking into account "
                "the context of use and potential harms."
            ),
            aria_evidence_types=["epoch_record", "txid", "merkle_proof"],
        ),
        ISO42001Control(
            control_id="8.5",
            clause=ISO42001Clause.OPERATION,
            title="AI system lifecycle",
            description=(
                "The organization shall plan, implement, control, and maintain "
                "processes for the AI system lifecycle, including development, "
                "deployment, operation, and decommissioning."
            ),
            aria_evidence_types=["epoch_record", "model_hash", "txid"],
        ),
        # ---- Clause 9: Performance evaluation -------------------------
        ISO42001Control(
            control_id="9.1.1",
            clause=ISO42001Clause.PERFORMANCE,
            title="Monitoring and measurement of AI system performance",
            description=(
                "The organization shall evaluate the AI system's performance "
                "against established metrics, including accuracy, fairness, "
                "reliability, and safety indicators."
            ),
            aria_evidence_types=["inference_record", "epoch_record", "merkle_proof"],
        ),
        ISO42001Control(
            control_id="9.1.2",
            clause=ISO42001Clause.PERFORMANCE,
            title="Logging of AI system inputs, outputs, and decisions",
            description=(
                "The organization shall maintain logs of AI system inputs, "
                "outputs, and decisions in a form that supports auditability, "
                "traceability, and post-incident analysis."
            ),
            aria_evidence_types=["inference_record", "epoch_record", "txid"],
        ),
        ISO42001Control(
            control_id="9.3",
            clause=ISO42001Clause.PERFORMANCE,
            title="Management review",
            description=(
                "Top management shall review the organization's AI management "
                "system at planned intervals to ensure its continuing suitability, "
                "adequacy, and effectiveness."
            ),
            aria_evidence_types=["epoch_record", "txid"],
        ),
        # ---- Clause 10: Improvement -----------------------------------
        ISO42001Control(
            control_id="10.2",
            clause=ISO42001Clause.IMPROVEMENT,
            title="Nonconformity identification and corrective action",
            description=(
                "When a nonconformity occurs, the organization shall take action "
                "to control and correct it, evaluate the need for action to "
                "eliminate the causes, and implement corrective actions as needed."
            ),
            aria_evidence_types=["epoch_record", "inference_record"],
        ),
        ISO42001Control(
            control_id="10.3",
            clause=ISO42001Clause.IMPROVEMENT,
            title="Continual improvement",
            description=(
                "The organization shall continually improve the suitability, "
                "adequacy, and effectiveness of the AI management system."
            ),
            aria_evidence_types=["epoch_record", "txid"],
        ),
    ]

    def __init__(
        self,
        system_id: str,
        model_id: str | None = None,
    ) -> None:
        self.system_id = system_id
        self.model_id  = model_id

    # ------------------------------------------------------------------
    # Core assessment logic
    # ------------------------------------------------------------------

    def assess_from_records(
        self,
        records: list[dict],
        epochs: list[dict],
    ) -> list[ConformanceRecord]:
        """Map ARIA records and epochs to ISO 42001 controls.

        Assessment logic
        ----------------
        - Presence of inference records satisfies Clause 9 logging controls
          (9.1.1 and 9.1.2 become CONFORMING).
        - Epochs with non-empty ``txid`` fields satisfy the impact assessment
          and on-chain anchoring controls (8.4, 6.1.2 become CONFORMING).
        - Presence of epochs (even without txids) provides partial evidence for
          planning and support controls.
        - Absence of records makes logging/monitoring controls NOT_CONFORMING.
        - Controls for which no ARIA artifact type is relevant default to
          PARTIALLY_CONFORMING when some epochs exist (evidence that a system
          is deployed and managed) or NOT_CONFORMING otherwise.

        Args:
            records: List of ARIA inference record dicts.
            epochs:  List of ARIA epoch dicts.

        Returns:
            A :class:`ConformanceRecord` for every control in ``CONTROLS``.
        """
        has_records = len(records) > 0
        has_epochs  = len(epochs) > 0
        txids       = [
            e.get("txid") or e.get("open_txid") or e.get("close_txid")
            for e in epochs
        ]
        has_txids   = any(t for t in txids if t and t != "pending")

        results: list[ConformanceRecord] = []

        for control in self.CONTROLS:
            level, evidence, notes = self._evaluate_control(
                control, records, epochs, has_records, has_epochs, has_txids
            )
            results.append(
                ConformanceRecord(
                    control=control,
                    level=level,
                    evidence=evidence,
                    notes=notes,
                )
            )

        return results

    def _evaluate_control(
        self,
        control: ISO42001Control,
        records: list[dict],
        epochs: list[dict],
        has_records: bool,
        has_epochs: bool,
        has_txids: bool,
    ) -> tuple[ConformanceLevel, list[dict], str]:
        """Return (level, evidence, notes) for a single control."""

        cid = control.control_id

        # ----------------------------------------------------------------
        # Clause 9 — Performance evaluation (logging / monitoring)
        # ----------------------------------------------------------------
        if cid == "9.1.1":
            if has_records:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "inference_record", "count": len(records)}],
                    f"{len(records)} inference record(s) provide performance monitoring evidence.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No inference records found; AI system performance cannot be verified.",
            )

        if cid == "9.1.2":
            if has_records:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "inference_record", "count": len(records)}],
                    f"{len(records)} inference record(s) demonstrate input/output logging.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No inference records found; logging obligations cannot be confirmed.",
            )

        if cid == "9.3":
            if has_epochs and has_txids:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epoch records anchored on-chain provide a management review trail.",
                )
            if has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epoch records exist but lack on-chain anchoring.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No epoch records available for management review.",
            )

        # ----------------------------------------------------------------
        # Clause 8 — Operation
        # ----------------------------------------------------------------
        if cid == "8.4":
            if has_txids:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "txid", "epochs_with_txid": sum(
                        1 for e in epochs
                        if (e.get("txid") or e.get("open_txid") or e.get("close_txid"))
                        and (e.get("txid") or e.get("open_txid") or e.get("close_txid")) != "pending"
                    )}],
                    "On-chain epoch anchoring provides an immutable impact assessment trail.",
                )
            if has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epochs exist but lack BSV transaction anchoring for full impact traceability.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No epoch data available; impact assessment trail cannot be established.",
            )

        if cid == "8.5":
            if has_epochs:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epoch lifecycle records demonstrate AI system lifecycle management.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No epoch records available to demonstrate lifecycle management.",
            )

        # ----------------------------------------------------------------
        # Clause 6 — Planning
        # ----------------------------------------------------------------
        if cid == "6.1.2":
            if has_txids:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "txid", "count": len(epochs)}],
                    "BSV-anchored epochs demonstrate risk treatment through immutable audit trail.",
                )
            if has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epochs provide partial risk evidence; on-chain anchoring recommended.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No epochs or records found; risk identification evidence is absent.",
            )

        if cid in ("6.1.1", "6.2"):
            if has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epoch records indicate active system operation; formal risk planning documentation required.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No deployment evidence found.",
            )

        # ----------------------------------------------------------------
        # Clause 10 — Improvement
        # ----------------------------------------------------------------
        if cid == "10.2":
            if has_records and has_epochs:
                return (
                    ConformanceLevel.CONFORMING,
                    [
                        {"type": "inference_record", "count": len(records)},
                        {"type": "epoch_record", "count": len(epochs)},
                    ],
                    "Inference and epoch records enable nonconformity identification.",
                )
            if has_records or has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "inference_record" if has_records else "epoch_record",
                      "count": len(records) if has_records else len(epochs)}],
                    "Partial evidence available; complete audit trail recommended.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No records or epochs available to support corrective action evidence.",
            )

        if cid == "10.3":
            if has_txids:
                return (
                    ConformanceLevel.CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Ongoing on-chain epoch generation demonstrates continual improvement.",
                )
            if has_epochs:
                return (
                    ConformanceLevel.PARTIALLY_CONFORMING,
                    [{"type": "epoch_record", "count": len(epochs)}],
                    "Epoch records exist; on-chain anchoring would strengthen continual improvement evidence.",
                )
            return (
                ConformanceLevel.NOT_CONFORMING,
                [],
                "No continual improvement evidence found.",
            )

        # ----------------------------------------------------------------
        # Clause 4, 5, 7 — Context, Leadership, Support
        # (ARIA provides partial evidence through system operation metadata)
        # ----------------------------------------------------------------
        if has_epochs or has_records:
            artifact_count = len(epochs) + len(records)
            return (
                ConformanceLevel.PARTIALLY_CONFORMING,
                [{"type": "system_operation_evidence", "artifact_count": artifact_count}],
                (
                    "ARIA provides audit evidence of system operation. Organizational "
                    "policy documents are required to fully satisfy this control."
                ),
            )

        return (
            ConformanceLevel.NOT_CONFORMING,
            [],
            "No ARIA artifacts available to support this control.",
        )

    # ------------------------------------------------------------------
    # Summary and reporting
    # ------------------------------------------------------------------

    def get_conformance_summary(
        self,
        records: list[ConformanceRecord],
    ) -> dict[str, Any]:
        """Aggregate conformance outcomes into a summary dict.

        Args:
            records: The list returned by :meth:`assess_from_records`.

        Returns:
            A dict with keys ``total``, ``conforming``, ``partially``,
            ``not_conforming``, ``not_applicable``, ``score_pct``, and
            ``clause_breakdown`` (per-clause counts).
        """
        total           = len(records)
        conforming      = sum(1 for r in records if r.level == ConformanceLevel.CONFORMING)
        partially       = sum(1 for r in records if r.level == ConformanceLevel.PARTIALLY_CONFORMING)
        not_conforming  = sum(1 for r in records if r.level == ConformanceLevel.NOT_CONFORMING)
        not_applicable  = sum(1 for r in records if r.level == ConformanceLevel.NOT_APPLICABLE)

        # Score: CONFORMING = 1.0, PARTIALLY = 0.5, NOT_CONFORMING = 0, N/A excluded
        scorable = total - not_applicable
        if scorable == 0:
            score_pct = 100.0
        else:
            raw_score = conforming + (partially * 0.5)
            score_pct = round((raw_score / scorable) * 100, 2)

        # Per-clause breakdown
        clause_breakdown: dict[str, dict[str, int]] = {}
        for cr in records:
            clause_val = cr.control.clause.value
            if clause_val not in clause_breakdown:
                clause_breakdown[clause_val] = {
                    "conforming":      0,
                    "partially":       0,
                    "not_conforming":  0,
                    "not_applicable":  0,
                }
            if cr.level == ConformanceLevel.CONFORMING:
                clause_breakdown[clause_val]["conforming"] += 1
            elif cr.level == ConformanceLevel.PARTIALLY_CONFORMING:
                clause_breakdown[clause_val]["partially"] += 1
            elif cr.level == ConformanceLevel.NOT_CONFORMING:
                clause_breakdown[clause_val]["not_conforming"] += 1
            else:
                clause_breakdown[clause_val]["not_applicable"] += 1

        return {
            "total":            total,
            "conforming":       conforming,
            "partially":        partially,
            "not_conforming":   not_conforming,
            "not_applicable":   not_applicable,
            "score_pct":        score_pct,
            "clause_breakdown": clause_breakdown,
        }

    def generate_evidence_package(
        self,
        conformance_records: list[ConformanceRecord],
    ) -> dict[str, Any]:
        """Build a complete, hashable evidence package for audit submission.

        The returned dict includes a ``evidence_package_hash`` field computed
        over the entire package payload (excluding the hash field itself) so
        that any tampering is detectable.

        Args:
            conformance_records: The list returned by :meth:`assess_from_records`.

        Returns:
            A dict suitable for JSON serialisation and archival submission.
        """
        summary = self.get_conformance_summary(conformance_records)

        controls_assessed = [
            {
                "control_id":    cr.control.control_id,
                "clause":        cr.control.clause.value,
                "title":         cr.control.title,
                "level":         cr.level.value,
                "evidence_hash": cr.evidence_hash,
                "notes":         cr.notes,
                "assessed_at":   cr.assessed_at.isoformat(),
            }
            for cr in conformance_records
        ]

        payload: dict[str, Any] = {
            "system_id":         self.system_id,
            "model_id":          self.model_id,
            "framework":         "ISO 42001:2023",
            "assessed_at":       datetime.now(timezone.utc).isoformat(),
            "overall_score_pct": summary["score_pct"],
            "summary":           summary,
            "controls_assessed": controls_assessed,
        }

        payload["evidence_package_hash"] = hash_object(
            {k: v for k, v in payload.items() if k != "evidence_package_hash"}
        )

        return payload
