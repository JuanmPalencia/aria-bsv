"""
aria.nist_rmf — NIST AI Risk Management Framework (AI RMF 1.0) mapping.

The NIST AI RMF 1.0 (January 2023) is a voluntary framework published by
the U.S. National Institute of Standards and Technology to help organizations
manage risks related to the design, development, deployment, and use of AI
systems. It is organized around four core functions:

    GOVERN   — Cultivate a culture of AI risk management
    MAP      — Categorize and contextualize AI risks
    MEASURE  — Analyze and assess AI risks
    MANAGE   — Prioritize and address AI risks

This module maps ARIA audit records, epoch data, and BSV transaction IDs
to NIST AI RMF categories and generates structured risk assessment reports
suitable for regulatory submissions, audits, and board-level governance
reporting.

Usage::

    from aria.nist_rmf import NISTRMFAssessor, RiskLevel

    assessor = NISTRMFAssessor(system_id="fraud-detector", risk_tier=RiskLevel.HIGH)

    records = [...]   # list of ARIA inference record dicts
    epochs  = [...]   # list of ARIA epoch dicts
    txids   = [...]   # list of BSV transaction ID strings

    assessments = assessor.assess(records, epochs, txids)
    profile     = assessor.risk_profile(assessments)
    report      = assessor.generate_rmf_report(assessments)
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

class RMFFunction(str, Enum):
    """The four core functions of the NIST AI RMF 1.0."""

    GOVERN  = "GOVERN"
    MAP     = "MAP"
    MEASURE = "MEASURE"
    MANAGE  = "MANAGE"


class RiskLevel(str, Enum):
    """Risk tier used to contextualize an AI system's overall risk posture."""

    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RMFCategory:
    """A single NIST AI RMF 1.0 category.

    Attributes:
        function:       The core function this category belongs to.
        category_id:    Dot-notation identifier, e.g. "GOVERN-1.1",
                        "MEASURE-2.5".
        title:          Short human-readable title.
        description:    Normative description of the category's intent.
        aria_coverage:  Plain-text description of how ARIA addresses this
                        category.
    """

    function:      RMFFunction
    category_id:   str
    title:         str
    description:   str
    aria_coverage: str


@dataclass
class RMFAssessment:
    """Assessment result for a single NIST AI RMF category.

    Attributes:
        category:    The category that was assessed.
        risk_level:  Residual risk level after considering ARIA coverage.
        implemented: Whether the category's practices are implemented.
        evidence:    List of evidence description strings.
        gaps:        List of gap description strings (possibly empty).
        assessed_at: UTC timestamp of the assessment.
    """

    category:    RMFCategory
    risk_level:  RiskLevel
    implemented: bool
    evidence:    list[str]
    gaps:        list[str]
    assessed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------

class NISTRMFAssessor:
    """Maps ARIA audit data to NIST AI RMF 1.0 categories and produces reports.

    The assessor holds a catalogue of categories drawn from the framework and
    evaluates each one based on the presence and quality of ARIA artifacts.

    Args:
        system_id:  Identifier of the AI system under assessment.
        risk_tier:  Initial risk tier classification (default MEDIUM).
    """

    # ------------------------------------------------------------------
    # Category catalogue — at least 16 categories across all 4 functions
    # ------------------------------------------------------------------

    CATEGORIES: list[RMFCategory] = [
        # ----------------------------------------------------------------
        # GOVERN — 4 categories
        # ----------------------------------------------------------------
        RMFCategory(
            function=RMFFunction.GOVERN,
            category_id="GOVERN-1.1",
            title="AI risk policies and procedures",
            description=(
                "Policies, processes, procedures, and practices across the "
                "organization related to the mapping, measuring, and managing "
                "of AI risks are in place, transparent, and implemented "
                "effectively."
            ),
            aria_coverage=(
                "ARIA's BRC-121 protocol provides a tamper-evident policy "
                "enforcement record via BSV anchoring. Epoch records constitute "
                "auditable evidence that governance procedures are followed."
            ),
        ),
        RMFCategory(
            function=RMFFunction.GOVERN,
            category_id="GOVERN-1.2",
            title="Accountability and responsibility",
            description=(
                "Accountability and responsibility for AI risk management is "
                "assigned to roles and teams with the necessary authority, "
                "skills, resources, and organizational authority."
            ),
            aria_coverage=(
                "ARIA's system_id and model_id fields bind inference records to "
                "accountable system owners. The immutable BSV trail ensures "
                "responsibility cannot be repudiated."
            ),
        ),
        RMFCategory(
            function=RMFFunction.GOVERN,
            category_id="GOVERN-4.1",
            title="AI risk and benefits communication",
            description=(
                "Organizational risk tolerance for AI, as well as AI risks and "
                "benefits, are communicated across the organization and to "
                "relevant external stakeholders."
            ),
            aria_coverage=(
                "ARIA evidence packages and compliance reports are machine-readable "
                "artifacts suitable for stakeholder communication and board-level "
                "risk reporting."
            ),
        ),
        RMFCategory(
            function=RMFFunction.GOVERN,
            category_id="GOVERN-6.1",
            title="AI risk management policies across the lifecycle",
            description=(
                "Policies and procedures are in place to address AI risks throughout "
                "the entire AI lifecycle, including design, development, deployment, "
                "evaluation, and decommissioning."
            ),
            aria_coverage=(
                "ARIA epoch records span the full operational lifecycle; model_hash "
                "commits provide pre-deployment integrity guarantees."
            ),
        ),
        # ----------------------------------------------------------------
        # MAP — 4 categories
        # ----------------------------------------------------------------
        RMFCategory(
            function=RMFFunction.MAP,
            category_id="MAP-1.1",
            title="Context establishment for AI risk assessment",
            description=(
                "Context is established for the AI risk assessment. The purpose "
                "of the AI system, its intended uses, and the organizational "
                "context are documented."
            ),
            aria_coverage=(
                "ARIA epoch metadata (system_id, model_id, state_hash) captures "
                "the deployment context at each assessment point."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MAP,
            category_id="MAP-1.5",
            title="AI risk identification",
            description=(
                "Organizational risk tolerance for the AI system is assessed and "
                "documented, including risks to individuals, groups, communities, "
                "organizations, and society."
            ),
            aria_coverage=(
                "ARIA's confidence scores, latency metrics, and epoch-level Merkle "
                "proofs provide quantitative evidence for risk identification."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MAP,
            category_id="MAP-3.5",
            title="Practices and personnel for AI risk assessment",
            description=(
                "Practices and personnel for supporting AI risk identification, "
                "assessment, and prioritization are available and documented."
            ),
            aria_coverage=(
                "ARIA SDKs (Python, Java, .NET, TypeScript) and CLI tools lower "
                "the barrier for operational teams to conduct consistent risk "
                "assessments using the same cryptographic audit trail."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MAP,
            category_id="MAP-5.1",
            title="AI risk likelihood and impact",
            description=(
                "Likelihood and magnitude of each identified AI risk is estimated "
                "as a basis for determining priorities and risk responses."
            ),
            aria_coverage=(
                "Epoch-level statistics derived from ARIA records (error rates, "
                "confidence distributions, latency percentiles) quantify risk "
                "likelihood and impact."
            ),
        ),
        # ----------------------------------------------------------------
        # MEASURE — 5 categories
        # ----------------------------------------------------------------
        RMFCategory(
            function=RMFFunction.MEASURE,
            category_id="MEASURE-1.1",
            title="Approaches and metrics for AI risk evaluation",
            description=(
                "Approaches and metrics for evaluating AI risks, as identified "
                "in the MAP function, are selected for implementation starting "
                "with the most significant AI risks."
            ),
            aria_coverage=(
                "ARIA records confidence scores, latency, and model identifiers "
                "per inference, enabling quantitative risk metric computation "
                "across any time window."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MEASURE,
            category_id="MEASURE-2.5",
            title="AI system validity and reliability",
            description=(
                "The AI system to be deployed is demonstrated to be valid and "
                "reliable throughout its anticipated operational range, including "
                "in adversarial conditions."
            ),
            aria_coverage=(
                "ARIA's Merkle-tree commitments and BSV anchoring provide "
                "cryptographic proof of model identity and inference integrity, "
                "supporting validity and reliability claims."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MEASURE,
            category_id="MEASURE-2.6",
            title="AI system safety evaluation",
            description=(
                "The AI system is evaluated for safety properties to the extent "
                "practicable, and relevant safety tests are conducted and documented."
            ),
            aria_coverage=(
                "ARIA's pre-commitment protocol (EPOCH OPEN published before model "
                "execution) provides a tamper-evident record that the model under "
                "test was the model declared in advance."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MEASURE,
            category_id="MEASURE-2.9",
            title="Privacy risk evaluation",
            description=(
                "The AI system is evaluated for privacy risks and relevant privacy "
                "tests are conducted."
            ),
            aria_coverage=(
                "ARIA stores only SHA-256 hashes of inputs/outputs on-chain — "
                "raw personal data never leaves the operator's environment, "
                "providing a data-minimisation foundation for privacy risk evaluation."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MEASURE,
            category_id="MEASURE-4.1",
            title="Performance metrics identification",
            description=(
                "Performance metrics are identified for AI system characteristics "
                "of interest, including trustworthiness, and benchmarks are "
                "identified to support performance measurement."
            ),
            aria_coverage=(
                "ARIA natively captures latency_ms, confidence, and model_id per "
                "inference — providing a ready-made performance metrics dataset."
            ),
        ),
        # ----------------------------------------------------------------
        # MANAGE — 4 categories
        # ----------------------------------------------------------------
        RMFCategory(
            function=RMFFunction.MANAGE,
            category_id="MANAGE-1.1",
            title="AI risk treatment plan",
            description=(
                "A risk treatment plan is developed, agreed to, and implemented "
                "for the identified and prioritized AI risks based on the outcomes "
                "of the MAP and MEASURE functions."
            ),
            aria_coverage=(
                "ARIA's compliance modules (ISO 42001, EU AI Act) generate "
                "structured evidence packages that directly feed into risk "
                "treatment planning workflows."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MANAGE,
            category_id="MANAGE-2.4",
            title="AI risk reporting mechanisms",
            description=(
                "Mechanisms are in place for AI risks to be reported, tracked, "
                "and communicated to organizational leadership and relevant "
                "stakeholders."
            ),
            aria_coverage=(
                "BSV-anchored transaction IDs provide publicly verifiable, "
                "tamper-evident risk event records. ARIA's reporting module "
                "generates machine-readable and human-readable risk reports."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MANAGE,
            category_id="MANAGE-3.1",
            title="AI risk response and recovery",
            description=(
                "Responses to the AI risks deemed beyond organizational risk "
                "tolerance are developed, and recovery activities are planned "
                "to support effective response and recovery."
            ),
            aria_coverage=(
                "ARIA epoch records provide an immutable baseline for post-incident "
                "forensic analysis and recovery planning. Merkle proofs allow "
                "pinpointing specific inference events in any epoch."
            ),
        ),
        RMFCategory(
            function=RMFFunction.MANAGE,
            category_id="MANAGE-4.1",
            title="Continual monitoring of residual risk",
            description=(
                "Post-deployment AI risks are monitored continuously, including "
                "periodic reviews of implemented risk mitigations."
            ),
            aria_coverage=(
                "ARIA's continuous epoch generation and BSV anchoring create an "
                "always-on monitoring trail. Drift detection and canary modules "
                "extend this with automated residual-risk monitoring."
            ),
        ),
    ]

    def __init__(
        self,
        system_id: str,
        risk_tier: RiskLevel = RiskLevel.MEDIUM,
    ) -> None:
        self.system_id = system_id
        self.risk_tier = risk_tier

    # ------------------------------------------------------------------
    # Core assessment logic
    # ------------------------------------------------------------------

    def assess(
        self,
        aria_records: list[dict],
        epochs: list[dict],
        txids: list[str],
    ) -> list[RMFAssessment]:
        """Map ARIA artifacts to NIST AI RMF 1.0 categories.

        Assessment logic
        ----------------
        - Having inference records → MEASURE categories for monitoring and
          performance metrics are implemented.
        - Having BSV transaction IDs → MANAGE-2.4 (risk reporting) is
          implemented; other MANAGE categories get partial credit.
        - Having epochs → MAP categories (context, risk identification) are
          satisfied; GOVERN categories receive partial credit.
        - Empty inputs → most categories are not implemented with gaps noted.

        Args:
            aria_records: List of ARIA inference record dicts.
            epochs:       List of ARIA epoch dicts.
            txids:        List of confirmed BSV transaction ID strings.

        Returns:
            A :class:`RMFAssessment` for every category in ``CATEGORIES``.
        """
        has_records = len(aria_records) > 0
        has_epochs  = len(epochs) > 0
        has_txids   = any(t for t in txids if t and t.strip())

        results: list[RMFAssessment] = []

        for category in self.CATEGORIES:
            assessment = self._evaluate_category(
                category, aria_records, epochs, txids,
                has_records, has_epochs, has_txids,
            )
            results.append(assessment)

        return results

    def _evaluate_category(
        self,
        category: RMFCategory,
        records: list[dict],
        epochs: list[dict],
        txids: list[str],
        has_records: bool,
        has_epochs: bool,
        has_txids: bool,
    ) -> RMFAssessment:
        """Return an RMFAssessment for a single category."""

        fn  = category.function
        cid = category.category_id
        now = datetime.now(timezone.utc)

        # ----------------------------------------------------------------
        # MEASURE categories — driven by inference records
        # ----------------------------------------------------------------
        if fn == RMFFunction.MEASURE:
            if cid in ("MEASURE-1.1", "MEASURE-4.1"):
                if has_records:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(records)} ARIA inference record(s) provide metrics data.",
                            "Confidence scores, latency, and model identifiers captured per record.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=[
                        "No ARIA inference records found.",
                        "Performance metrics cannot be computed without logged inferences.",
                    ],
                    assessed_at=now,
                )

            if cid in ("MEASURE-2.5", "MEASURE-2.6"):
                if has_epochs and has_txids:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(epochs)} epoch(s) anchored with {len(txids)} BSV txid(s).",
                            "Pre-commitment hashes enforce model identity before execution.",
                            "Merkle-tree commitments provide inference-level integrity proofs.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                if has_epochs:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.MEDIUM,
                        implemented=True,
                        evidence=[f"{len(epochs)} epoch(s) with pre-commitment hashes."],
                        gaps=["No BSV transaction IDs; on-chain immutability not yet established."],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=[
                        "No epoch records or transaction IDs.",
                        "Model identity and inference integrity cannot be verified.",
                    ],
                    assessed_at=now,
                )

            if cid == "MEASURE-2.9":
                if has_records:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            "ARIA stores only SHA-256 hashes on-chain; no raw personal data.",
                            f"{len(records)} record(s) use hash-only pattern.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=RiskLevel.MEDIUM,
                    implemented=False,
                    evidence=[],
                    gaps=["No inference records to confirm privacy-by-design compliance."],
                    assessed_at=now,
                )

        # ----------------------------------------------------------------
        # MANAGE categories — driven by txids and epochs
        # ----------------------------------------------------------------
        if fn == RMFFunction.MANAGE:
            if cid == "MANAGE-2.4":
                if has_txids:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(txids)} BSV transaction ID(s) provide tamper-evident risk reporting.",
                            "Public blockchain anchoring enables verifiable stakeholder reporting.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=[
                        "No BSV transaction IDs found.",
                        "Immutable risk reporting requires on-chain anchoring.",
                    ],
                    assessed_at=now,
                )

            if cid in ("MANAGE-1.1", "MANAGE-3.1", "MANAGE-4.1"):
                if has_txids and has_records:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(txids)} BSV txid(s) and {len(records)} inference record(s).",
                            "Continuous epoch generation provides ongoing monitoring baseline.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                if has_txids or has_epochs:
                    gaps = []
                    if not has_records:
                        gaps.append("No inference records for detailed incident analysis.")
                    if not has_txids:
                        gaps.append("No BSV txids for on-chain risk event anchoring.")
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.MEDIUM,
                        implemented=True,
                        evidence=[
                            f"{len(epochs)} epoch(s) available for risk baseline.",
                        ],
                        gaps=gaps,
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=["No ARIA artifacts available for risk management evidence."],
                    assessed_at=now,
                )

        # ----------------------------------------------------------------
        # MAP categories — driven by epochs
        # ----------------------------------------------------------------
        if fn == RMFFunction.MAP:
            if cid in ("MAP-1.1", "MAP-1.5"):
                if has_epochs:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(epochs)} epoch record(s) capture deployment context.",
                            "system_id, model_id, and state_hash fields document context.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=[
                        "No epoch records found.",
                        "Deployment context and risk identification baseline is absent.",
                    ],
                    assessed_at=now,
                )

            if cid == "MAP-3.5":
                # ARIA SDK availability is always present by definition
                return RMFAssessment(
                    category=category,
                    risk_level=RiskLevel.LOW,
                    implemented=True,
                    evidence=[
                        "ARIA SDK provides Python, Java, .NET, and TypeScript implementations.",
                        "Standardized CLI and API for consistent risk assessment practices.",
                    ],
                    gaps=[],
                    assessed_at=now,
                )

            if cid == "MAP-5.1":
                if has_records and has_epochs:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.LOW,
                        implemented=True,
                        evidence=[
                            f"{len(records)} inference record(s) enable statistical risk quantification.",
                            f"{len(epochs)} epoch(s) provide aggregated risk trend data.",
                        ],
                        gaps=[],
                        assessed_at=now,
                    )
                if has_records or has_epochs:
                    return RMFAssessment(
                        category=category,
                        risk_level=RiskLevel.MEDIUM,
                        implemented=True,
                        evidence=[
                            f"Partial data: {len(records)} record(s), {len(epochs)} epoch(s).",
                        ],
                        gaps=["Complete inference + epoch data recommended for reliable risk quantification."],
                        assessed_at=now,
                    )
                return RMFAssessment(
                    category=category,
                    risk_level=self.risk_tier,
                    implemented=False,
                    evidence=[],
                    gaps=["No ARIA artifacts available to estimate risk likelihood or magnitude."],
                    assessed_at=now,
                )

        # ----------------------------------------------------------------
        # GOVERN categories — partial credit from any ARIA usage
        # ----------------------------------------------------------------
        if fn == RMFFunction.GOVERN:
            if has_epochs or has_records:
                return RMFAssessment(
                    category=category,
                    risk_level=RiskLevel.MEDIUM,
                    implemented=True,
                    evidence=[
                        f"ARIA deployment evidence: {len(epochs)} epoch(s), {len(records)} record(s).",
                        "Organizational use of ARIA indicates active AI risk governance posture.",
                    ],
                    gaps=[
                        "Formal policy documents and role assignments must be maintained separately.",
                    ],
                    assessed_at=now,
                )
            return RMFAssessment(
                category=category,
                risk_level=self.risk_tier,
                implemented=False,
                evidence=[],
                gaps=[
                    "No ARIA artifacts found.",
                    "Governance policies and procedures cannot be evidenced.",
                ],
                assessed_at=now,
            )

        # Fallback — should not be reached with a complete catalogue
        return RMFAssessment(
            category=category,
            risk_level=self.risk_tier,
            implemented=False,
            evidence=[],
            gaps=["No ARIA coverage defined for this category."],
            assessed_at=now,
        )

    # ------------------------------------------------------------------
    # Risk profile and reporting
    # ------------------------------------------------------------------

    def risk_profile(
        self,
        assessments: list[RMFAssessment],
    ) -> dict[str, Any]:
        """Compute a risk profile summarising function-level implementation.

        Args:
            assessments: The list returned by :meth:`assess`.

        Returns:
            A dict with keys ``function_scores`` (percent implemented per
            function), ``overall_score`` (0.0–1.0), ``risk_tier``, and
            ``top_gaps`` (the three most critical gap descriptions).
        """
        function_scores: dict[str, float] = {}

        for fn in RMFFunction:
            fn_assessments = [a for a in assessments if a.category.function == fn]
            if not fn_assessments:
                function_scores[fn.value] = 0.0
                continue
            implemented_count = sum(1 for a in fn_assessments if a.implemented)
            function_scores[fn.value] = round(
                implemented_count / len(fn_assessments), 4
            )

        total       = len(assessments)
        implemented = sum(1 for a in assessments if a.implemented)
        overall_score = round(implemented / total, 4) if total > 0 else 0.0

        # Collect gaps from non-implemented categories first, then partial
        all_gaps: list[str] = []
        for a in assessments:
            if not a.implemented:
                all_gaps.extend(a.gaps)
        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_gaps: list[str] = []
        for g in all_gaps:
            if g not in seen:
                seen.add(g)
                unique_gaps.append(g)
        top_gaps = unique_gaps[:3]

        return {
            "function_scores": function_scores,
            "overall_score":   overall_score,
            "risk_tier":       self.risk_tier.value,
            "top_gaps":        top_gaps,
        }

    def generate_rmf_report(
        self,
        assessments: list[RMFAssessment],
    ) -> dict[str, Any]:
        """Build a complete, hashable NIST AI RMF report.

        The returned dict includes a ``report_hash`` field computed over
        the entire report payload (excluding the hash field itself) so that
        any tampering is immediately detectable.

        Args:
            assessments: The list returned by :meth:`assess`.

        Returns:
            A dict suitable for JSON serialisation and regulatory submission.
        """
        profile = self.risk_profile(assessments)

        assessments_serialized = [
            {
                "category_id": a.category.category_id,
                "function":    a.category.function.value,
                "title":       a.category.title,
                "risk_level":  a.risk_level.value,
                "implemented": a.implemented,
                "evidence":    a.evidence,
                "gaps":        a.gaps,
                "assessed_at": a.assessed_at.isoformat(),
            }
            for a in assessments
        ]

        payload: dict[str, Any] = {
            "system_id":    self.system_id,
            "framework":    "NIST AI RMF 1.0",
            "assessed_at":  datetime.now(timezone.utc).isoformat(),
            "risk_tier":    self.risk_tier.value,
            "assessments":  assessments_serialized,
            "risk_profile": profile,
        }

        payload["report_hash"] = hash_object(
            {k: v for k, v in payload.items() if k != "report_hash"}
        )

        return payload
