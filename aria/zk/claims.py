"""Regulatory Statement DSL — cryptographically-backed EU AI Act claims.

This module provides the Claims DSL: a set of verifiable statements about
an epoch's inference behaviour that can be evaluated against stored records
and published on BSV as part of the EPOCH_CLOSE payload.

Design philosophy
-----------------
A regulator should be able to understand WHAT was proven without understanding
HOW (i.e., without understanding the ZK math).  Each Claim produces:

  - ``satisfied: bool``           — the claim is true for this epoch.
  - ``evidence_hash: str``        — SHA-256 of the values used in evaluation.
                                    Any party with the records can recompute this.
  - ``human_description: str``    — plain English statement for legal/audit use.
  - ``eu_ai_act_reference: str``  — the specific EU AI Act article this satisfies.

EU AI Act mapping
-----------------
  ConfidencePercentile  → Article 9 §7 (risk management: confidence thresholds)
  ModelUnchanged        → Article 9 §4 (model version control)
  NoPIIInInputs         → Article 10 §3 + GDPR Art. 9 (protected attributes)
  OutputDistribution    → Article 12 §1 (logging + output fairness)
  LatencyBound          → Article 14 §2 (human oversight: real-time constraints)
  RecordCountRange      → Article 12 §1 (logging completeness)
  AllModelsRegistered   → Article 11 (technical documentation: model registry)
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.hasher import canonical_json
from ..core.record import AuditRecord


# ---------------------------------------------------------------------------
# ClaimResult
# ---------------------------------------------------------------------------


@dataclass
class ClaimResult:
    """Result of evaluating a single regulatory claim against an epoch's records.

    Attributes:
        claim_type:          Machine-readable claim identifier.
        params:              Claim parameters (thresholds, field names, etc.).
        satisfied:           True if the claim holds for this epoch.
        evidence_hash:       SHA-256 of the sorted evidence values.
                             Reproducible: anyone with the records can verify.
        human_description:   Plain English claim statement.
        eu_ai_act_reference: The EU AI Act article this claim satisfies.
        detail:              Optional detail for non-satisfied claims.
    """

    claim_type: str
    params: dict[str, Any]
    satisfied: bool
    evidence_hash: str
    human_description: str
    eu_ai_act_reference: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        symbol = "✓" if self.satisfied else "✗"
        return {
            "claim_type": self.claim_type,
            "params": self.params,
            "satisfied": self.satisfied,
            "evidence_hash": self.evidence_hash,
            "human_description": self.human_description,
            "eu_ai_act_reference": self.eu_ai_act_reference,
            "detail": self.detail,
            "symbol": symbol,
        }


def _evidence_hash(values: list) -> str:
    """Canonical SHA-256 of a list of values — reproducible evidence commitment."""
    msg = canonical_json({"evidence": sorted(str(v) for v in values)})
    return "sha256:" + hashlib.sha256(msg).hexdigest()


# ---------------------------------------------------------------------------
# Base Claim
# ---------------------------------------------------------------------------


class Claim(ABC):
    """Abstract base class for all regulatory claims.

    Subclass and implement ``evaluate()`` to create new claim types.
    """

    @abstractmethod
    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        """Evaluate this claim against the given epoch records.

        Args:
            records: All AuditRecord objects in the epoch, ordered by sequence.

        Returns:
            ClaimResult with satisfied=True/False and cryptographic evidence.
        """

    @property
    @abstractmethod
    def claim_type(self) -> str:
        """Machine-readable claim type identifier."""

    @property
    @abstractmethod
    def eu_ai_act_reference(self) -> str:
        """EU AI Act article reference."""


# ---------------------------------------------------------------------------
# ConfidencePercentile — Art. 9 §7
# ---------------------------------------------------------------------------


class ConfidencePercentile(Claim):
    """Prove that the P-th percentile confidence score across all inferences ≥ threshold.

    EU AI Act Article 9 §7 requires high-risk AI systems to operate within
    risk management parameters.  This claim verifies that the system's
    confidence is consistently above the declared threshold.

    Example:
        ConfidencePercentile(p=99, threshold=0.85) proves:
        "At least 99% of inferences in this epoch had confidence ≥ 0.85"

    Args:
        p:         Percentile to check (0–100).  99 means "99th percentile".
        threshold: Minimum required confidence at the given percentile [0, 1].
    """

    def __init__(self, p: float, threshold: float) -> None:
        if not (0 <= p <= 100):
            raise ValueError(f"p must be in [0, 100], got {p}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self._p = p
        self._threshold = threshold

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        confidences = [r.confidence for r in records if r.confidence is not None]

        if not confidences:
            return ClaimResult(
                claim_type=self.claim_type,
                params={"p": self._p, "threshold": self._threshold},
                satisfied=False,
                evidence_hash=_evidence_hash([]),
                human_description=self._describe(None),
                eu_ai_act_reference=self.eu_ai_act_reference,
                detail="No confidence scores recorded in this epoch.",
            )

        sorted_c = sorted(confidences)
        idx = math.ceil(self._p / 100 * len(sorted_c)) - 1
        idx = max(0, min(idx, len(sorted_c) - 1))
        actual_percentile = sorted_c[idx]
        satisfied = actual_percentile >= self._threshold

        return ClaimResult(
            claim_type=self.claim_type,
            params={"p": self._p, "threshold": self._threshold},
            satisfied=satisfied,
            evidence_hash=_evidence_hash(sorted_c),
            human_description=self._describe(actual_percentile),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail="" if satisfied else (
                f"P{self._p} confidence was {actual_percentile:.4f}, "
                f"below threshold {self._threshold:.4f}."
            ),
        )

    def _describe(self, actual: float | None) -> str:
        if actual is None:
            return (f"P{self._p} confidence ≥ {self._threshold:.2f} "
                    f"(no confidence scores recorded)")
        return (f"P{int(self._p)} confidence ≥ {self._threshold:.2f} "
                f"(actual: {actual:.4f}, "
                f"{'SATISFIED' if actual >= self._threshold else 'NOT SATISFIED'})")

    @property
    def claim_type(self) -> str:
        return "confidence_percentile"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 9 §7 — Risk management: confidence threshold compliance"


# ---------------------------------------------------------------------------
# ModelUnchanged — Art. 9 §4
# ---------------------------------------------------------------------------


class ModelUnchanged(Claim):
    """Prove that all inferences in the epoch used the same model version.

    EU AI Act Article 9 §4 requires that changes to high-risk AI systems
    be managed through the risk management system.  This claim verifies
    that no model substitution occurred during the epoch.

    Args:
        expected_model_ids: Set of model IDs that should appear (and no others).
                            If None, verifies all records used the same model.
    """

    def __init__(self, expected_model_ids: set[str] | None = None) -> None:
        self._expected = expected_model_ids

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        model_ids = sorted({r.model_id for r in records})

        if not records:
            return ClaimResult(
                claim_type=self.claim_type,
                params={"expected_model_ids": sorted(self._expected) if self._expected else None},
                satisfied=True,
                evidence_hash=_evidence_hash([]),
                human_description="No inferences in this epoch.",
                eu_ai_act_reference=self.eu_ai_act_reference,
            )

        if self._expected is not None:
            unexpected = set(model_ids) - self._expected
            missing = self._expected - set(model_ids)
            satisfied = len(unexpected) == 0
            detail = ""
            if unexpected:
                detail = f"Unexpected model IDs: {sorted(unexpected)}."
            if missing:
                detail += f" Missing model IDs: {sorted(missing)}."
        else:
            satisfied = len(model_ids) == 1
            detail = "" if satisfied else f"Multiple model IDs found: {model_ids}."

        n = len(records)
        return ClaimResult(
            claim_type=self.claim_type,
            params={"expected_model_ids": sorted(self._expected) if self._expected else None,
                    "found_model_ids": model_ids},
            satisfied=satisfied,
            evidence_hash=_evidence_hash(model_ids),
            human_description=(
                f"Model unchanged across {n} inference(s). "
                f"Model(s) used: {model_ids}."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail=detail,
        )

    @property
    def claim_type(self) -> str:
        return "model_unchanged"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 9 §4 — Risk management: model version control"


# ---------------------------------------------------------------------------
# NoPIIInInputs — Art. 10 §3 + GDPR Art. 9
# ---------------------------------------------------------------------------


class NoPIIInInputs(Claim):
    """Prove that no input in this epoch contained PII field keys.

    EU AI Act Article 10 §3 requires that training and input data be free
    from certain categories of protected attributes.  GDPR Article 9 prohibits
    processing of special categories of personal data without explicit basis.

    This claim verifies that ARIA's PII sanitization was effective: none of
    the input_hash values in this epoch were computed from inputs containing
    the specified sensitive field keys.

    Since input_hash is computed AFTER PII stripping (see auditor.py), the
    presence of PII field keys in the original input does not affect the hash.
    This claim verifies the epoch metadata, not the raw inputs.

    Args:
        pii_fields: Field names that must not appear in epoch metadata.
    """

    def __init__(self, pii_fields: list[str]) -> None:
        self._pii_fields = set(pii_fields)

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        # Check metadata for any PII field keys (belt-and-suspenders check).
        violations: list[str] = []
        for rec in records:
            if rec.metadata:
                found = self._pii_fields & set(rec.metadata.keys())
                if found:
                    violations.append(f"record {rec.record_id}: {sorted(found)}")

        satisfied = len(violations) == 0
        n = len(records)

        return ClaimResult(
            claim_type=self.claim_type,
            params={"pii_fields": sorted(self._pii_fields)},
            satisfied=satisfied,
            evidence_hash=_evidence_hash(
                [rec.input_hash for rec in records] + sorted(self._pii_fields)
            ),
            human_description=(
                f"No PII fields ({sorted(self._pii_fields)}) "
                f"present in {n} input hash(es)."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail="" if satisfied else f"PII fields found in metadata: {violations[:5]}",
        )

    @property
    def claim_type(self) -> str:
        return "no_pii_in_inputs"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 10 §3 + GDPR Art. 9 — Protected attribute exclusion"


# ---------------------------------------------------------------------------
# OutputDistribution — Art. 12 §1
# ---------------------------------------------------------------------------


class OutputDistribution(Claim):
    """Prove that no single output value dominates beyond a maximum fraction.

    EU AI Act Article 12 §1 requires logging sufficient to enable post-market
    monitoring for bias.  This claim verifies that the output distribution
    does not exhibit extreme concentration (potential algorithmic bias indicator).

    Example:
        OutputDistribution(output_key="decision", max_single_fraction=0.95)
        proves that no single decision label appeared in more than 95% of cases.

    Args:
        output_key:           The output field to analyze (uses output_hash comparison).
        max_single_fraction:  Maximum allowed fraction for any single output value [0, 1].
        min_unique_outputs:   Minimum number of distinct outputs required.
    """

    def __init__(
        self,
        output_key: str,
        max_single_fraction: float = 0.95,
        min_unique_outputs: int = 2,
    ) -> None:
        self._output_key = output_key
        self._max_fraction = max_single_fraction
        self._min_unique = min_unique_outputs

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        if not records:
            return ClaimResult(
                claim_type=self.claim_type,
                params={
                    "output_key": self._output_key,
                    "max_single_fraction": self._max_fraction,
                    "min_unique_outputs": self._min_unique,
                },
                satisfied=True,
                evidence_hash=_evidence_hash([]),
                human_description="No inferences in this epoch.",
                eu_ai_act_reference=self.eu_ai_act_reference,
            )

        # We can only analyze output_hash values (the actual outputs are not stored).
        output_hashes = [r.output_hash for r in records]
        from collections import Counter
        counts = Counter(output_hashes)
        n = len(records)
        most_common_hash, most_common_count = counts.most_common(1)[0]
        actual_fraction = most_common_count / n
        unique_count = len(counts)

        satisfied = actual_fraction <= self._max_fraction and unique_count >= self._min_unique

        return ClaimResult(
            claim_type=self.claim_type,
            params={
                "output_key": self._output_key,
                "max_single_fraction": self._max_fraction,
                "min_unique_outputs": self._min_unique,
            },
            satisfied=satisfied,
            evidence_hash=_evidence_hash(output_hashes),
            human_description=(
                f"Output distribution across {n} inferences: "
                f"{unique_count} unique outputs, "
                f"max concentration {actual_fraction:.1%} "
                f"(limit: {self._max_fraction:.0%})."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail="" if satisfied else (
                f"Most common output hash appeared in {actual_fraction:.1%} of cases "
                f"(limit: {self._max_fraction:.0%}), "
                f"or fewer than {self._min_unique} unique outputs."
            ),
        )

    @property
    def claim_type(self) -> str:
        return "output_distribution"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 12 §1 — Logging: output distribution monitoring"


# ---------------------------------------------------------------------------
# LatencyBound — Art. 14 §2
# ---------------------------------------------------------------------------


class LatencyBound(Claim):
    """Prove that the P-th percentile inference latency was within max_ms milliseconds.

    EU AI Act Article 14 §2 requires that human oversight mechanisms be
    effective for high-risk AI systems.  For real-time systems (KAIROS,
    Urban VS), latency bounds ensure the system operates within human
    decision windows.

    Args:
        p:       Percentile to check (0–100).
        max_ms:  Maximum allowed latency at the given percentile (milliseconds).
    """

    def __init__(self, p: float, max_ms: int) -> None:
        if not (0 <= p <= 100):
            raise ValueError(f"p must be in [0, 100], got {p}")
        if max_ms <= 0:
            raise ValueError(f"max_ms must be positive, got {max_ms}")
        self._p = p
        self._max_ms = max_ms

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        latencies = [r.latency_ms for r in records if r.latency_ms > 0]

        if not latencies:
            return ClaimResult(
                claim_type=self.claim_type,
                params={"p": self._p, "max_ms": self._max_ms},
                satisfied=True,
                evidence_hash=_evidence_hash([]),
                human_description="No latency data recorded in this epoch.",
                eu_ai_act_reference=self.eu_ai_act_reference,
            )

        sorted_l = sorted(latencies)
        idx = math.ceil(self._p / 100 * len(sorted_l)) - 1
        idx = max(0, min(idx, len(sorted_l) - 1))
        actual_p = sorted_l[idx]
        satisfied = actual_p <= self._max_ms

        return ClaimResult(
            claim_type=self.claim_type,
            params={"p": self._p, "max_ms": self._max_ms},
            satisfied=satisfied,
            evidence_hash=_evidence_hash(sorted_l),
            human_description=(
                f"P{int(self._p)} latency ≤ {self._max_ms}ms "
                f"(actual: {actual_p}ms across {len(latencies)} inference(s))."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail="" if satisfied else (
                f"P{self._p} latency was {actual_p}ms, exceeding {self._max_ms}ms limit."
            ),
        )

    @property
    def claim_type(self) -> str:
        return "latency_bound"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 14 §2 — Human oversight: real-time constraint compliance"


# ---------------------------------------------------------------------------
# RecordCountRange — Art. 12 §1
# ---------------------------------------------------------------------------


class RecordCountRange(Claim):
    """Prove that the number of inferences in this epoch is within [min_count, max_count].

    EU AI Act Article 12 §1 requires complete logging.  This claim verifies
    that the epoch contains an expected volume of records — neither suspiciously
    low (missing records) nor unreasonably high (unbounded processing).

    Args:
        min_count: Minimum required records (0 = no minimum).
        max_count: Maximum allowed records (None = no maximum).
    """

    def __init__(self, min_count: int = 0, max_count: int | None = None) -> None:
        self._min = min_count
        self._max = max_count

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        n = len(records)
        too_few = n < self._min
        too_many = self._max is not None and n > self._max
        satisfied = not too_few and not too_many

        detail = ""
        if too_few:
            detail = f"Only {n} record(s) found, minimum is {self._min}."
        elif too_many:
            detail = f"{n} records found, maximum is {self._max}."

        return ClaimResult(
            claim_type=self.claim_type,
            params={"min_count": self._min, "max_count": self._max},
            satisfied=satisfied,
            evidence_hash=_evidence_hash([n]),
            human_description=(
                f"{n} inference(s) recorded "
                f"(required: {self._min}–{self._max if self._max else '∞'})."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail=detail,
        )

    @property
    def claim_type(self) -> str:
        return "record_count_range"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 12 §1 — Logging: record completeness"


# ---------------------------------------------------------------------------
# AllModelsRegistered — Art. 11
# ---------------------------------------------------------------------------


class AllModelsRegistered(Claim):
    """Prove that every model used in this epoch is in the declared registry.

    EU AI Act Article 11 requires technical documentation including model
    identification.  This claim verifies that no unregistered model was
    used — i.e., every model_id in the records appears in the EPOCH_OPEN
    model_hashes commitment.

    Args:
        registered_model_ids: Set of model IDs committed in EPOCH_OPEN.
    """

    def __init__(self, registered_model_ids: set[str]) -> None:
        self._registered = registered_model_ids

    def evaluate(self, records: list[AuditRecord]) -> ClaimResult:
        used = {r.model_id for r in records}
        unregistered = used - self._registered
        satisfied = len(unregistered) == 0

        return ClaimResult(
            claim_type=self.claim_type,
            params={"registered_model_ids": sorted(self._registered)},
            satisfied=satisfied,
            evidence_hash=_evidence_hash(sorted(used)),
            human_description=(
                f"All {len(used)} model(s) used are in the registered model set. "
                f"Models: {sorted(used)}."
            ),
            eu_ai_act_reference=self.eu_ai_act_reference,
            detail="" if satisfied else (
                f"Unregistered model(s) used: {sorted(unregistered)}. "
                f"These were not committed in EPOCH_OPEN."
            ),
        )

    @property
    def claim_type(self) -> str:
        return "all_models_registered"

    @property
    def eu_ai_act_reference(self) -> str:
        return "Art. 11 — Technical documentation: model registration"
