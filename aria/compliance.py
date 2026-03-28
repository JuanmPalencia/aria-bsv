"""
aria.compliance — EU AI Act + BRC-121 compliance checker for ARIA epochs.

Runs structured compliance checks against stored epochs and produces a
machine-readable + human-readable report that can be attached to audit trails,
shared with regulators, or embedded in the HTML portal.

Supported regulation frameworks
---------------------------------
BRC-121          Blockchain-anchored AI accountability protocol.
                 Checks §4 (epoch integrity) and §6 (audit trail completeness).

EU AI Act        Regulation (EU) 2024/1689.
                 Checks Art. 13 (transparency), Art. 17 (quality management),
                 Art. 72 (post-market monitoring) obligations that can be
                 verified from the audit record alone.

GDPR             Regulation (EU) 2016/679.
                 Data-minimisation checks on stored audit fields.

Usage::

    from aria.compliance import ComplianceChecker
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    checker = ComplianceChecker(storage)

    report = checker.check_epoch("epoch-xyz")
    if not report.passed:
        for v in report.violations:
            print("  ✗", v)

    # Or check an entire system:
    summary = checker.check_system("my-system", last_n=10)
    print(f"Compliance rate: {summary.compliance_rate:.1%}")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Regulation(str, Enum):
    BRC121   = "BRC-121"
    EU_AI    = "EU AI Act"
    GDPR     = "GDPR"
    INTERNAL = "Internal"


class CheckSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # Epoch fails audit — must fix
    WARNING  = "WARNING"    # Should fix but does not invalidate epoch
    INFO     = "INFO"       # Advisory


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ComplianceCheck:
    """Result of a single compliance rule check."""
    rule_id:     str
    regulation:  Regulation
    description: str
    passed:      bool
    severity:    CheckSeverity = CheckSeverity.CRITICAL
    detail:      str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id":     self.rule_id,
            "regulation":  self.regulation.value,
            "description": self.description,
            "passed":      self.passed,
            "severity":    self.severity.value,
            "detail":      self.detail,
        }


@dataclass
class ComplianceReport:
    """Full compliance report for a single epoch."""
    epoch_id:    str
    system_id:   str
    checked_at:  float = field(default_factory=time.time)
    checks:      list[ComplianceCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if all CRITICAL checks pass."""
        return all(
            c.passed for c in self.checks
            if c.severity == CheckSeverity.CRITICAL
        )

    @property
    def violations(self) -> list[str]:
        return [
            f"[{c.regulation.value}] {c.rule_id}: {c.description}"
            + (f" — {c.detail}" if c.detail else "")
            for c in self.checks if not c.passed
        ]

    @property
    def critical_failures(self) -> int:
        return sum(
            1 for c in self.checks
            if not c.passed and c.severity == CheckSeverity.CRITICAL
        )

    @property
    def warnings(self) -> int:
        return sum(
            1 for c in self.checks
            if not c.passed and c.severity == CheckSeverity.WARNING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_id":          self.epoch_id,
            "system_id":         self.system_id,
            "checked_at":        self.checked_at,
            "passed":            self.passed,
            "critical_failures": self.critical_failures,
            "warnings":          self.warnings,
            "checks":            [c.to_dict() for c in self.checks],
            "violations":        self.violations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = [
            f"Compliance Report — Epoch {self.epoch_id}",
            f"System: {self.system_id}",
            f"Overall: {'PASS' if self.passed else 'FAIL'}  "
            f"({self.critical_failures} critical, {self.warnings} warnings)",
            "-" * 60,
        ]
        for c in self.checks:
            icon = "✓" if c.passed else ("✗" if c.severity == CheckSeverity.CRITICAL else "⚠")
            lines.append(
                f"{icon} [{c.regulation.value}] {c.rule_id} — {c.description}"
            )
            if not c.passed and c.detail:
                lines.append(f"    Detail: {c.detail}")
        return "\n".join(lines)


@dataclass
class SystemComplianceReport:
    """Aggregate compliance summary across multiple epochs."""
    system_id: str
    epoch_reports: list[ComplianceReport] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def total_epochs(self) -> int:
        return len(self.epoch_reports)

    @property
    def passing_epochs(self) -> int:
        return sum(1 for r in self.epoch_reports if r.passed)

    @property
    def compliance_rate(self) -> float:
        if not self.epoch_reports:
            return 1.0
        return self.passing_epochs / self.total_epochs

    @property
    def all_violations(self) -> list[str]:
        violations = []
        for r in self.epoch_reports:
            for v in r.violations:
                violations.append(f"[{r.epoch_id}] {v}")
        return violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id":       self.system_id,
            "checked_at":      self.checked_at,
            "total_epochs":    self.total_epochs,
            "passing_epochs":  self.passing_epochs,
            "compliance_rate": round(self.compliance_rate, 4),
            "violations":      self.all_violations,
            "epoch_reports":   [r.to_dict() for r in self.epoch_reports],
        }


# ---------------------------------------------------------------------------
# Compliance rules
# ---------------------------------------------------------------------------

def _check_brc120_epoch_anchored(epoch: Any) -> ComplianceCheck:
    """BRC-121 §4.1: Epoch must have an open transaction ID."""
    passed = bool(epoch.open_txid and epoch.open_txid != "pending")
    return ComplianceCheck(
        rule_id="BRC121-4.1",
        regulation=Regulation.BRC121,
        description="Epoch open must be anchored to BSV",
        passed=passed,
        severity=CheckSeverity.CRITICAL,
        detail="" if passed else f"open_txid={epoch.open_txid!r}",
    )


def _check_brc120_epoch_closed(epoch: Any) -> ComplianceCheck:
    """BRC-121 §4.2: Epoch must be closed with a close transaction ID."""
    passed = bool(epoch.close_txid and epoch.close_txid != "pending")
    return ComplianceCheck(
        rule_id="BRC121-4.2",
        regulation=Regulation.BRC121,
        description="Epoch close must be anchored to BSV",
        passed=passed,
        severity=CheckSeverity.CRITICAL,
        detail="" if passed else f"close_txid={epoch.close_txid!r}",
    )


def _check_brc120_merkle_root(epoch: Any) -> ComplianceCheck:
    """BRC-121 §4.3: Epoch must have a non-empty Merkle root."""
    passed = bool(epoch.merkle_root)
    return ComplianceCheck(
        rule_id="BRC121-4.3",
        regulation=Regulation.BRC121,
        description="Epoch must have a Merkle root of all inference hashes",
        passed=passed,
        severity=CheckSeverity.CRITICAL,
        detail="" if passed else "merkle_root is empty",
    )


def _check_brc120_records_match(epoch: Any, actual_count: int) -> ComplianceCheck:
    """BRC-121 §4.4: Stored record count must match declared count."""
    passed = (epoch.records_count == actual_count)
    return ComplianceCheck(
        rule_id="BRC121-4.4",
        regulation=Regulation.BRC121,
        description="Declared record count must match stored records",
        passed=passed,
        severity=CheckSeverity.CRITICAL,
        detail=(
            ""
            if passed
            else f"declared={epoch.records_count}, actual={actual_count}"
        ),
    )


def _check_brc120_model_hashes(epoch: Any) -> ComplianceCheck:
    """BRC-121 §6.1: Epoch must commit to at least one model hash."""
    hashes = epoch.model_hashes or {}
    passed = len(hashes) >= 1
    return ComplianceCheck(
        rule_id="BRC121-6.1",
        regulation=Regulation.BRC121,
        description="Epoch must commit to at least one model hash",
        passed=passed,
        severity=CheckSeverity.WARNING,
        detail="" if passed else "model_hashes is empty",
    )


def _check_eu_ai_transparency(epoch: Any, records: list[Any]) -> ComplianceCheck:
    """EU AI Act Art. 13.1: Inference records must include model identification."""
    if not records:
        return ComplianceCheck(
            rule_id="EUAI-13.1",
            regulation=Regulation.EU_AI,
            description="Inference records must include model identification",
            passed=False,
            severity=CheckSeverity.WARNING,
            detail="No inference records found for this epoch",
        )
    without_model = sum(1 for r in records if not r.model_id)
    passed = without_model == 0
    return ComplianceCheck(
        rule_id="EUAI-13.1",
        regulation=Regulation.EU_AI,
        description="Inference records must include model identification",
        passed=passed,
        severity=CheckSeverity.WARNING,
        detail=(
            ""
            if passed
            else f"{without_model}/{len(records)} records missing model_id"
        ),
    )


def _check_eu_ai_confidence_logged(epoch: Any, records: list[Any]) -> ComplianceCheck:
    """EU AI Act Art. 13.2: Confidence/certainty scores should be logged."""
    if not records:
        return ComplianceCheck(
            rule_id="EUAI-13.2",
            regulation=Regulation.EU_AI,
            description="Confidence scores should be logged for high-risk AI",
            passed=True,   # vacuously OK when no records
            severity=CheckSeverity.INFO,
        )
    with_conf = sum(1 for r in records if r.confidence is not None)
    rate = with_conf / len(records)
    passed = rate >= 0.80  # Allow up to 20% missing (some models may not expose it)
    return ComplianceCheck(
        rule_id="EUAI-13.2",
        regulation=Regulation.EU_AI,
        description="Confidence scores should be logged for high-risk AI",
        passed=passed,
        severity=CheckSeverity.INFO,
        detail=(
            ""
            if passed
            else f"Only {with_conf}/{len(records)} ({rate:.0%}) records have confidence"
        ),
    )


def _check_eu_ai_latency_logged(records: list[Any]) -> ComplianceCheck:
    """EU AI Act Art. 17: Performance monitoring — latency must be recorded."""
    if not records:
        return ComplianceCheck(
            rule_id="EUAI-17.1",
            regulation=Regulation.EU_AI,
            description="Inference latency must be recorded for performance monitoring",
            passed=True,
            severity=CheckSeverity.INFO,
        )
    with_lat = sum(1 for r in records if r.latency_ms and r.latency_ms > 0)
    rate = with_lat / len(records)
    passed = rate >= 0.95
    return ComplianceCheck(
        rule_id="EUAI-17.1",
        regulation=Regulation.EU_AI,
        description="Inference latency must be recorded for performance monitoring",
        passed=passed,
        severity=CheckSeverity.WARNING,
        detail=(
            ""
            if passed
            else f"Only {with_lat}/{len(records)} ({rate:.0%}) records have latency_ms"
        ),
    )


def _check_eu_ai_post_market(epoch: Any) -> ComplianceCheck:
    """EU AI Act Art. 72: Epochs must form a continuous post-market trail."""
    # Proxy: epoch was actually closed (not still open)
    passed = epoch.close_txid is not None
    return ComplianceCheck(
        rule_id="EUAI-72.1",
        regulation=Regulation.EU_AI,
        description="Epochs must be formally closed to form a continuous monitoring trail",
        passed=passed,
        severity=CheckSeverity.WARNING,
        detail="" if passed else "Epoch has not been closed — post-market trail is incomplete",
    )


def _check_gdpr_no_pii_in_record_ids(records: list[Any]) -> ComplianceCheck:
    """GDPR Art. 5.1(c): Record IDs should not contain personal identifiers."""
    # Heuristic: check for common PII patterns (email, phone pattern) in record_id
    import re
    email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    violations = [
        r.record_id for r in records
        if r.record_id and email_re.search(r.record_id)
    ]
    passed = len(violations) == 0
    return ComplianceCheck(
        rule_id="GDPR-5.1c",
        regulation=Regulation.GDPR,
        description="Record IDs must not contain personal identifiers (email pattern)",
        passed=passed,
        severity=CheckSeverity.CRITICAL,
        detail=(
            ""
            if passed
            else f"Suspect record IDs: {violations[:3]}"
        ),
    )


def _check_gdpr_metadata_minimisation(records: list[Any]) -> ComplianceCheck:
    """GDPR Art. 5.1(c): Metadata fields should not contain obvious PII keys."""
    _PII_KEYS = frozenset({
        "email", "name", "phone", "ip", "ip_address", "user_id",
        "username", "address", "dob", "ssn", "passport",
    })
    violations = []
    for r in records:
        if not r.metadata:
            continue
        found = _PII_KEYS & {k.lower() for k in r.metadata}
        if found:
            violations.append((r.record_id, sorted(found)))
    passed = len(violations) == 0
    return ComplianceCheck(
        rule_id="GDPR-5.1c-meta",
        regulation=Regulation.GDPR,
        description="Record metadata must not store PII key names",
        passed=passed,
        severity=CheckSeverity.WARNING,
        detail=(
            ""
            if passed
            else f"Found PII keys in {len(violations)} record(s): {violations[:2]}"
        ),
    )


def _check_internal_min_records(records: list[Any], min_records: int) -> ComplianceCheck:
    """Internal policy: minimum records per epoch for statistical validity."""
    passed = len(records) >= min_records
    return ComplianceCheck(
        rule_id="INT-MIN-REC",
        regulation=Regulation.INTERNAL,
        description=f"Epoch should contain at least {min_records} inference records",
        passed=passed,
        severity=CheckSeverity.INFO,
        detail=(
            ""
            if passed
            else f"Only {len(records)} records; statistical drift tests need ≥{min_records}"
        ),
    )


# ---------------------------------------------------------------------------
# ComplianceChecker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    """Runs compliance checks on ARIA epochs against multiple frameworks.

    Args:
        storage:       Any StorageInterface implementation.
        min_records:   Minimum records per epoch for internal policy check (default 10).
        frameworks:    List of regulations to check. Default: all.
    """

    _ALL_FRAMEWORKS = (Regulation.BRC121, Regulation.EU_AI, Regulation.GDPR, Regulation.INTERNAL)

    def __init__(
        self,
        storage: "StorageInterface",
        min_records: int = 10,
        frameworks: list[Regulation] | None = None,
    ) -> None:
        self._storage = storage
        self._min_records = min_records
        self._frameworks = frozenset(frameworks or self._ALL_FRAMEWORKS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_epoch(self, epoch_id: str) -> ComplianceReport:
        """Run all compliance checks for a single epoch.

        Returns a ComplianceReport — never raises even if the epoch doesn't exist.
        """
        epochs = self._storage.list_epochs(limit=10_000)
        epoch = next((e for e in epochs if e.epoch_id == epoch_id), None)

        if epoch is None:
            # Synthesise a failed report
            return ComplianceReport(
                epoch_id=epoch_id,
                system_id="",
                checks=[
                    ComplianceCheck(
                        rule_id="ARIA-EPOCH",
                        regulation=Regulation.INTERNAL,
                        description="Epoch must exist in storage",
                        passed=False,
                        severity=CheckSeverity.CRITICAL,
                        detail=f"Epoch '{epoch_id}' not found",
                    )
                ],
            )

        records = self._storage.list_records_by_epoch(epoch_id)
        checks: list[ComplianceCheck] = []

        if Regulation.BRC121 in self._frameworks:
            checks += [
                _check_brc120_epoch_anchored(epoch),
                _check_brc120_epoch_closed(epoch),
                _check_brc120_merkle_root(epoch),
                _check_brc120_records_match(epoch, len(records)),
                _check_brc120_model_hashes(epoch),
            ]

        if Regulation.EU_AI in self._frameworks:
            checks += [
                _check_eu_ai_transparency(epoch, records),
                _check_eu_ai_confidence_logged(epoch, records),
                _check_eu_ai_latency_logged(records),
                _check_eu_ai_post_market(epoch),
            ]

        if Regulation.GDPR in self._frameworks:
            checks += [
                _check_gdpr_no_pii_in_record_ids(records),
                _check_gdpr_metadata_minimisation(records),
            ]

        if Regulation.INTERNAL in self._frameworks:
            checks += [
                _check_internal_min_records(records, self._min_records),
            ]

        return ComplianceReport(
            epoch_id=epoch_id,
            system_id=epoch.system_id,
            checks=checks,
        )

    def check_system(
        self,
        system_id: str | None = None,
        last_n: int = 20,
    ) -> SystemComplianceReport:
        """Run compliance checks across the last N epochs for a system.

        Args:
            system_id: Filter by system. None = all systems.
            last_n:    Maximum number of epochs to check (default 20).

        Returns:
            A SystemComplianceReport aggregating all epoch results.
        """
        epochs = self._storage.list_epochs(system_id=system_id, limit=last_n)
        reports = [self.check_epoch(e.epoch_id) for e in epochs]
        return SystemComplianceReport(
            system_id=system_id or "",
            epoch_reports=reports,
        )

    def quick_check(self, epoch_id: str) -> bool:
        """Convenience method — returns True only if epoch fully complies."""
        return self.check_epoch(epoch_id).passed
