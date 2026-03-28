"""
aria.gdpr — GDPR compliance utilities for AI inference data.

Provides tools for:
- Right-to-erasure (Art. 17): redact/delete personal data from inference records
- Data minimisation (Art. 5.1.c): mask PII fields before storage
- Retention policy (Art. 5.1.e): flag records exceeding retention window
- Consent logging: track consent tokens linked to inference records

Usage::

    from aria.gdpr import GDPRManager

    mgr = GDPRManager(storage, retention_days=90)

    # Redact all records for a data subject
    result = mgr.erase_subject("user-123")
    print(f"Erased {result.records_erased} records")

    # Check retention compliance
    violations = mgr.retention_violations()
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)

# Common PII patterns (conservative — only clearly identifiable patterns)
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",    "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b",                                         "[SSN]"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "[CARD]"),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ErasureResult:
    """Result of a right-to-erasure request."""
    subject_id:      str
    records_erased:  int
    records_found:   int
    timestamp:       str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def __str__(self) -> str:
        return (
            f"ErasureResult: subject={self.subject_id}  "
            f"erased={self.records_erased}/{self.records_found}  "
            f"at={self.timestamp}"
        )


@dataclass
class RetentionViolation:
    """A record that exceeds the configured retention window."""
    record_id:    str
    epoch_id:     str
    created_at:   str
    age_days:     float
    max_days:     int


@dataclass
class ConsentRecord:
    """Links a consent token to an inference subject."""
    subject_id:   str
    consent_token: str
    purpose:      str
    granted_at:   str = ""
    expires_at:   str = ""
    revoked:      bool = False

    def __post_init__(self):
        if not self.granted_at:
            self.granted_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_valid(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at:
            try:
                exp = datetime.fromisoformat(self.expires_at)
                if datetime.now(timezone.utc) > exp:
                    return False
            except ValueError:
                pass
        return True


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

def mask_pii(text: str, replacement_map: list[tuple[str, str]] | None = None) -> str:
    """Replace PII patterns in text with placeholder tokens.

    Args:
        text:            Input string.
        replacement_map: Custom list of (regex_pattern, replacement) tuples.
                         Defaults to built-in _PII_PATTERNS.

    Returns:
        String with PII replaced by placeholder tokens.
    """
    patterns = replacement_map if replacement_map is not None else _PII_PATTERNS
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


def mask_dict(data: dict, keys_to_mask: list[str] | None = None) -> dict:
    """Recursively mask PII in dict values.

    Args:
        data:         Input dictionary.
        keys_to_mask: Keys whose string values will be PII-masked.
                      If None, all string values are masked.
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = mask_dict(v, keys_to_mask)
        elif isinstance(v, str):
            if keys_to_mask is None or k in keys_to_mask:
                result[k] = mask_pii(v)
            else:
                result[k] = v
        elif isinstance(v, list):
            result[k] = [
                mask_pii(item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            result[k] = v
    return result


def pseudonymise(subject_id: str, salt: str = "") -> str:
    """One-way pseudonymisation of a subject identifier via SHA-256.

    Args:
        subject_id: Original identifier (e.g. user email, UUID).
        salt:       Optional salt for domain separation.

    Returns:
        Hex string (first 16 chars) suitable for use as a pseudonym.
    """
    raw = f"{salt}:{subject_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# GDPRManager
# ---------------------------------------------------------------------------

class GDPRManager:
    """GDPR compliance manager for ARIA inference storage.

    Args:
        storage:        StorageInterface implementation.
        retention_days: Maximum days to retain inference records (Art. 5.1.e).
        subject_field:  Metadata key that identifies the data subject.
    """

    def __init__(
        self,
        storage: "StorageInterface",
        retention_days: int = 90,
        subject_field: str = "subject_id",
    ) -> None:
        self._storage = storage
        self._retention_days = retention_days
        self._subject_field = subject_field
        self._consent_log: dict[str, ConsentRecord] = {}

    # ------------------------------------------------------------------
    # Right to erasure (Art. 17)
    # ------------------------------------------------------------------

    def erase_subject(self, subject_id: str) -> ErasureResult:
        """Erase all inference records associated with a data subject.

        Records are identified by ``subject_id`` in their metadata.
        This implementation masks PII in matched records (soft erasure).

        Args:
            subject_id: The data subject's identifier.

        Returns:
            ErasureResult with counts.
        """
        all_records = self._all_records()
        matched = [
            r for r in all_records
            if self._record_subject(r) == subject_id
        ]

        erased = 0
        for record in matched:
            try:
                self._erase_record(record)
                erased += 1
            except Exception as exc:
                _log.warning("GDPR erase failed for record %s: %s",
                             getattr(record, "record_id", "?"), exc)

        return ErasureResult(
            subject_id=subject_id,
            records_erased=erased,
            records_found=len(matched),
        )

    # ------------------------------------------------------------------
    # Retention policy (Art. 5.1.e)
    # ------------------------------------------------------------------

    def retention_violations(
        self,
        epoch_ids: list[str] | None = None,
    ) -> list[RetentionViolation]:
        """Find records exceeding the retention window.

        Args:
            epoch_ids: Optionally scope check to these epoch IDs.

        Returns:
            List of RetentionViolation for records that are too old.
        """
        all_records = self._all_records(epoch_ids)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        violations = []

        for record in all_records:
            created_at = self._record_timestamp(record)
            if created_at and created_at < cutoff:
                age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400
                violations.append(RetentionViolation(
                    record_id=str(getattr(record, "record_id", "")),
                    epoch_id=str(getattr(record, "epoch_id", "")),
                    created_at=created_at.isoformat(),
                    age_days=round(age_days, 1),
                    max_days=self._retention_days,
                ))

        return violations

    # ------------------------------------------------------------------
    # Consent management
    # ------------------------------------------------------------------

    def log_consent(self, consent: ConsentRecord) -> None:
        """Record a consent grant."""
        key = f"{consent.subject_id}:{consent.purpose}"
        self._consent_log[key] = consent

    def revoke_consent(self, subject_id: str, purpose: str) -> bool:
        """Revoke consent for a subject+purpose. Returns True if found."""
        key = f"{subject_id}:{purpose}"
        if key in self._consent_log:
            self._consent_log[key].revoked = True
            return True
        return False

    def has_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if valid consent exists for subject+purpose."""
        key = f"{subject_id}:{purpose}"
        record = self._consent_log.get(key)
        return record is not None and record.is_valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _all_records(self, epoch_ids: list[str] | None = None) -> list[Any]:
        try:
            if epoch_ids:
                records = []
                for eid in epoch_ids:
                    records.extend(self._storage.list_records_by_epoch(eid))
                return records
            if hasattr(self._storage, "list_all_records"):
                return self._storage.list_all_records()
            # Fallback: return empty — storage doesn't support bulk listing
            return []
        except Exception as exc:
            _log.warning("GDPR: could not list records: %s", exc)
            return []

    def _record_subject(self, record: Any) -> str | None:
        metadata = getattr(record, "metadata", {}) or {}
        return metadata.get(self._subject_field)

    def _record_timestamp(self, record: Any) -> datetime | None:
        raw = getattr(record, "created_at", None) or getattr(record, "timestamp", None)
        if raw is None:
            return None
        try:
            if isinstance(raw, datetime):
                return raw.replace(tzinfo=timezone.utc) if raw.tzinfo is None else raw
            dt = datetime.fromisoformat(str(raw))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except (ValueError, TypeError):
            return None

    def _erase_record(self, record: Any) -> None:
        """Soft-erase: replace PII fields with masked values."""
        if hasattr(record, "input_data") and isinstance(record.input_data, dict):
            record.input_data = mask_dict(record.input_data)
        if hasattr(record, "output_data") and isinstance(record.output_data, dict):
            record.output_data = mask_dict(record.output_data)
        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            meta = record.metadata
            if self._subject_field in meta:
                meta[self._subject_field] = "[ERASED]"
