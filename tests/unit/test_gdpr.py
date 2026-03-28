"""Tests for aria.gdpr — GDPR compliance utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from aria.gdpr import (
    ConsentRecord,
    ErasureResult,
    GDPRManager,
    RetentionViolation,
    mask_dict,
    mask_pii,
    pseudonymise,
)


# ---------------------------------------------------------------------------
# mask_pii
# ---------------------------------------------------------------------------

class TestMaskPii:
    def test_masks_email(self):
        text = "Contact us at user@example.com for support"
        result = mask_pii(text)
        assert "user@example.com" not in result
        assert "[EMAIL]" in result

    def test_masks_phone(self):
        text = "Call 555-867-5309 now"
        result = mask_pii(text)
        assert "555-867-5309" not in result
        assert "[PHONE]" in result

    def test_masks_ssn(self):
        text = "SSN: 123-45-6789"
        result = mask_pii(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_no_pii(self):
        text = "Hello world, this is clean text."
        result = mask_pii(text)
        assert result == text

    def test_multiple_emails(self):
        text = "From: a@b.com To: c@d.org"
        result = mask_pii(text)
        assert "a@b.com" not in result
        assert "c@d.org" not in result

    def test_custom_patterns(self):
        text = "ID: ABC-1234"
        result = mask_pii(text, replacement_map=[(r"ABC-\d+", "[ID]")])
        assert "ABC-1234" not in result
        assert "[ID]" in result

    def test_empty_string(self):
        assert mask_pii("") == ""


# ---------------------------------------------------------------------------
# mask_dict
# ---------------------------------------------------------------------------

class TestMaskDict:
    def test_masks_string_values(self):
        d = {"content": "Call me at 555-123-4567"}
        result = mask_dict(d)
        assert "555-123-4567" not in result["content"]

    def test_preserves_non_string(self):
        d = {"count": 42, "flag": True}
        result = mask_dict(d)
        assert result["count"] == 42
        assert result["flag"] is True

    def test_recursive_dict(self):
        d = {"messages": [{"role": "user", "content": "hi"}], "meta": {"email": "a@b.com"}}
        result = mask_dict(d)
        assert "a@b.com" not in result["meta"]["email"]

    def test_keys_to_mask_filter(self):
        d = {"content": "user@test.com", "safe": "user@test.com"}
        result = mask_dict(d, keys_to_mask=["content"])
        assert "[EMAIL]" in result["content"]
        assert result["safe"] == "user@test.com"  # not masked

    def test_list_values_masked(self):
        d = {"items": ["hello user@test.com"]}
        result = mask_dict(d)
        assert "user@test.com" not in result["items"][0]


# ---------------------------------------------------------------------------
# pseudonymise
# ---------------------------------------------------------------------------

class TestPseudonymise:
    def test_returns_string(self):
        result = pseudonymise("user-123")
        assert isinstance(result, str)

    def test_deterministic(self):
        assert pseudonymise("user-123") == pseudonymise("user-123")

    def test_different_ids(self):
        assert pseudonymise("user-1") != pseudonymise("user-2")

    def test_salt_changes_result(self):
        assert pseudonymise("user-1", "salt-a") != pseudonymise("user-1", "salt-b")

    def test_length(self):
        result = pseudonymise("user-123")
        assert len(result) == 16


# ---------------------------------------------------------------------------
# ConsentRecord
# ---------------------------------------------------------------------------

class TestConsentRecord:
    def test_basic(self):
        c = ConsentRecord(
            subject_id="user-1",
            consent_token="tok-abc",
            purpose="analytics",
        )
        assert c.is_valid is True

    def test_revoked(self):
        c = ConsentRecord(
            subject_id="user-1",
            consent_token="tok-abc",
            purpose="analytics",
            revoked=True,
        )
        assert c.is_valid is False

    def test_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        c = ConsentRecord(
            subject_id="u",
            consent_token="t",
            purpose="p",
            expires_at=past,
        )
        assert c.is_valid is False

    def test_future_expiry(self):
        future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        c = ConsentRecord(
            subject_id="u",
            consent_token="t",
            purpose="p",
            expires_at=future,
        )
        assert c.is_valid is True

    def test_granted_at_auto_set(self):
        c = ConsentRecord(subject_id="u", consent_token="t", purpose="p")
        assert c.granted_at != ""


# ---------------------------------------------------------------------------
# GDPRManager helpers
# ---------------------------------------------------------------------------

def _record(subject_id: str | None = None, days_old: int = 0,
            email: str | None = None):
    r = MagicMock()
    r.record_id = "rec-test"
    r.epoch_id = "ep-test"
    r.metadata = {}
    if subject_id:
        r.metadata["subject_id"] = subject_id
    if days_old:
        ts = datetime.now(timezone.utc) - timedelta(days=days_old)
        r.created_at = ts.isoformat()
    else:
        r.created_at = None
    r.input_data = {"content": email or "hello"}
    r.output_data = {"result": "ok"}
    return r


def _storage_with_records(records: list):
    storage = MagicMock()
    storage.list_all_records.return_value = records
    storage.list_records_by_epoch.side_effect = lambda eid: [
        r for r in records if getattr(r, "epoch_id", None) == eid
    ]
    return storage


# ---------------------------------------------------------------------------
# GDPRManager.erase_subject
# ---------------------------------------------------------------------------

class TestEraseSubject:
    def test_erases_matched_records(self):
        recs = [
            _record(subject_id="user-1"),
            _record(subject_id="user-2"),
            _record(subject_id="user-1"),
        ]
        mgr = GDPRManager(_storage_with_records(recs))
        result = mgr.erase_subject("user-1")
        assert result.records_erased == 2
        assert result.records_found == 2

    def test_only_erases_matching(self):
        recs = [
            _record(subject_id="user-1"),
            _record(subject_id="user-9"),
        ]
        mgr = GDPRManager(_storage_with_records(recs))
        result = mgr.erase_subject("user-1")
        assert result.records_erased == 1

    def test_no_matches(self):
        recs = [_record(subject_id="user-99")]
        mgr = GDPRManager(_storage_with_records(recs))
        result = mgr.erase_subject("user-1")
        assert result.records_erased == 0
        assert result.records_found == 0

    def test_erases_pii_in_input(self):
        r = _record(subject_id="user-1", email="victim@example.com")
        mgr = GDPRManager(_storage_with_records([r]))
        mgr.erase_subject("user-1")
        assert "victim@example.com" not in r.input_data.get("content", "")

    def test_marks_subject_erased(self):
        r = _record(subject_id="user-1")
        mgr = GDPRManager(_storage_with_records([r]))
        mgr.erase_subject("user-1")
        assert r.metadata.get("subject_id") == "[ERASED]"

    def test_result_has_timestamp(self):
        mgr = GDPRManager(_storage_with_records([]))
        result = mgr.erase_subject("user-1")
        assert result.timestamp != ""

    def test_str_representation(self):
        mgr = GDPRManager(_storage_with_records([]))
        result = ErasureResult(subject_id="u", records_erased=3, records_found=3)
        s = str(result)
        assert "erased=3" in s


# ---------------------------------------------------------------------------
# GDPRManager.retention_violations
# ---------------------------------------------------------------------------

class TestRetentionViolations:
    def test_old_record_flagged(self):
        r = _record(days_old=100)
        r.epoch_id = "ep-old"
        mgr = GDPRManager(_storage_with_records([r]), retention_days=90)
        violations = mgr.retention_violations(epoch_ids=["ep-old"])
        assert len(violations) == 1
        assert violations[0].age_days >= 100

    def test_recent_record_ok(self):
        r = _record(days_old=10)
        r.epoch_id = "ep-new"
        mgr = GDPRManager(_storage_with_records([r]), retention_days=90)
        violations = mgr.retention_violations(epoch_ids=["ep-new"])
        assert len(violations) == 0

    def test_no_timestamp_skipped(self):
        r = MagicMock()
        r.record_id = "r"
        r.epoch_id = "ep-1"
        r.metadata = {}
        r.created_at = None
        r.timestamp = None
        storage = MagicMock()
        storage.list_records_by_epoch.return_value = [r]
        mgr = GDPRManager(storage, retention_days=1)
        violations = mgr.retention_violations(epoch_ids=["ep-1"])
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# GDPRManager consent
# ---------------------------------------------------------------------------

class TestConsent:
    def test_log_and_check(self):
        mgr = GDPRManager(_storage_with_records([]))
        c = ConsentRecord(subject_id="u1", consent_token="t", purpose="analytics")
        mgr.log_consent(c)
        assert mgr.has_consent("u1", "analytics") is True

    def test_no_consent(self):
        mgr = GDPRManager(_storage_with_records([]))
        assert mgr.has_consent("u1", "analytics") is False

    def test_revoke(self):
        mgr = GDPRManager(_storage_with_records([]))
        c = ConsentRecord(subject_id="u1", consent_token="t", purpose="analytics")
        mgr.log_consent(c)
        mgr.revoke_consent("u1", "analytics")
        assert mgr.has_consent("u1", "analytics") is False

    def test_revoke_nonexistent(self):
        mgr = GDPRManager(_storage_with_records([]))
        assert mgr.revoke_consent("u1", "analytics") is False
