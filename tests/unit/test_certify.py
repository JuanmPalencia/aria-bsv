"""Tests for aria.certify — Cryptographic audit certificates."""

from __future__ import annotations

import time
import pytest

from aria.certify import Certifier, Certificate
from aria.core.record import AuditRecord
from aria.storage.sqlite import SQLiteStorage


def _storage_with_closed_epoch():
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-cert",
        system_id="sys-cert",
        open_txid="tx_" + "a" * 60,
        model_hashes={"m1": "sha256:" + "a" * 64},
        state_hash="sha256:" + "b" * 64,
        opened_at=now,
    )
    storage.save_epoch_close(
        epoch_id="ep-cert",
        close_txid="tx_" + "c" * 60,
        merkle_root="sha256:" + "d" * 64,
        records_count=5,
        closed_at=now + 60,
    )

    for i in range(5):
        storage.save_record(AuditRecord(
            epoch_id="ep-cert",
            model_id="m1",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.9,
            latency_ms=100,
        ))

    return storage


def _storage_with_open_epoch():
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-open",
        system_id="sys",
        open_txid="tx_" + "e" * 60,
        model_hashes={},
        state_hash="sha256:" + "f" * 64,
        opened_at=now,
    )
    for i in range(3):
        storage.save_record(AuditRecord(
            epoch_id="ep-open",
            model_id="m1",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.8,
            latency_ms=50,
        ))

    return storage


class TestCertifier:
    def test_certify_closed_epoch_passes(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        assert isinstance(cert, Certificate)
        assert cert.passed
        assert cert.grade in ("A", "B")
        assert len(cert.checks_passed) > 0

    def test_certify_open_epoch_has_failures(self):
        storage = _storage_with_open_epoch()
        cert = Certifier(storage).certify_epoch("ep-open")
        assert not cert.passed
        assert any("epoch_closed" in f for f in cert.checks_failed)

    def test_certify_nonexistent_raises(self):
        storage = SQLiteStorage(dsn="sqlite://")
        with pytest.raises(ValueError, match="not found"):
            Certifier(storage).certify_epoch("nonexistent")

    def test_certificate_id_format(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        assert cert.certificate_id.startswith("cert_")

    def test_fingerprint_is_sha256(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        assert len(cert.fingerprint) == 64
        int(cert.fingerprint, 16)  # valid hex

    def test_to_dict(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        d = cert.to_dict()
        assert d["passed"] is True
        assert "certificate_id" in d
        assert "fingerprint" in d

    def test_to_json(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        j = cert.to_json()
        import json
        data = json.loads(j)
        assert data["epoch_id"] == "ep-cert"

    def test_summary(self):
        storage = _storage_with_closed_epoch()
        cert = Certifier(storage).certify_epoch("ep-cert")
        text = cert.summary()
        assert "PASSED" in text
        assert "ep-cert" in text


class TestVerify:
    def test_verify_valid_certificate(self):
        storage = _storage_with_closed_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-cert")
        assert certifier.verify(cert) is True

    def test_verify_tampered_fingerprint(self):
        storage = _storage_with_closed_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-cert")
        cert.fingerprint = "0" * 64
        assert certifier.verify(cert) is False

    def test_verify_tampered_records_count(self):
        storage = _storage_with_closed_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-cert")
        cert.records_count = 999
        assert certifier.verify(cert) is False

    def test_verify_nonexistent_epoch(self):
        storage = _storage_with_closed_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-cert")
        cert.epoch_id = "nonexistent"
        assert certifier.verify(cert) is False


class TestBadge:
    def test_badge_svg_for_passing(self):
        storage = _storage_with_closed_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-cert")
        svg = certifier.badge(cert)
        assert "<svg" in svg
        assert "ARIA Certified" in svg
        assert "#22c55e" in svg

    def test_badge_svg_for_failing(self):
        storage = _storage_with_open_epoch()
        certifier = Certifier(storage)
        cert = certifier.certify_epoch("ep-open")
        svg = certifier.badge(cert)
        assert "<svg" in svg
        assert "FAILED" in svg
        assert "#ef4444" in svg
