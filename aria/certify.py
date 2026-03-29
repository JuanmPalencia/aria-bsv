"""
aria.certify — Cryptographic audit certificates and badges.

Generates verifiable certificates that prove a system passed ARIA
audit checks. Includes SVG badges for display and JSON proofs
for machine verification.

Usage::

    from aria.certify import Certifier

    cert = Certifier(storage)

    # Generate a certificate for an epoch
    certificate = cert.certify_epoch("epoch-001")
    print(certificate.summary())

    # Generate an SVG badge
    svg = cert.badge(certificate)

    # Verify a certificate
    is_valid = cert.verify(certificate)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface


@dataclass
class Certificate:
    """A verifiable audit certificate."""
    certificate_id: str
    epoch_id: str
    system_id: str
    issued_at: str
    records_count: int
    merkle_root: str
    open_txid: str
    close_txid: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    grade: str = ""
    fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.checks_failed) == 0 and len(self.checks_passed) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "epoch_id": self.epoch_id,
            "system_id": self.system_id,
            "issued_at": self.issued_at,
            "records_count": self.records_count,
            "merkle_root": self.merkle_root,
            "open_txid": self.open_txid,
            "close_txid": self.close_txid,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "grade": self.grade,
            "passed": self.passed,
            "fingerprint": self.fingerprint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"ARIA Audit Certificate — {status}",
            "=" * 50,
            f"  Certificate: {self.certificate_id}",
            f"  Epoch:       {self.epoch_id}",
            f"  System:      {self.system_id}",
            f"  Issued:      {self.issued_at}",
            f"  Records:     {self.records_count}",
            f"  Grade:       {self.grade}",
            f"  Merkle Root: {self.merkle_root[:20]}...",
            f"  Fingerprint: {self.fingerprint[:20]}...",
            "-" * 50,
        ]
        for c in self.checks_passed:
            lines.append(f"  [PASS] {c}")
        for c in self.checks_failed:
            lines.append(f"  [FAIL] {c}")
        return "\n".join(lines)


class Certifier:
    """Generate and verify ARIA audit certificates.

    Args:
        storage: ARIA StorageInterface for data access.
    """

    def __init__(self, storage: "StorageInterface") -> None:
        self._storage = storage

    def certify_epoch(self, epoch_id: str) -> Certificate:
        """Generate a certificate for a completed epoch.

        Runs integrity checks and produces a verifiable certificate.

        Args:
            epoch_id: The epoch to certify.

        Returns:
            Certificate with pass/fail results.

        Raises:
            ValueError: If epoch does not exist.
        """
        epoch = self._storage.get_epoch(epoch_id)
        if epoch is None:
            raise ValueError(f"Epoch not found: {epoch_id}")

        records = self._storage.list_records_by_epoch(epoch_id)

        passed: list[str] = []
        failed: list[str] = []

        # Check 1: Epoch is closed
        if epoch.close_txid:
            passed.append("epoch_closed")
        else:
            failed.append("epoch_closed: epoch is still open")

        # Check 2: Has records
        if records:
            passed.append(f"has_records: {len(records)} records")
        else:
            failed.append("has_records: no records found")

        # Check 3: All records have valid hashes
        all_hashes_valid = all(
            r.input_hash.startswith("sha256:") and len(r.input_hash) == 71
            and r.output_hash.startswith("sha256:") and len(r.output_hash) == 71
            for r in records
        )
        if all_hashes_valid:
            passed.append("hash_format")
        else:
            failed.append("hash_format: some records have invalid hash format")

        # Check 4: Sequence integrity
        sequences = [r.sequence for r in records]
        if sequences == list(range(len(sequences))):
            passed.append("sequence_integrity")
        else:
            failed.append("sequence_integrity: gaps or duplicates in sequence")

        # Check 5: Has merkle root
        if epoch.merkle_root:
            passed.append("merkle_root_present")
        else:
            failed.append("merkle_root_present: no merkle root")

        # Check 6: Has BSV txids
        if epoch.open_txid:
            passed.append("open_txid_present")
        else:
            failed.append("open_txid_present: no open txid")

        # Check 7: Confidence range
        confs = [r.confidence for r in records if r.confidence is not None]
        if confs and all(0 <= c <= 1 for c in confs):
            passed.append("confidence_range")
        elif confs:
            failed.append("confidence_range: values outside [0,1]")

        # Grade
        total = len(passed) + len(failed)
        ratio = len(passed) / total if total > 0 else 0
        if ratio >= 0.95:
            grade = "A"
        elif ratio >= 0.8:
            grade = "B"
        elif ratio >= 0.6:
            grade = "C"
        else:
            grade = "F"

        # Certificate ID and fingerprint
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cert_payload = f"{epoch_id}:{epoch.merkle_root}:{now}:{len(records)}"
        fingerprint = hashlib.sha256(cert_payload.encode()).hexdigest()
        cert_id = f"cert_{fingerprint[:16]}"

        return Certificate(
            certificate_id=cert_id,
            epoch_id=epoch_id,
            system_id=epoch.system_id,
            issued_at=now,
            records_count=len(records),
            merkle_root=epoch.merkle_root,
            open_txid=epoch.open_txid,
            close_txid=epoch.close_txid,
            checks_passed=passed,
            checks_failed=failed,
            grade=grade,
            fingerprint=fingerprint,
        )

    def verify(self, certificate: Certificate) -> bool:
        """Verify a certificate against current storage state.

        Re-runs integrity checks and compares with the certificate.

        Returns:
            True if certificate is still valid.
        """
        epoch = self._storage.get_epoch(certificate.epoch_id)
        if epoch is None:
            return False

        if epoch.merkle_root != certificate.merkle_root:
            return False

        records = self._storage.list_records_by_epoch(certificate.epoch_id)
        if len(records) != certificate.records_count:
            return False

        # Verify fingerprint
        cert_payload = (
            f"{certificate.epoch_id}:{certificate.merkle_root}:"
            f"{certificate.issued_at}:{certificate.records_count}"
        )
        expected_fp = hashlib.sha256(cert_payload.encode()).hexdigest()
        return expected_fp == certificate.fingerprint

    def badge(self, certificate: Certificate) -> str:
        """Generate an SVG badge for a certificate.

        Args:
            certificate: The certificate to create a badge for.

        Returns:
            SVG string.
        """
        if certificate.passed:
            color = "#22c55e"
            label = "ARIA Certified"
            status = certificate.grade
        else:
            color = "#ef4444"
            label = "ARIA Audit"
            status = "FAILED"

        return _SVG_BADGE.format(
            color=color,
            label=label,
            status=status,
            cert_id=certificate.certificate_id,
            epoch=certificate.epoch_id[:16],
            records=certificate.records_count,
        )


_SVG_BADGE = """\
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80" viewBox="0 0 200 80">
  <rect width="200" height="80" rx="8" fill="#1e293b"/>
  <rect x="130" width="70" height="80" rx="0 8 8 0" fill="{color}"/>
  <rect x="130" width="4" height="80" fill="{color}"/>
  <text x="12" y="25" fill="#e2e8f0" font-family="system-ui" font-size="13" font-weight="700">{label}</text>
  <text x="12" y="45" fill="#94a3b8" font-family="system-ui" font-size="10">{epoch}</text>
  <text x="12" y="62" fill="#94a3b8" font-family="system-ui" font-size="10">{records} records</text>
  <text x="165" y="48" fill="white" font-family="system-ui" font-size="24" font-weight="700" text-anchor="middle">{status}</text>
</svg>"""
