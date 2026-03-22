"""EpochStatement — complete cryptographic attestation for a closed epoch.

An EpochStatement combines:
  1. Evaluated regulatory claims (Claims DSL)
  2. ZK aggregate proof of all inferences (if ZK prover configured)
  3. A binding commitment (SHA-256 of all the above)

The statement is included in the EPOCH_CLOSE BSV payload.  Anyone with:
  - The two BSV transaction IDs (EPOCH_OPEN + EPOCH_CLOSE)
  - The verifying key (published once per model version)
  - The proof bytes (available from the operator's local storage)

can independently verify:
  - All regulatory claims hold for the epoch
  - Every inference proof is valid
  - No claim was fabricated after the fact (anchored to BSV timestamp)

Regulatory report
-----------------
``EpochStatement.to_regulatory_report()`` produces a human-readable
plain-text report suitable for submission to a notified body, DPA,
or market surveillance authority under the EU AI Act.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.hasher import canonical_json
from .aggregate import AggregateProof
from .claims import ClaimResult


@dataclass
class EpochStatement:
    """Full cryptographic attestation for a single ARIA epoch.

    Attributes:
        epoch_id:         Epoch identifier (``ep_{ts}_{counter}``).
        system_id:        Registered system identifier.
        claims:           List of evaluated ClaimResults.
        aggregate_proof:  ZK aggregate proof over all inference proofs
                          (None if ZK is not configured).
        open_txid:        BSV txid of the EPOCH_OPEN transaction.
        closed_at:        UTC datetime when the epoch was closed.
        n_records:        Number of inferences in the epoch.
        statement_hash:   SHA-256 commitment over (epoch_id, claims, aggregate).
                          Computed automatically at construction.
    """

    epoch_id: str
    system_id: str
    claims: list[ClaimResult]
    aggregate_proof: AggregateProof | None = None
    open_txid: str = ""
    closed_at: datetime | None = None
    n_records: int = 0
    statement_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.statement_hash = self._compute_hash()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_satisfied(self) -> bool:
        """Return True if every claim in this statement is satisfied."""
        return all(c.satisfied for c in self.claims)

    def to_bsv_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for inclusion in EPOCH_CLOSE OP_RETURN.

        Full proof bytes are NOT included (too large for on-chain storage).
        The on-chain payload contains only hashes and claim results — enough
        for a regulator to verify the commitment and check if all claims passed.

        Full proof bytes are stored in local SQLite and available from the
        operator's storage for independent verification.
        """
        payload: dict[str, Any] = {
            "zk_enabled": self.aggregate_proof is not None,
            "claims_count": len(self.claims),
            "all_claims_satisfied": self.all_satisfied(),
            "claims": [c.to_dict() for c in self.claims],
            "statement_hash": self.statement_hash,
        }
        if self.aggregate_proof is not None:
            payload["aggregate_proof"] = self.aggregate_proof.to_dict()
        return payload

    def to_regulatory_report(self) -> str:
        """Generate a human-readable plain-text audit report for regulators.

        The format is designed for:
          - Notified bodies conducting conformity assessments (Art. 43)
          - Data Protection Authorities (GDPR Art. 51)
          - Market surveillance authorities (Art. 63)
          - Internal audit teams

        Returns:
            Multi-line plain-text string.  Suitable for PDF generation.
        """
        lines = [
            "=" * 72,
            "  ARIA EPOCH AUDIT REPORT",
            "  BRC-120 Cryptographic Accountability — Auditable Real-time",
            "  Inference Architecture",
            "=" * 72,
            "",
            f"  System:        {self.system_id}",
            f"  Epoch:         {self.epoch_id}",
            f"  Inferences:    {self.n_records}",
        ]

        if self.closed_at:
            lines.append(f"  Closed at:     {self.closed_at.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        if self.open_txid:
            lines.append(f"  BSV Open TX:   {self.open_txid}")

        lines += [
            f"  Statement:     {self.statement_hash}",
            "",
            "-" * 72,
            "  REGULATORY CLAIMS",
            "-" * 72,
            "",
        ]

        for claim in self.claims:
            symbol = "✓" if claim.satisfied else "✗"
            lines.append(f"  {symbol}  [{claim.eu_ai_act_reference}]")
            lines.append(f"     {claim.human_description}")
            if claim.detail:
                lines.append(f"     NOTE: {claim.detail}")
            lines.append(f"     Evidence: {claim.evidence_hash}")
            lines.append("")

        if self.aggregate_proof is not None:
            lines += [
                "-" * 72,
                "  ZERO-KNOWLEDGE INFERENCE PROOF",
                "-" * 72,
                "",
                f"  Proving system:  {self.aggregate_proof.aggregation_scheme}",
                f"  Proofs included: {self.aggregate_proof.n_proofs}",
                f"  Proofs root:     {self.aggregate_proof.proofs_merkle_root}",
                f"  Aggregate hash:  {self.aggregate_proof.digest()}",
                "",
                "  What this proves:",
                "    Every listed inference was cryptographically verified.",
                "    No inference can be added, removed, or altered without",
                "    invalidating this proof.",
                "",
            ]
        else:
            lines += [
                "-" * 72,
                "  ZERO-KNOWLEDGE PROOF",
                "-" * 72,
                "",
                "  ZK proving not configured for this epoch.",
                "  Pre-commitment anchoring (BRC-120) provides anti-backdating",
                "  and anti-model-substitution guarantees.",
                "",
            ]

        overall = "ALL CLAIMS SATISFIED" if self.all_satisfied() else "ONE OR MORE CLAIMS NOT SATISFIED"
        lines += [
            "=" * 72,
            f"  OVERALL RESULT:  {overall}",
            f"  STATEMENT HASH:  {self.statement_hash}",
            "=" * 72,
            "",
            "  Verify this epoch at:",
            f"  https://portal.aria-bsv.io/verify/{self.open_txid or '<open_txid>'}",
            "",
            "  This report was generated automatically by the ARIA SDK.",
            "  The statement_hash commits to all claims and proof data.",
            "  Independent verification requires only the two BSV transaction",
            "  IDs and the public verifying key.",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_hash(self) -> str:
        """Canonical SHA-256 of the complete statement."""
        content: dict[str, Any] = {
            "epoch_id": self.epoch_id,
            "system_id": self.system_id,
            "n_records": self.n_records,
            "claims": [c.to_dict() for c in self.claims],
        }
        if self.aggregate_proof is not None:
            content["aggregate_digest"] = self.aggregate_proof.digest()

        msg = canonical_json(content)
        return "sha256:" + hashlib.sha256(msg).hexdigest()
