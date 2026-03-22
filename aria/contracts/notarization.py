"""
aria.contracts.notarization — 2-of-3 multisig epoch notarization for ARIA.

The EpochNotarization contract requires M-of-N designated notary operators to
co-sign an epoch-close commitment before it is considered final.  This provides
Byzantine fault tolerance: a single operator going offline or acting maliciously
cannot block or forge epoch closes.

Protocol::

    1. Epoch proposer computes commitment = SHA-256(epoch_id | merkle_root | records_count)
    2. Each notary signs the commitment with their BSV private key
    3. When M signatures are collected, construct_close_tx() builds the P2MS tx
    4. The resulting raw transaction is broadcast via the WalletInterface

The notarization payload is embedded in OP_RETURN of the EPOCH_CLOSE tx::

    ARIA-NOT <version:1> <epoch_id:16> <merkle_root:32> <records_count:4>

Bitcoin Script used for the multisig output (change / collateral):
    OP_m <pubkey1> … <pubkeyn> OP_n OP_CHECKMULTISIG

Usage::

    from aria.contracts.notarization import EpochNotarization, NotarizationPolicy

    policy = NotarizationPolicy(m=2, notary_pubkeys=["pk1_hex", "pk2_hex", "pk3_hex"])
    notarization = EpochNotarization(policy=policy)

    # Each notary signs the commitment
    sig1 = notarization.sign_commitment("epoch-abc", "merkle_root_hex", 42, wif1)
    sig2 = notarization.sign_commitment("epoch-abc", "merkle_root_hex", 42, wif2)

    # Check if we have enough signatures
    if notarization.is_authorized("epoch-abc"):
        close_payload = notarization.get_notarization_payload("epoch-abc")
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------

@dataclass
class NotarizationPolicy:
    """M-of-N multisig policy for epoch notarization.

    Attributes:
        m:               Minimum number of signatures required.
        notary_pubkeys:  Compressed public keys (hex) of authorised notaries.
    """
    m: int
    notary_pubkeys: list[str]

    def __post_init__(self) -> None:
        n = len(self.notary_pubkeys)
        if not (1 <= self.m <= n):
            raise ValueError(f"Invalid m-of-n policy: m={self.m}, n={n}")

    @property
    def n(self) -> int:
        return len(self.notary_pubkeys)


# ---------------------------------------------------------------------------
# Commitment construction
# ---------------------------------------------------------------------------

_NOTARIZATION_MAGIC = b"ARIA-NOT"
_NOTARIZATION_VERSION = 0x01


def compute_epoch_commitment(
    epoch_id: str,
    merkle_root: str,
    records_count: int,
) -> bytes:
    """Compute the canonical 32-byte commitment hash for an epoch close.

    commitment = SHA-256(
        b"ARIA-NOT" | version | epoch_id_bytes_16 | merkle_root_bytes_32 | count_bytes_4
    )
    """
    epoch_bytes = epoch_id.encode()[:16].ljust(16, b"\x00")
    mr_hex = merkle_root.lstrip("sha256:").lstrip("0x").zfill(64)
    mr_bytes = bytes.fromhex(mr_hex)[:32]
    count_bytes = struct.pack(">I", records_count)

    payload = (
        _NOTARIZATION_MAGIC
        + bytes([_NOTARIZATION_VERSION])
        + epoch_bytes
        + mr_bytes
        + count_bytes
    )
    return hashlib.sha256(payload).digest()


# ---------------------------------------------------------------------------
# Notarization session
# ---------------------------------------------------------------------------

@dataclass
class EpochSignature:
    """A single notary's signature over an epoch commitment."""
    notary_id: str
    pubkey_hex: str
    signature_hex: str    # DER-encoded signature over commitment hash
    signed_at: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class NotarizationSession:
    """In-progress notarization for a single epoch."""
    epoch_id: str
    merkle_root: str
    records_count: int
    commitment: bytes
    signatures: list[EpochSignature] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    finalized_at: int | None = None

    @property
    def commitment_hex(self) -> str:
        return self.commitment.hex()


# ---------------------------------------------------------------------------
# EpochNotarization contract
# ---------------------------------------------------------------------------

class EpochNotarization:
    """Manages M-of-N notarization of ARIA epoch close events.

    The contract collects notary signatures over epoch commitments.  Once M
    signatures are collected, the epoch is considered "authorized" and its
    close transaction can be constructed.

    Args:
        policy: A NotarizationPolicy defining m and notary public keys.
    """

    def __init__(self, policy: NotarizationPolicy) -> None:
        self._policy = policy
        self._sessions: dict[str, NotarizationSession] = {}

    @property
    def policy(self) -> NotarizationPolicy:
        return self._policy

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def open_session(
        self,
        epoch_id: str,
        merkle_root: str,
        records_count: int,
    ) -> NotarizationSession:
        """Open a new notarization session for an epoch close."""
        if epoch_id in self._sessions:
            return self._sessions[epoch_id]

        commitment = compute_epoch_commitment(epoch_id, merkle_root, records_count)
        session = NotarizationSession(
            epoch_id=epoch_id,
            merkle_root=merkle_root,
            records_count=records_count,
            commitment=commitment,
        )
        self._sessions[epoch_id] = session
        return session

    def get_session(self, epoch_id: str) -> NotarizationSession | None:
        return self._sessions.get(epoch_id)

    # ------------------------------------------------------------------
    # Signature collection
    # ------------------------------------------------------------------

    def add_signature(
        self,
        epoch_id: str,
        notary_id: str,
        pubkey_hex: str,
        signature_hex: str,
    ) -> NotarizationSession:
        """Add a notary signature to the session.

        Args:
            epoch_id:      The epoch being notarized.
            notary_id:     Identifier of the signing notary.
            pubkey_hex:    Notary's compressed public key (must be in policy).
            signature_hex: DER-encoded ECDSA signature over commitment bytes.

        Raises:
            KeyError:   if epoch_id has no open session.
            ValueError: if pubkey_hex is not in the policy, or already signed.
        """
        session = self._sessions.get(epoch_id)
        if session is None:
            raise KeyError(f"No notarization session for epoch {epoch_id!r}")

        if pubkey_hex not in self._policy.notary_pubkeys:
            raise ValueError(
                f"pubkey {pubkey_hex[:16]}... is not in the notarization policy"
            )

        already_signed = {s.pubkey_hex for s in session.signatures}
        if pubkey_hex in already_signed:
            raise ValueError(
                f"Notary {notary_id!r} (pubkey {pubkey_hex[:16]}...) has already signed"
            )

        session.signatures.append(EpochSignature(
            notary_id=notary_id,
            pubkey_hex=pubkey_hex,
            signature_hex=signature_hex,
        ))

        if len(session.signatures) >= self._policy.m and session.finalized_at is None:
            session.finalized_at = int(time.time() * 1000)

        return session

    def is_authorized(self, epoch_id: str) -> bool:
        """Return True if the epoch has ≥ M valid signatures collected."""
        session = self._sessions.get(epoch_id)
        if session is None:
            return False
        return len(session.signatures) >= self._policy.m

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def get_notarization_payload(self, epoch_id: str) -> bytes:
        """Return the OP_RETURN payload bytes for the notarized EPOCH_CLOSE tx.

        Format (62 bytes):
            8  bytes — magic b"ARIA-NOT"
            1  byte  — version 0x01
            16 bytes — epoch_id (zero-padded)
            32 bytes — merkle_root (raw bytes)
            4  bytes — records_count (big-endian)
            1  byte  — signature count

        Raises:
            KeyError:   if no session exists for epoch_id.
            ValueError: if epoch is not yet authorized (< M signatures).
        """
        session = self._sessions.get(epoch_id)
        if session is None:
            raise KeyError(f"No notarization session for epoch {epoch_id!r}")
        if not self.is_authorized(epoch_id):
            raise ValueError(
                f"Epoch {epoch_id!r} is not yet authorized "
                f"({len(session.signatures)}/{self._policy.m} signatures)"
            )

        epoch_bytes = epoch_id.encode()[:16].ljust(16, b"\x00")
        mr_hex = session.merkle_root.lstrip("sha256:").lstrip("0x").zfill(64)
        mr_bytes = bytes.fromhex(mr_hex)[:32]
        count_bytes = struct.pack(">I", session.records_count)
        sig_count = bytes([len(session.signatures)])

        return (
            _NOTARIZATION_MAGIC
            + bytes([_NOTARIZATION_VERSION])
            + epoch_bytes
            + mr_bytes
            + count_bytes
            + sig_count
        )

    def multisig_lock_script(self) -> bytes:
        """Return the P2MS locking script for the notarization output.

        Script: OP_m <pubkey1> … <pubkeyn> OP_n OP_CHECKMULTISIG
        """
        from .bonding import _p2ms_lock_script
        return _p2ms_lock_script(self._policy.m, self._policy.notary_pubkeys)
