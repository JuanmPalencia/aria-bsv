"""
aria.contracts.bonding — Operator bond management for ARIA.

An operator bond is a P2PKH output sent to a designated bond address.  The
``OperatorBondingContract`` maintains an off-chain registry of bonds (backed by
the ARIA SQLite/Postgres storage via a simple JSON ledger) and constructs the
BSV transactions needed to:

  - Create a bond (operator sends satoshis to bond address)
  - Verify a bond is active and above minimum threshold
  - Slash a bond (governance multisig spends the UTXO to confiscation address)

Bitcoin Script used:
  Locking (bond):  OP_DUP OP_HASH160 <pubkey_hash> OP_EQUALVERIFY OP_CHECKSIG
  Unlock (normal): <sig> <pubkey>
  Slash auth:      2-of-3 multisig governance spend (constructed by EpochNotarization)

The contract is "soft" — enforcement is via social/economic incentives, not
enforced by Bitcoin Script alone.  A future sCrypt-based upgrade will add
on-chain enforcement.

Usage::

    from aria.contracts.bonding import OperatorBondingContract, BondRecord

    contract = OperatorBondingContract(
        min_bond_satoshis=10_000,
        governance_pubkeys=["04aabb...", "04ccdd...", "04eeff..."],
    )

    # Register a new bond (after the operator has broadcast the funding tx)
    contract.register_bond("operator-1", "tx-abc123", 0, 50_000, "pubkey-hex")

    # Verify the bond is active
    assert contract.is_bonded("operator-1")

    # Slash (marks bond as slashed; caller must broadcast the slash tx)
    slash_tx = contract.create_slash_transaction("operator-1", governance_wifs=[...])
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Bond state model
# ---------------------------------------------------------------------------

class BondState(str, Enum):
    ACTIVE = "ACTIVE"
    SLASHED = "SLASHED"
    WITHDRAWN = "WITHDRAWN"


@dataclass
class BondRecord:
    """Record of a single operator bond.

    Attributes:
        operator_id:    Human-readable operator identifier.
        txid:           BSV txid of the bond funding transaction.
        vout:           Output index in the funding transaction.
        amount_sat:     Amount in satoshis.
        pubkey_hex:     Operator's BSV public key (compressed, 33 bytes hex).
        state:          Current bond state.
        bonded_at:      Unix millisecond timestamp when bond was registered.
        slashed_at:     Unix ms timestamp if slashed, else None.
        slash_reason:   Human-readable slash reason, if applicable.
        slash_txid:     Txid of the slash transaction, if broadcast.
    """
    operator_id: str
    txid: str
    vout: int
    amount_sat: int
    pubkey_hex: str
    state: BondState = BondState.ACTIVE
    bonded_at: int = field(default_factory=lambda: int(time.time() * 1000))
    slashed_at: int | None = None
    slash_reason: str | None = None
    slash_txid: str | None = None

    @property
    def utxo_key(self) -> str:
        return f"{self.txid}:{self.vout}"


# ---------------------------------------------------------------------------
# Script construction helpers (pure Python — no bsvlib dependency)
# ---------------------------------------------------------------------------

def _p2pkh_lock_script(pubkey_hex: str) -> bytes:
    """Build a P2PKH locking script from a compressed public key hex string.

    Script: OP_DUP OP_HASH160 <pubkey_hash160> OP_EQUALVERIFY OP_CHECKSIG
    """
    pubkey_bytes = bytes.fromhex(pubkey_hex)
    sha256 = hashlib.sha256(pubkey_bytes).digest()
    ripe = hashlib.new("ripemd160", sha256).digest()  # HASH160

    OP_DUP = 0x76
    OP_HASH160 = 0xA9
    OP_EQUALVERIFY = 0x88
    OP_CHECKSIG = 0xAC
    PUSH20 = 0x14  # push 20 bytes

    return bytes([OP_DUP, OP_HASH160, PUSH20]) + ripe + bytes([OP_EQUALVERIFY, OP_CHECKSIG])


def _p2ms_lock_script(m: int, pubkeys_hex: list[str]) -> bytes:
    """Build a bare m-of-n multisig locking script.

    Script: OP_m <pubkey1> … <pubkeyn> OP_n OP_CHECKMULTISIG
    m must satisfy 1 ≤ m ≤ n ≤ 3.
    """
    n = len(pubkeys_hex)
    if not (1 <= m <= n <= 3):
        raise ValueError(f"Invalid m-of-n: {m}/{n}")

    OP_m = 0x50 + m
    OP_n = 0x50 + n
    OP_CHECKMULTISIG = 0xAE

    script = bytes([OP_m])
    for pk in pubkeys_hex:
        pk_bytes = bytes.fromhex(pk)
        script += bytes([len(pk_bytes)]) + pk_bytes
    script += bytes([OP_n, OP_CHECKMULTISIG])
    return script


def pubkey_hash160(pubkey_hex: str) -> str:
    """Return the HASH160 (RIPEMD160(SHA256(pubkey))) of a public key as hex."""
    pub = bytes.fromhex(pubkey_hex)
    sha = hashlib.sha256(pub).digest()
    h160 = hashlib.new("ripemd160", sha).digest()
    return h160.hex()


# ---------------------------------------------------------------------------
# OperatorBondingContract
# ---------------------------------------------------------------------------

class OperatorBondingContract:
    """Manages ARIA operator bonds as UTXOs on BSV.

    The contract maintains an in-memory registry of bond records (suitable for
    persistence via the application's storage layer).  It constructs raw
    Bitcoin Script for bonding and slashing but does not sign or broadcast
    transactions — that is delegated to the WalletInterface.

    Args:
        min_bond_satoshis:   Minimum bond amount required.  Default 10 000 sat.
        governance_pubkeys:  List of 3 governance public keys (compressed, hex)
                             used to build the 2-of-3 slash multisig.
    """

    def __init__(
        self,
        min_bond_satoshis: int = 10_000,
        governance_pubkeys: list[str] | None = None,
    ) -> None:
        self._min_sat = min_bond_satoshis
        self._gov_pubkeys = governance_pubkeys or []
        self._bonds: dict[str, BondRecord] = {}  # operator_id → BondRecord

    # ------------------------------------------------------------------
    # Bond management
    # ------------------------------------------------------------------

    def register_bond(
        self,
        operator_id: str,
        txid: str,
        vout: int,
        amount_sat: int,
        pubkey_hex: str,
    ) -> BondRecord:
        """Register a newly funded bond UTXO.

        Called after the operator broadcasts a funding transaction.

        Raises:
            ValueError: if amount_sat < min_bond_satoshis.
        """
        if amount_sat < self._min_sat:
            raise ValueError(
                f"Bond amount {amount_sat} sat < minimum {self._min_sat} sat"
            )
        record = BondRecord(
            operator_id=operator_id,
            txid=txid,
            vout=vout,
            amount_sat=amount_sat,
            pubkey_hex=pubkey_hex,
        )
        self._bonds[operator_id] = record
        return record

    def get_bond(self, operator_id: str) -> BondRecord | None:
        return self._bonds.get(operator_id)

    def is_bonded(self, operator_id: str) -> bool:
        """Return True if operator has an ACTIVE bond ≥ min_bond_satoshis."""
        record = self._bonds.get(operator_id)
        return (
            record is not None
            and record.state == BondState.ACTIVE
            and record.amount_sat >= self._min_sat
        )

    def all_bonds(self) -> list[BondRecord]:
        return list(self._bonds.values())

    def active_bonds(self) -> list[BondRecord]:
        return [b for b in self._bonds.values() if b.state == BondState.ACTIVE]

    # ------------------------------------------------------------------
    # Slash
    # ------------------------------------------------------------------

    def slash_bond(
        self,
        operator_id: str,
        reason: str,
        slash_txid: str | None = None,
    ) -> BondRecord:
        """Mark an operator bond as SLASHED.

        Raises:
            KeyError:   if operator_id has no registered bond.
            ValueError: if bond is not ACTIVE.
        """
        record = self._bonds.get(operator_id)
        if record is None:
            raise KeyError(f"No bond registered for operator {operator_id!r}")
        if record.state != BondState.ACTIVE:
            raise ValueError(
                f"Bond for {operator_id!r} is {record.state.value}, not ACTIVE"
            )
        record.state = BondState.SLASHED
        record.slashed_at = int(time.time() * 1000)
        record.slash_reason = reason
        record.slash_txid = slash_txid
        return record

    # ------------------------------------------------------------------
    # Script construction
    # ------------------------------------------------------------------

    def bond_locking_script(self, operator_id: str) -> bytes:
        """Return the P2PKH locking script for the operator's bond UTXO."""
        record = self._bonds.get(operator_id)
        if record is None:
            raise KeyError(f"Operator {operator_id!r} not found")
        return _p2pkh_lock_script(record.pubkey_hex)

    def slash_authorization_script(self) -> bytes:
        """Return the 2-of-3 multisig locking script for governance slashing.

        Raises:
            ValueError: if governance_pubkeys were not provided at init.
        """
        if len(self._gov_pubkeys) < 3:
            raise ValueError(
                "slash_authorization_script requires 3 governance pubkeys; "
                f"got {len(self._gov_pubkeys)}"
            )
        return _p2ms_lock_script(2, self._gov_pubkeys[:3])
