"""Tests for ARIA smart contract primitives (bonding, notarization, registry)."""

from __future__ import annotations

import hashlib

import pytest

from aria.contracts.bonding import (
    OperatorBondingContract,
    BondRecord,
    BondState,
    _p2pkh_lock_script,
    _p2ms_lock_script,
    pubkey_hash160,
)
from aria.contracts.notarization import (
    EpochNotarization,
    NotarizationPolicy,
    NotarizationSession,
    compute_epoch_commitment,
)
from aria.contracts.registry import (
    ARIARegistry,
    RegistryEntry,
    RegistryEntryType,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

# Minimal deterministic "public key" stubs (33-byte compressed key format)
_PK1 = "02" + "11" * 32  # fake compressed pubkey
_PK2 = "02" + "22" * 32
_PK3 = "03" + "33" * 32

_GOV_PUBKEYS = [_PK1, _PK2, _PK3]


# ---------------------------------------------------------------------------
# Script construction helpers
# ---------------------------------------------------------------------------

class TestP2PKHScript:
    def test_script_starts_with_op_dup(self):
        script = _p2pkh_lock_script(_PK1)
        assert script[0] == 0x76  # OP_DUP

    def test_script_ends_with_op_checksig(self):
        script = _p2pkh_lock_script(_PK1)
        assert script[-1] == 0xAC  # OP_CHECKSIG

    def test_script_contains_hash160(self):
        script = _p2pkh_lock_script(_PK1)
        assert script[1] == 0xA9  # OP_HASH160

    def test_script_length_is_25_bytes(self):
        """P2PKH locking script is always exactly 25 bytes."""
        script = _p2pkh_lock_script(_PK1)
        assert len(script) == 25

    def test_different_keys_produce_different_scripts(self):
        assert _p2pkh_lock_script(_PK1) != _p2pkh_lock_script(_PK2)


class TestP2MSScript:
    def test_2of3_script_starts_with_op_2(self):
        script = _p2ms_lock_script(2, _GOV_PUBKEYS)
        assert script[0] == 0x52  # OP_2

    def test_2of3_script_ends_with_op_checkmultisig(self):
        script = _p2ms_lock_script(2, _GOV_PUBKEYS)
        assert script[-1] == 0xAE  # OP_CHECKMULTISIG

    def test_1of1_script(self):
        script = _p2ms_lock_script(1, [_PK1])
        assert script[0] == 0x51  # OP_1
        assert script[-1] == 0xAE

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            _p2ms_lock_script(0, [_PK1])

    def test_m_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            _p2ms_lock_script(3, [_PK1, _PK2])

    def test_too_many_keys_raises(self):
        with pytest.raises(ValueError):
            _p2ms_lock_script(2, [_PK1, _PK2, _PK3, _PK1])  # n=4 > 3


class TestPubkeyHash160:
    def test_returns_40_char_hex(self):
        h = pubkey_hash160(_PK1)
        assert len(h) == 40
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert pubkey_hash160(_PK1) == pubkey_hash160(_PK1)

    def test_different_keys_differ(self):
        assert pubkey_hash160(_PK1) != pubkey_hash160(_PK2)


# ---------------------------------------------------------------------------
# OperatorBondingContract
# ---------------------------------------------------------------------------

class TestOperatorBondingContract:
    def test_register_bond_succeeds(self):
        contract = OperatorBondingContract(min_bond_satoshis=10_000)
        record = contract.register_bond("op-1", "tx-abc", 0, 50_000, _PK1)
        assert record.operator_id == "op-1"
        assert record.state == BondState.ACTIVE
        assert record.amount_sat == 50_000

    def test_register_bond_below_minimum_raises(self):
        contract = OperatorBondingContract(min_bond_satoshis=10_000)
        with pytest.raises(ValueError, match="minimum"):
            contract.register_bond("op-1", "tx-abc", 0, 5_000, _PK1)

    def test_is_bonded_returns_true_for_active(self):
        contract = OperatorBondingContract(min_bond_satoshis=10_000)
        contract.register_bond("op-1", "tx-abc", 0, 50_000, _PK1)
        assert contract.is_bonded("op-1")

    def test_is_bonded_returns_false_for_unknown(self):
        contract = OperatorBondingContract()
        assert not contract.is_bonded("nobody")

    def test_is_bonded_returns_false_for_slashed(self):
        contract = OperatorBondingContract(min_bond_satoshis=1_000)
        contract.register_bond("op-1", "tx", 0, 5_000, _PK1)
        contract.slash_bond("op-1", "misbehaviour")
        assert not contract.is_bonded("op-1")

    def test_slash_bond_marks_state(self):
        contract = OperatorBondingContract(min_bond_satoshis=1_000)
        contract.register_bond("op-1", "tx", 0, 5_000, _PK1)
        record = contract.slash_bond("op-1", "double-sign", slash_txid="tx-slash")
        assert record.state == BondState.SLASHED
        assert record.slash_reason == "double-sign"
        assert record.slash_txid == "tx-slash"
        assert record.slashed_at is not None

    def test_slash_missing_operator_raises(self):
        contract = OperatorBondingContract()
        with pytest.raises(KeyError):
            contract.slash_bond("nobody", "reason")

    def test_slash_already_slashed_raises(self):
        contract = OperatorBondingContract(min_bond_satoshis=1_000)
        contract.register_bond("op-1", "tx", 0, 5_000, _PK1)
        contract.slash_bond("op-1", "first")
        with pytest.raises(ValueError, match="SLASHED"):
            contract.slash_bond("op-1", "second")

    def test_bond_locking_script_is_p2pkh(self):
        contract = OperatorBondingContract(min_bond_satoshis=1_000)
        contract.register_bond("op-1", "tx", 0, 5_000, _PK1)
        script = contract.bond_locking_script("op-1")
        assert script[0] == 0x76  # OP_DUP
        assert len(script) == 25

    def test_slash_authorization_script_is_2of3_multisig(self):
        contract = OperatorBondingContract(governance_pubkeys=_GOV_PUBKEYS)
        script = contract.slash_authorization_script()
        assert script[0] == 0x52  # OP_2
        assert script[-1] == 0xAE  # OP_CHECKMULTISIG

    def test_slash_auth_without_gov_keys_raises(self):
        contract = OperatorBondingContract()
        with pytest.raises(ValueError, match="governance pubkeys"):
            contract.slash_authorization_script()

    def test_active_bonds_returns_only_active(self):
        contract = OperatorBondingContract(min_bond_satoshis=1_000)
        contract.register_bond("op-1", "tx1", 0, 5_000, _PK1)
        contract.register_bond("op-2", "tx2", 0, 5_000, _PK2)
        contract.slash_bond("op-1", "reason")
        active = contract.active_bonds()
        assert len(active) == 1
        assert active[0].operator_id == "op-2"


# ---------------------------------------------------------------------------
# EpochNotarization
# ---------------------------------------------------------------------------

class TestEpochCommitment:
    def test_commitment_is_32_bytes(self):
        c = compute_epoch_commitment("ep-1", "a" * 64, 10)
        assert len(c) == 32

    def test_commitment_is_deterministic(self):
        c1 = compute_epoch_commitment("ep-1", "a" * 64, 10)
        c2 = compute_epoch_commitment("ep-1", "a" * 64, 10)
        assert c1 == c2

    def test_commitment_differs_on_different_inputs(self):
        c1 = compute_epoch_commitment("ep-1", "a" * 64, 10)
        c2 = compute_epoch_commitment("ep-1", "b" * 64, 10)
        assert c1 != c2


class TestEpochNotarization:
    def _make_policy(self, m=2):
        return NotarizationPolicy(m=m, notary_pubkeys=_GOV_PUBKEYS)

    def test_policy_validation(self):
        with pytest.raises(ValueError):
            NotarizationPolicy(m=4, notary_pubkeys=_GOV_PUBKEYS)  # m > n

    def test_open_session(self):
        n = EpochNotarization(self._make_policy())
        session = n.open_session("ep-n1", "a" * 64, 5)
        assert session.epoch_id == "ep-n1"
        assert len(session.commitment) == 32

    def test_open_session_idempotent(self):
        n = EpochNotarization(self._make_policy())
        s1 = n.open_session("ep-n1", "a" * 64, 5)
        s2 = n.open_session("ep-n1", "a" * 64, 5)
        assert s1 is s2

    def test_add_signature_records(self):
        n = EpochNotarization(self._make_policy())
        n.open_session("ep-n2", "b" * 64, 3)
        n.add_signature("ep-n2", "op-1", _PK1, "sig1hex")
        session = n.get_session("ep-n2")
        assert len(session.signatures) == 1

    def test_is_authorized_after_m_signatures(self):
        n = EpochNotarization(self._make_policy(m=2))
        n.open_session("ep-n3", "c" * 64, 7)
        assert not n.is_authorized("ep-n3")
        n.add_signature("ep-n3", "op-1", _PK1, "sig1")
        assert not n.is_authorized("ep-n3")
        n.add_signature("ep-n3", "op-2", _PK2, "sig2")
        assert n.is_authorized("ep-n3")

    def test_signature_not_in_policy_raises(self):
        n = EpochNotarization(self._make_policy())
        n.open_session("ep-n4", "d" * 64, 1)
        outsider_pk = "04" + "ff" * 32
        with pytest.raises(ValueError, match="not in the notarization policy"):
            n.add_signature("ep-n4", "outsider", outsider_pk, "sig")

    def test_duplicate_signature_raises(self):
        n = EpochNotarization(self._make_policy())
        n.open_session("ep-n5", "e" * 64, 1)
        n.add_signature("ep-n5", "op-1", _PK1, "sig1")
        with pytest.raises(ValueError, match="already signed"):
            n.add_signature("ep-n5", "op-1", _PK1, "sig1")

    def test_add_signature_missing_session_raises(self):
        n = EpochNotarization(self._make_policy())
        with pytest.raises(KeyError):
            n.add_signature("nonexistent", "op-1", _PK1, "sig")

    def test_get_notarization_payload_structure(self):
        n = EpochNotarization(self._make_policy(m=1))
        n.open_session("ep-n6", "f" * 64, 99)
        n.add_signature("ep-n6", "op-1", _PK1, "sig1")
        payload = n.get_notarization_payload("ep-n6")
        assert payload[:8] == b"ARIA-NOT"
        assert payload[8] == 0x01  # version

    def test_get_payload_unauthorized_raises(self):
        n = EpochNotarization(self._make_policy(m=2))
        n.open_session("ep-n7", "a" * 64, 1)
        n.add_signature("ep-n7", "op-1", _PK1, "sig1")  # only 1 of 2
        with pytest.raises(ValueError, match="not yet authorized"):
            n.get_notarization_payload("ep-n7")

    def test_session_finalized_at_set_when_authorized(self):
        n = EpochNotarization(self._make_policy(m=2))
        n.open_session("ep-n8", "a" * 64, 1)
        n.add_signature("ep-n8", "op-1", _PK1, "sig1")
        assert n.get_session("ep-n8").finalized_at is None
        n.add_signature("ep-n8", "op-2", _PK2, "sig2")
        assert n.get_session("ep-n8").finalized_at is not None

    def test_multisig_lock_script_is_correct(self):
        n = EpochNotarization(self._make_policy(m=2))
        script = n.multisig_lock_script()
        assert script[0] == 0x52  # OP_2
        assert script[-1] == 0xAE  # OP_CHECKMULTISIG


# ---------------------------------------------------------------------------
# ARIARegistry
# ---------------------------------------------------------------------------

class TestARIARegistry:
    def test_build_system_entry_is_valid(self):
        r = ARIARegistry()
        payload = r.build_system_entry("sys-alpha", _PK1)
        assert payload[:8] == b"ARIA-REG"
        assert payload[8] == 0x01  # version
        assert payload[9] == 0x01  # SYSTEM

    def test_build_model_entry_is_valid(self):
        r = ARIARegistry()
        payload = r.build_model_entry("model-v2", "a" * 64)
        assert payload[:8] == b"ARIA-REG"
        assert payload[9] == 0x02  # MODEL

    def test_build_operator_entry_is_valid(self):
        r = ARIARegistry()
        payload = r.build_operator_entry("op-zeus", _PK2)
        assert payload[:8] == b"ARIA-REG"
        assert payload[9] == 0x03  # OPERATOR

    def test_entries_indexed_locally(self):
        r = ARIARegistry()
        r.build_system_entry("sys-beta", _PK1)
        assert r.is_system_registered("sys-beta")
        assert not r.is_system_registered("sys-gamma")

    def test_get_system_returns_entry(self):
        r = ARIARegistry()
        r.build_system_entry("sys-delta", _PK1)
        entry = r.get_system("sys-delta")
        assert entry is not None
        assert entry.system_id == "sys-delta"

    def test_get_model_returns_entry(self):
        r = ARIARegistry()
        r.build_model_entry("mdl-1", "b" * 64)
        entry = r.get_model("mdl-1")
        assert entry is not None
        assert entry.model_hash == "b" * 64

    def test_parse_entry_system_round_trip(self):
        r = ARIARegistry()
        payload = r.build_system_entry("sys-rt", _PK1)
        decoded = ARIARegistry.parse_entry(payload, txid="tx-abc")
        assert decoded is not None
        assert decoded.entry_type == RegistryEntryType.SYSTEM
        assert decoded.system_id == "sys-rt"
        assert decoded.txid == "tx-abc"

    def test_parse_entry_model_round_trip(self):
        r = ARIARegistry()
        payload = r.build_model_entry("mdl-rt", "c" * 64)
        decoded = ARIARegistry.parse_entry(payload)
        assert decoded is not None
        assert decoded.entry_type == RegistryEntryType.MODEL
        assert decoded.model_hash == "c" * 64

    def test_parse_entry_operator_round_trip(self):
        r = ARIARegistry()
        payload = r.build_operator_entry("op-rt", _PK3)
        decoded = ARIARegistry.parse_entry(payload)
        assert decoded is not None
        assert decoded.entry_type == RegistryEntryType.OPERATOR
        assert decoded.operator_id == "op-rt"

    def test_parse_entry_wrong_magic_returns_none(self):
        assert ARIARegistry.parse_entry(b"BADMAGIC" + b"\x00" * 20) is None

    def test_parse_entry_short_data_returns_none(self):
        assert ARIARegistry.parse_entry(b"ARIA-REG") is None

    def test_parse_entry_unknown_type_returns_none(self):
        bad = b"ARIA-REG" + bytes([0x01, 0xFF]) + b"\x00" * 64  # 0xFF is not a valid entry type
        assert ARIARegistry.parse_entry(bad) is None

    def test_model_entry_with_sha256_prefix(self):
        r = ARIARegistry()
        payload = r.build_model_entry("mdl-pfx", "sha256:" + "d" * 64)
        # Should not raise; prefix is stripped
        decoded = ARIARegistry.parse_entry(payload)
        assert decoded is not None
        assert decoded.model_hash == "d" * 64
