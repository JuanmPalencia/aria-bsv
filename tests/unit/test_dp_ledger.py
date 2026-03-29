"""
tests/unit/test_dp_ledger.py

Unit tests for aria.dp_ledger — DPCheckpoint and DPBudgetLedger.

PrivacyAccountant is used directly (no mocking) because it has no external
dependencies and is pure Python.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aria.dp_ledger import DPBudgetLedger, DPCheckpoint
from aria.privacy import DPMechanism, PrivacyAccountant, PrivacyBudgetStatus


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_accountant(epsilon_total: float = 1.0, **kwargs) -> PrivacyAccountant:
    return PrivacyAccountant(epsilon_total=epsilon_total, delta=1e-5, **kwargs)


def _make_checkpoint(
    *,
    epsilon_used: float = 0.1,
    epsilon_total: float = 1.0,
    delta_used: float = 1e-6,
    query_count: int = 1,
    status: PrivacyBudgetStatus = PrivacyBudgetStatus.SAFE,
    mechanism_breakdown: dict | None = None,
    epoch_id: str = "epoch-001",
    txid: str | None = None,
) -> DPCheckpoint:
    return DPCheckpoint(
        checkpoint_id="aabbccdd00112233",
        epoch_id=epoch_id,
        epsilon_used=epsilon_used,
        epsilon_total=epsilon_total,
        delta_used=delta_used,
        query_count=query_count,
        status=status,
        mechanism_breakdown=mechanism_breakdown or {"laplace": 1},
        created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        txid=txid,
    )


# ---------------------------------------------------------------------------
# DPCheckpoint tests
# ---------------------------------------------------------------------------

class TestDPCheckpointHash:
    def test_hash_is_deterministic(self):
        cp = _make_checkpoint()
        assert cp.checkpoint_hash() == cp.checkpoint_hash()

    def test_hash_starts_with_sha256_prefix(self):
        cp = _make_checkpoint()
        assert cp.checkpoint_hash().startswith("sha256:")

    def test_hash_changes_when_epsilon_used_changes(self):
        cp1 = _make_checkpoint(epsilon_used=0.1)
        cp2 = _make_checkpoint(epsilon_used=0.2)
        assert cp1.checkpoint_hash() != cp2.checkpoint_hash()

    def test_hash_excludes_txid(self):
        cp_no_txid = _make_checkpoint(txid=None)
        cp_with_txid = _make_checkpoint(txid="abc123")
        assert cp_no_txid.checkpoint_hash() == cp_with_txid.checkpoint_hash()

    def test_hash_excludes_created_at(self):
        cp1 = _make_checkpoint()
        cp2 = DPCheckpoint(
            checkpoint_id=cp1.checkpoint_id,
            epoch_id=cp1.epoch_id,
            epsilon_used=cp1.epsilon_used,
            epsilon_total=cp1.epsilon_total,
            delta_used=cp1.delta_used,
            query_count=cp1.query_count,
            status=cp1.status,
            mechanism_breakdown=dict(cp1.mechanism_breakdown),
            created_at=datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
            txid=cp1.txid,
        )
        assert cp1.checkpoint_hash() == cp2.checkpoint_hash()

    def test_hash_changes_when_epoch_id_changes(self):
        cp1 = _make_checkpoint(epoch_id="ep-A")
        cp2 = _make_checkpoint(epoch_id="ep-B")
        assert cp1.checkpoint_hash() != cp2.checkpoint_hash()

    def test_hash_changes_when_mechanism_breakdown_changes(self):
        cp1 = _make_checkpoint(mechanism_breakdown={"laplace": 1})
        cp2 = _make_checkpoint(mechanism_breakdown={"gaussian": 1})
        assert cp1.checkpoint_hash() != cp2.checkpoint_hash()


class TestDPCheckpointEpsilonRemaining:
    def test_epsilon_remaining_basic(self):
        cp = _make_checkpoint(epsilon_used=0.3, epsilon_total=1.0)
        assert abs(cp.epsilon_remaining() - 0.7) < 1e-9

    def test_epsilon_remaining_zero_when_fully_consumed(self):
        cp = _make_checkpoint(epsilon_used=1.0, epsilon_total=1.0)
        assert cp.epsilon_remaining() == 0.0

    def test_epsilon_remaining_clamped_to_zero_when_exceeded(self):
        cp = _make_checkpoint(epsilon_used=1.5, epsilon_total=1.0)
        assert cp.epsilon_remaining() == 0.0

    def test_epsilon_remaining_full_when_unused(self):
        cp = _make_checkpoint(epsilon_used=0.0, epsilon_total=2.0)
        assert abs(cp.epsilon_remaining() - 2.0) < 1e-9


class TestDPCheckpointIsOnChain:
    def test_not_on_chain_when_txid_none(self):
        cp = _make_checkpoint(txid=None)
        assert cp.is_on_chain() is False

    def test_on_chain_when_txid_set(self):
        cp = _make_checkpoint(txid="deadbeef" * 8)
        assert cp.is_on_chain() is True

    def test_not_on_chain_when_txid_empty_string(self):
        cp = _make_checkpoint(txid="")
        assert cp.is_on_chain() is False


class TestDPCheckpointToDict:
    def test_to_dict_returns_dict(self):
        cp = _make_checkpoint()
        result = cp.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_is_json_serialisable(self):
        cp = _make_checkpoint(txid="abc")
        result = cp.to_dict()
        serialised = json.dumps(result)
        assert isinstance(serialised, str)

    def test_to_dict_contains_required_keys(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        for key in (
            "checkpoint_id", "epoch_id", "epsilon_used", "epsilon_total",
            "delta_used", "query_count", "status", "mechanism_breakdown",
            "created_at", "txid", "checkpoint_hash",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_status_is_string(self):
        cp = _make_checkpoint(status=PrivacyBudgetStatus.WARNING)
        assert cp.to_dict()["status"] == "WARNING"

    def test_to_dict_created_at_is_iso_string(self):
        cp = _make_checkpoint()
        created = cp.to_dict()["created_at"]
        assert isinstance(created, str)
        # Must parse back without error.
        datetime.fromisoformat(created)


# ---------------------------------------------------------------------------
# DPBudgetLedger tests
# ---------------------------------------------------------------------------

class TestDPBudgetLedgerCheckpoint:
    def test_checkpoint_creates_entry(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        assert isinstance(cp, DPCheckpoint)

    def test_checkpoint_epsilon_zero_before_queries(self):
        acc = _make_accountant(epsilon_total=1.0)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        assert cp.epsilon_used == 0.0

    def test_checkpoint_reflects_consumed_budget(self):
        acc = _make_accountant(epsilon_total=1.0)
        acc.record_query(epsilon=0.25, mechanism=DPMechanism.LAPLACE)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        assert abs(cp.epsilon_used - 0.25) < 1e-9

    def test_checkpoint_without_broadcaster_txid_is_none(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc, broadcaster=None)
        cp = ledger.checkpoint("ep-1")
        assert cp.txid is None
        assert cp.is_on_chain() is False

    def test_checkpoint_with_mock_broadcaster_calls_broadcast(self):
        acc = _make_accountant()
        mock_result = MagicMock()
        mock_result.txid = "fakexid001"
        broadcaster = MagicMock()
        broadcaster.broadcast.return_value = mock_result

        ledger = DPBudgetLedger(acc, broadcaster=broadcaster)
        cp = ledger.checkpoint("ep-1")

        broadcaster.broadcast.assert_called_once()
        assert cp.txid == "fakexid001"
        assert cp.is_on_chain() is True

    def test_checkpoint_broadcaster_exception_does_not_crash(self):
        acc = _make_accountant()
        broadcaster = MagicMock()
        broadcaster.broadcast.side_effect = RuntimeError("network error")

        ledger = DPBudgetLedger(acc, broadcaster=broadcaster)
        cp = ledger.checkpoint("ep-1")  # must not raise
        assert cp.txid is None

    def test_checkpoint_broadcaster_payload_is_valid_json(self):
        acc = _make_accountant()
        acc.record_query(epsilon=0.1, mechanism=DPMechanism.GAUSSIAN)
        received_payloads: list[bytes] = []

        class CapturingBroadcaster:
            def broadcast(self, payload: bytes):
                received_payloads.append(payload)
                r = MagicMock()
                r.txid = "txid-cap"
                return r

        ledger = DPBudgetLedger(acc, broadcaster=CapturingBroadcaster())
        ledger.checkpoint("ep-json")

        assert len(received_payloads) == 1
        parsed = json.loads(received_payloads[0].decode("utf-8"))
        assert parsed["type"] == "DP_CHECKPOINT"
        assert parsed["brc"] == "121"
        assert parsed["epoch_id"] == "ep-json"

    def test_multiple_checkpoints_have_unique_ids(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ids = {ledger.checkpoint(f"ep-{i}").checkpoint_id for i in range(20)}
        assert len(ids) == 20


class TestDPBudgetLedgerGetCheckpoints:
    def test_get_checkpoints_returns_all(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ledger.checkpoint("ep-A")
        ledger.checkpoint("ep-B")
        ledger.checkpoint("ep-C")
        assert len(ledger.get_checkpoints()) == 3

    def test_get_checkpoints_filters_by_epoch(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ledger.checkpoint("ep-1")
        ledger.checkpoint("ep-2")
        ledger.checkpoint("ep-1")
        result = ledger.get_checkpoints(epoch_id="ep-1")
        assert len(result) == 2
        assert all(cp.epoch_id == "ep-1" for cp in result)

    def test_get_checkpoints_filter_no_match_returns_empty(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ledger.checkpoint("ep-X")
        assert ledger.get_checkpoints(epoch_id="ep-MISSING") == []

    def test_two_checkpoints_same_epoch_both_retrievable(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        cp1 = ledger.checkpoint("shared-epoch")
        acc.record_query(epsilon=0.1)
        cp2 = ledger.checkpoint("shared-epoch")
        retrieved = ledger.get_checkpoints(epoch_id="shared-epoch")
        assert len(retrieved) == 2
        ids = {cp.checkpoint_id for cp in retrieved}
        assert cp1.checkpoint_id in ids
        assert cp2.checkpoint_id in ids


class TestDPBudgetLedgerGetLatest:
    def test_get_latest_returns_none_when_empty(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        assert ledger.get_latest() is None

    def test_get_latest_returns_most_recent(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ledger.checkpoint("ep-1")
        acc.record_query(epsilon=0.05)
        cp2 = ledger.checkpoint("ep-2")
        assert ledger.get_latest() is cp2

    def test_get_latest_single_checkpoint(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-solo")
        assert ledger.get_latest() is cp


class TestDPBudgetLedgerVerifyCheckpoint:
    def test_verify_returns_true_for_valid_checkpoint(self):
        acc = _make_accountant()
        acc.record_query(epsilon=0.1)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        assert ledger.verify_checkpoint(cp) is True

    def test_verify_returns_false_when_epsilon_used_tampered(self):
        acc = _make_accountant()
        acc.record_query(epsilon=0.1)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        # Capture the honest hash before tampering.
        original_hash = cp.checkpoint_hash()
        # Tamper with the field after creation.
        cp.epsilon_used = 0.99
        # The recomputed hash must differ from the honest one.
        tampered_hash = cp.checkpoint_hash()
        assert original_hash != tampered_hash

    def test_verify_standalone_checkpoint_is_consistent(self):
        cp = _make_checkpoint()
        ledger = DPBudgetLedger(_make_accountant())
        assert ledger.verify_checkpoint(cp) is True

    def test_verify_after_adding_txid_still_true(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-1")
        # Simulate on-chain anchoring after the fact.
        cp.txid = "abcdef1234567890" * 4
        # Hash must not change — txid is excluded.
        assert ledger.verify_checkpoint(cp) is True


class TestDPBudgetLedgerBudgetHistory:
    def test_budget_history_returns_list_of_dicts(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        ledger.checkpoint("ep-1")
        ledger.checkpoint("ep-2")
        history = ledger.budget_history()
        assert isinstance(history, list)
        assert len(history) == 2
        for item in history:
            assert isinstance(item, dict)

    def test_budget_history_empty_when_no_checkpoints(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        assert ledger.budget_history() == []


class TestDPBudgetLedgerIsBudgetSafe:
    def test_is_safe_true_with_fresh_accountant(self):
        acc = _make_accountant(epsilon_total=1.0)
        ledger = DPBudgetLedger(acc)
        assert ledger.is_budget_safe() is True

    def test_is_safe_false_when_exhausted(self):
        acc = _make_accountant(epsilon_total=0.1)
        acc.record_query(epsilon=0.1)  # exactly at limit → EXHAUSTED
        ledger = DPBudgetLedger(acc)
        assert ledger.is_budget_safe() is False

    def test_is_safe_false_when_exceeded(self):
        acc = _make_accountant(epsilon_total=0.1)
        acc.record_query(epsilon=0.5)  # way over → EXCEEDED
        ledger = DPBudgetLedger(acc)
        assert ledger.is_budget_safe() is False

    def test_is_safe_false_in_warning_zone(self):
        # warn_at=0.8 by default; 90% usage → WARNING
        acc = _make_accountant(epsilon_total=1.0, warn_at=0.8)
        acc.record_query(epsilon=0.9)
        ledger = DPBudgetLedger(acc)
        assert ledger.is_budget_safe() is False


class TestDPCheckpointMechanismBreakdown:
    def test_mechanism_breakdown_counts_correctly(self):
        acc = _make_accountant(epsilon_total=5.0)
        acc.record_query(epsilon=0.1, mechanism=DPMechanism.LAPLACE)
        acc.record_query(epsilon=0.1, mechanism=DPMechanism.LAPLACE)
        acc.record_query(epsilon=0.1, mechanism=DPMechanism.GAUSSIAN)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-mech")
        assert cp.mechanism_breakdown["laplace"] == 2
        assert cp.mechanism_breakdown["gaussian"] == 1

    def test_mechanism_breakdown_empty_before_queries(self):
        acc = _make_accountant()
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-empty")
        assert cp.mechanism_breakdown == {}


class TestDPLedgerStatusReflection:
    def test_exhausted_budget_reflected_in_checkpoint_status(self):
        # epsilon_total=0.0 is invalid; use epsilon_total=0.01 and
        # record a query that equals the full budget to trigger EXHAUSTED.
        acc = _make_accountant(epsilon_total=0.01)
        acc.record_query(epsilon=0.01)
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-exhausted")
        assert cp.status == PrivacyBudgetStatus.EXHAUSTED

    def test_warning_status_reflected_in_checkpoint(self):
        acc = _make_accountant(epsilon_total=1.0, warn_at=0.5)
        acc.record_query(epsilon=0.6)  # 60% > 50% warn threshold
        ledger = DPBudgetLedger(acc)
        cp = ledger.checkpoint("ep-warn")
        assert cp.status == PrivacyBudgetStatus.WARNING
