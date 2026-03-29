"""
tests/unit/test_multisig.py

Unit tests for aria.multisig — multi-signature epoch governance.

Coverage targets:
  - MultiSigPolicy construction and validation
  - EpochProposal hash determinism and query helpers
  - MultiSigEpochManager propose / approve / reject / execute flows
  - Security edge cases (unknown signer, duplicate approval, wrong status)
  - Full M-of-N end-to-end flows
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from aria.multisig import (
    EpochApproval,
    EpochProposal,
    MultiSigEpochManager,
    MultiSigPolicy,
    ProposalAction,
    ProposalStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def policy_2_of_3() -> MultiSigPolicy:
    return MultiSigPolicy(
        threshold=2,
        signers=["auditor-1", "auditor-2", "regulator-1"],
    )


@pytest.fixture
def policy_1_of_1() -> MultiSigPolicy:
    return MultiSigPolicy(threshold=1, signers=["sole-signer"])


@pytest.fixture
def policy_1_of_2() -> MultiSigPolicy:
    return MultiSigPolicy(threshold=1, signers=["signer-a", "signer-b"])


@pytest.fixture
def manager_2_of_3(policy_2_of_3: MultiSigPolicy) -> MultiSigEpochManager:
    return MultiSigEpochManager(policy=policy_2_of_3)


@pytest.fixture
def manager_1_of_1(policy_1_of_1: MultiSigPolicy) -> MultiSigEpochManager:
    return MultiSigEpochManager(policy=policy_1_of_1)


@pytest.fixture
def open_payload() -> dict:
    return {
        "model_hashes": {"model-a": "sha256:deadbeef"},
        "system_state": {"env": "prod"},
    }


# ---------------------------------------------------------------------------
# MultiSigPolicy — validation
# ---------------------------------------------------------------------------


def test_policy_threshold_zero_raises():
    with pytest.raises(ValueError, match="at least 1"):
        MultiSigPolicy(threshold=0, signers=["a", "b"])


def test_policy_threshold_negative_raises():
    with pytest.raises(ValueError, match="at least 1"):
        MultiSigPolicy(threshold=-1, signers=["a"])


def test_policy_threshold_exceeds_signers_raises():
    with pytest.raises(ValueError, match="cannot exceed"):
        MultiSigPolicy(threshold=4, signers=["a", "b", "c"])


def test_policy_threshold_equals_signers_is_valid():
    policy = MultiSigPolicy(threshold=3, signers=["a", "b", "c"])
    assert policy.threshold == 3


def test_policy_threshold_one_is_valid():
    policy = MultiSigPolicy(threshold=1, signers=["only-one"])
    assert policy.threshold == 1


# ---------------------------------------------------------------------------
# MultiSigPolicy — quorum_size and is_quorum_reached
# ---------------------------------------------------------------------------


def test_policy_quorum_size_equals_len_signers(policy_2_of_3: MultiSigPolicy):
    assert policy_2_of_3.quorum_size == 3


def test_policy_quorum_size_single_signer(policy_1_of_1: MultiSigPolicy):
    assert policy_1_of_1.quorum_size == 1


def test_is_quorum_reached_true_at_threshold(policy_2_of_3: MultiSigPolicy):
    assert policy_2_of_3.is_quorum_reached(2) is True


def test_is_quorum_reached_true_above_threshold(policy_2_of_3: MultiSigPolicy):
    assert policy_2_of_3.is_quorum_reached(3) is True


def test_is_quorum_reached_false_below_threshold(policy_2_of_3: MultiSigPolicy):
    assert policy_2_of_3.is_quorum_reached(1) is False


def test_is_quorum_reached_false_zero(policy_2_of_3: MultiSigPolicy):
    assert policy_2_of_3.is_quorum_reached(0) is False


# ---------------------------------------------------------------------------
# EpochProposal — hash determinism and query helpers
# ---------------------------------------------------------------------------


def test_proposal_hash_is_deterministic(manager_2_of_3: MultiSigEpochManager, open_payload: dict):
    """Same action + payload must always yield the same proposal_hash."""
    p1 = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    p2 = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    # Different proposal_ids → different hashes (proposal_id is part of hash input)
    assert p1.proposal_hash != p2.proposal_hash


def test_proposal_hash_commits_to_content(manager_2_of_3: MultiSigEpochManager):
    """Changing payload must change the proposal_hash."""
    p1 = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, {"key": "value-a"})
    p2 = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, {"key": "value-b"})
    assert p1.proposal_hash != p2.proposal_hash


def test_has_approved_false_before_any_approval(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.has_approved("auditor-1") is False


def test_is_ready_false_before_threshold(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.is_ready() is False


def test_approval_count_starts_at_zero(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.approval_count() == 0


# ---------------------------------------------------------------------------
# MultiSigEpochManager.propose()
# ---------------------------------------------------------------------------


def test_propose_creates_pending_proposal(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.status == ProposalStatus.PENDING


def test_propose_stores_correct_action(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.action == ProposalAction.OPEN_EPOCH


def test_propose_generates_unique_proposal_ids(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    ids = {
        manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload).proposal_id
        for _ in range(50)
    }
    assert len(ids) == 50


def test_propose_proposal_id_is_32_hex_chars(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert len(proposal.proposal_id) == 32
    int(proposal.proposal_id, 16)  # raises if not valid hex


def test_propose_sets_created_at(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert isinstance(proposal.created_at, datetime)
    assert proposal.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# MultiSigEpochManager.approve() — error paths
# ---------------------------------------------------------------------------


def test_approve_unknown_signer_raises_value_error(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    with pytest.raises(ValueError, match="not registered"):
        manager_2_of_3.approve(proposal.proposal_id, "unknown-signer", "secret")


def test_approve_nonexistent_proposal_raises_key_error(
    manager_2_of_3: MultiSigEpochManager,
):
    with pytest.raises(KeyError):
        manager_2_of_3.approve("nonexistent-id", "auditor-1", "secret")


def test_approve_duplicate_signer_raises_value_error(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "secret-a1")
    with pytest.raises(ValueError, match="already approved"):
        manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "secret-a1")


# ---------------------------------------------------------------------------
# MultiSigEpochManager.approve() — happy path
# ---------------------------------------------------------------------------


def test_approve_adds_approval_to_proposal(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    updated = manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "secret-a1")
    assert updated.approval_count() == 1
    assert updated.has_approved("auditor-1") is True


def test_approve_returns_updated_proposal_object(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    returned = manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "secret-a1")
    assert returned is proposal


def test_approve_below_threshold_stays_pending(open_payload: dict):
    """1-of-2 policy: first approval meets threshold immediately → APPROVED.
    This test verifies the PENDING state BEFORE that first approval is recorded.
    We use a 2-of-3 manager to test the sub-threshold case."""
    manager = MultiSigEpochManager(
        policy=MultiSigPolicy(threshold=2, signers=["s1", "s2", "s3"])
    )
    proposal = manager.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager.approve(proposal.proposal_id, "s1", "k1")
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.approval_count() == 1


def test_approve_at_threshold_sets_approved(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "secret-a1")
    manager_2_of_3.approve(proposal.proposal_id, "auditor-2", "secret-a2")
    assert proposal.status == ProposalStatus.APPROVED


def test_approve_1_of_1_sets_approved_after_single_approval(
    manager_1_of_1: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_1_of_1.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_1_of_1.approve(proposal.proposal_id, "sole-signer", "my-secret")
    assert proposal.status == ProposalStatus.APPROVED


def test_approve_stores_signature_field(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "k1")
    approval = proposal.approvals[0]
    assert isinstance(approval.signature, str)
    assert len(approval.signature) == 64  # HMAC-SHA256 hex = 32 bytes = 64 chars


# ---------------------------------------------------------------------------
# MultiSigEpochManager.reject()
# ---------------------------------------------------------------------------


def test_reject_sets_status_rejected(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.reject(proposal.proposal_id, "auditor-1")
    assert proposal.status == ProposalStatus.REJECTED


def test_reject_unknown_signer_raises(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    with pytest.raises(ValueError, match="not registered"):
        manager_2_of_3.reject(proposal.proposal_id, "outsider")


def test_approve_after_reject_raises(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.reject(proposal.proposal_id, "auditor-1")
    with pytest.raises(ValueError, match="not PENDING"):
        manager_2_of_3.approve(proposal.proposal_id, "auditor-2", "k2")


# ---------------------------------------------------------------------------
# MultiSigEpochManager.execute()
# ---------------------------------------------------------------------------


def test_execute_on_approved_proposal_returns_dict(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "k1")
    manager_2_of_3.approve(proposal.proposal_id, "auditor-2", "k2")
    result = manager_2_of_3.execute(proposal.proposal_id)
    assert result["executed"] is True
    assert result["proposal_id"] == proposal.proposal_id
    assert result["action"] == ProposalAction.OPEN_EPOCH.value


def test_execute_sets_executed_at(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "k1")
    manager_2_of_3.approve(proposal.proposal_id, "auditor-2", "k2")
    manager_2_of_3.execute(proposal.proposal_id)
    assert isinstance(proposal.executed_at, datetime)
    assert proposal.executed_at.tzinfo is not None


def test_execute_sets_status_executed(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.approve(proposal.proposal_id, "auditor-1", "k1")
    manager_2_of_3.approve(proposal.proposal_id, "auditor-2", "k2")
    manager_2_of_3.execute(proposal.proposal_id)
    assert proposal.status == ProposalStatus.EXECUTED


def test_execute_on_pending_raises_value_error(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    with pytest.raises(ValueError, match="not APPROVED"):
        manager_2_of_3.execute(proposal.proposal_id)


def test_execute_on_rejected_raises_value_error(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager_2_of_3.reject(proposal.proposal_id, "auditor-1")
    with pytest.raises(ValueError, match="not APPROVED"):
        manager_2_of_3.execute(proposal.proposal_id)


def test_execute_on_nonexistent_proposal_raises_key_error(
    manager_2_of_3: MultiSigEpochManager,
):
    with pytest.raises(KeyError):
        manager_2_of_3.execute("does-not-exist")


def test_execute_calls_epoch_manager_open_epoch():
    """execute() with OPEN_EPOCH action calls epoch_manager.open_epoch(**payload)."""
    mock_em = MagicMock()
    policy = MultiSigPolicy(threshold=1, signers=["signer-x"])
    manager = MultiSigEpochManager(policy=policy, epoch_manager=mock_em)
    payload = {"model_hashes": {"m": "sha256:aa"}, "system_state": {}}
    proposal = manager.propose(ProposalAction.OPEN_EPOCH, payload)
    manager.approve(proposal.proposal_id, "signer-x", "secret")
    manager.execute(proposal.proposal_id)
    mock_em.open_epoch.assert_called_once_with(**payload)


def test_execute_calls_epoch_manager_close_epoch():
    """execute() with CLOSE_EPOCH action calls epoch_manager.close_epoch(**payload)."""
    mock_em = MagicMock()
    policy = MultiSigPolicy(threshold=1, signers=["signer-x"])
    manager = MultiSigEpochManager(policy=policy, epoch_manager=mock_em)
    payload = {"epoch_id": "ep_001", "records": []}
    proposal = manager.propose(ProposalAction.CLOSE_EPOCH, payload)
    manager.approve(proposal.proposal_id, "signer-x", "secret")
    manager.execute(proposal.proposal_id)
    mock_em.close_epoch.assert_called_once_with(**payload)


def test_execute_rotate_key_does_not_call_epoch_manager():
    """execute() with ROTATE_KEY does not call epoch_manager methods."""
    mock_em = MagicMock()
    policy = MultiSigPolicy(threshold=1, signers=["signer-x"])
    manager = MultiSigEpochManager(policy=policy, epoch_manager=mock_em)
    proposal = manager.propose(ProposalAction.ROTATE_KEY, {"new_key_id": "key-2"})
    manager.approve(proposal.proposal_id, "signer-x", "secret")
    manager.execute(proposal.proposal_id)
    mock_em.open_epoch.assert_not_called()
    mock_em.close_epoch.assert_not_called()


# ---------------------------------------------------------------------------
# MultiSigEpochManager.get_pending()
# ---------------------------------------------------------------------------


def test_get_pending_returns_only_pending(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    p1 = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    p2 = manager_2_of_3.propose(ProposalAction.CLOSE_EPOCH, {"epoch_id": "ep"})
    manager_2_of_3.reject(p2.proposal_id, "auditor-1")
    pending = manager_2_of_3.get_pending()
    assert p1 in pending
    assert p2 not in pending


def test_get_pending_empty_when_no_proposals(manager_2_of_3: MultiSigEpochManager):
    assert manager_2_of_3.get_pending() == []


# ---------------------------------------------------------------------------
# MultiSigEpochManager.get_proposal()
# ---------------------------------------------------------------------------


def test_get_proposal_returns_proposal(
    manager_2_of_3: MultiSigEpochManager, open_payload: dict
):
    proposal = manager_2_of_3.propose(ProposalAction.OPEN_EPOCH, open_payload)
    found = manager_2_of_3.get_proposal(proposal.proposal_id)
    assert found is proposal


def test_get_proposal_returns_none_for_missing(
    manager_2_of_3: MultiSigEpochManager,
):
    assert manager_2_of_3.get_proposal("nonexistent") is None


# ---------------------------------------------------------------------------
# End-to-end flows
# ---------------------------------------------------------------------------


def test_full_2_of_3_flow(policy_2_of_3: MultiSigPolicy, open_payload: dict):
    """Full 2-of-3 flow: propose → 2 approvals → execute → EXECUTED."""
    manager = MultiSigEpochManager(policy=policy_2_of_3)

    proposal = manager.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.status == ProposalStatus.PENDING

    manager.approve(proposal.proposal_id, "auditor-1", "k1")
    assert proposal.status == ProposalStatus.PENDING

    manager.approve(proposal.proposal_id, "auditor-2", "k2")
    assert proposal.status == ProposalStatus.APPROVED

    result = manager.execute(proposal.proposal_id)
    assert result["executed"] is True
    assert proposal.status == ProposalStatus.EXECUTED
    assert proposal.executed_at is not None


def test_full_1_of_1_flow(policy_1_of_1: MultiSigPolicy, open_payload: dict):
    """Full 1-of-1 flow: propose → 1 approval → execute → EXECUTED."""
    manager = MultiSigEpochManager(policy=policy_1_of_1)

    proposal = manager.propose(ProposalAction.OPEN_EPOCH, open_payload)
    assert proposal.status == ProposalStatus.PENDING

    manager.approve(proposal.proposal_id, "sole-signer", "solo-secret")
    assert proposal.status == ProposalStatus.APPROVED

    result = manager.execute(proposal.proposal_id)
    assert result["executed"] is True
    assert proposal.status == ProposalStatus.EXECUTED


def test_third_approval_does_not_change_status_back(
    policy_2_of_3: MultiSigPolicy, open_payload: dict
):
    """After threshold reached and APPROVED, a third approval must not alter status."""
    manager = MultiSigEpochManager(policy=policy_2_of_3)
    proposal = manager.propose(ProposalAction.OPEN_EPOCH, open_payload)
    manager.approve(proposal.proposal_id, "auditor-1", "k1")
    manager.approve(proposal.proposal_id, "auditor-2", "k2")
    assert proposal.status == ProposalStatus.APPROVED
    # Third approval attempt should raise because status is no longer PENDING.
    with pytest.raises(ValueError, match="not PENDING"):
        manager.approve(proposal.proposal_id, "regulator-1", "k3")
