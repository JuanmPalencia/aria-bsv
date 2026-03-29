"""
aria/multisig.py

Multi-signature epoch governance for regulated industries.

For environments where no single operator may unilaterally submit an epoch to
BSV (e.g. banking, healthcare), this module enforces an M-of-N approval policy.
A proposal must collect at least ``threshold`` valid HMAC-SHA256 approvals from
the registered ``signers`` list before it can be executed.

Security notes:
  - ``signer_secret`` is an operator-held HMAC key. It is NEVER logged, stored,
    or included in any exception message.
  - On any key-related error the message is always "invalid signer credentials".
  - Proposal hashes are computed via ``aria.core.hasher`` (canonical JSON +
    SHA-256), keeping ARIA's single-hasher rule (Regla #2 §5).
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aria.core.hasher import canonical_json, hash_object


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProposalAction(str, Enum):
    """Actions that can be proposed through the multi-sig governance flow."""

    OPEN_EPOCH = "open_epoch"
    CLOSE_EPOCH = "close_epoch"
    ROTATE_KEY = "rotate_key"


class ProposalStatus(str, Enum):
    """Lifecycle states of an EpochProposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class MultiSigPolicy:
    """M-of-N approval policy governing epoch submissions.

    Attributes:
        threshold: Minimum number of approvals (M) required to approve a
            proposal.  Must satisfy 1 <= threshold <= len(signers).
        signers: Ordered list of authorised signer identifiers (e.g.
            ``["auditor-1", "regulator-2", "operator-3"]``).  These are
            public identifiers — they do not contain key material.

    Raises:
        ValueError: If *threshold* is outside the valid range.
    """

    threshold: int
    signers: list[str]

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError(
                f"threshold must be at least 1, got {self.threshold}"
            )
        if self.threshold > len(self.signers):
            raise ValueError(
                f"threshold ({self.threshold}) cannot exceed the number of "
                f"signers ({len(self.signers)})"
            )

    @property
    def quorum_size(self) -> int:
        """Total number of registered signers (N in M-of-N)."""
        return len(self.signers)

    def is_quorum_reached(self, approval_count: int) -> bool:
        """Return True if *approval_count* meets or exceeds the threshold."""
        return approval_count >= self.threshold


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


@dataclass
class EpochApproval:
    """A single signer's approval of an EpochProposal.

    Attributes:
        signer_id:   Identifier of the authorised signer.
        proposal_id: ID of the proposal being approved.
        signature:   HMAC-SHA256 hex digest of ``proposal_hash`` produced with
                     the signer's secret key.
        approved_at: UTC timestamp of when the approval was recorded.
    """

    signer_id: str
    proposal_id: str
    signature: str
    approved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


@dataclass
class EpochProposal:
    """A governance proposal awaiting M-of-N approval before execution.

    Attributes:
        proposal_id:  16 random hex bytes — guaranteed unique per construction.
        action:       The governance action being requested.
        payload:      Action-specific parameters (e.g. model_hashes for
                      OPEN_EPOCH).  Must be JSON-serialisable.
        proposal_hash: SHA-256 of the canonical JSON of
                       ``{proposal_id, action, payload}``.  Used as the HMAC
                       message so signers commit to the exact proposal content.
        policy:       The M-of-N policy in effect at proposal creation time.
        status:       Current lifecycle status.
        approvals:    Ordered list of collected EpochApproval objects.
        created_at:   UTC timestamp of proposal creation.
        executed_at:  UTC timestamp of execution, or None if not yet executed.
    """

    proposal_id: str
    action: ProposalAction
    payload: dict[str, Any]
    proposal_hash: str
    policy: MultiSigPolicy
    status: ProposalStatus = ProposalStatus.PENDING
    approvals: list[EpochApproval] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    executed_at: datetime | None = None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def approval_count(self) -> int:
        """Return the number of approvals collected so far."""
        return len(self.approvals)

    def is_ready(self) -> bool:
        """Return True when the threshold is met and the status is PENDING.

        Note: once status transitions to APPROVED the proposal is ready to
        *execute*, but ``is_ready`` only signals the transition point.
        """
        return (
            self.policy.is_quorum_reached(self.approval_count())
            and self.status == ProposalStatus.PENDING
        )

    def has_approved(self, signer_id: str) -> bool:
        """Return True if *signer_id* has already submitted an approval."""
        return any(a.signer_id == signer_id for a in self.approvals)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


def _compute_proposal_hash(
    proposal_id: str, action: ProposalAction, payload: dict[str, Any]
) -> str:
    """Return the SHA-256 hash of the canonical serialisation of the proposal core."""
    return hash_object(
        {"action": action.value, "payload": payload, "proposal_id": proposal_id}
    )


def _hmac_sign(signer_secret: str, message: str) -> str:
    """Return the HMAC-SHA256 hex digest of *message* using *signer_secret*.

    Args:
        signer_secret: The signer's private HMAC key.
        message:       The message to authenticate (the proposal_hash string).

    Returns:
        Lowercase hex string of the 32-byte HMAC digest.
    """
    try:
        mac = hmac.new(
            signer_secret.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        return mac.hexdigest()
    except Exception:
        raise ValueError("invalid signer credentials")


class MultiSigEpochManager:
    """Coordinates M-of-N approval workflows for ARIA epoch governance.

    Each proposal is tracked in an in-memory store.  When a proposal reaches
    the policy threshold it transitions to APPROVED; an authorised caller then
    calls ``execute()`` to finalise and optionally dispatch to the underlying
    EpochManager.

    Args:
        policy:        The M-of-N approval policy to enforce on all proposals.
        epoch_manager: Optional EpochManager instance.  When provided, executing
                       an OPEN_EPOCH proposal calls
                       ``epoch_manager.open_epoch(**payload)`` (async callers
                       must await the returned coroutine themselves), and
                       CLOSE_EPOCH calls ``epoch_manager.close_epoch(**payload)``.
                       ROTATE_KEY does not invoke the epoch_manager.
    """

    def __init__(
        self,
        policy: MultiSigPolicy,
        epoch_manager: Any = None,
    ) -> None:
        self._policy = policy
        self._epoch_manager = epoch_manager
        self._proposals: dict[str, EpochProposal] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self, action: ProposalAction, payload: dict[str, Any]
    ) -> EpochProposal:
        """Create a new governance proposal and store it as PENDING.

        Args:
            action:  The action being requested.
            payload: Action-specific parameters.

        Returns:
            The newly created EpochProposal in PENDING status.
        """
        proposal_id = secrets.token_hex(16)
        proposal_hash = _compute_proposal_hash(proposal_id, action, payload)
        proposal = EpochProposal(
            proposal_id=proposal_id,
            action=action,
            payload=payload,
            proposal_hash=proposal_hash,
            policy=self._policy,
        )
        self._proposals[proposal_id] = proposal
        return proposal

    def approve(
        self,
        proposal_id: str,
        signer_id: str,
        signer_secret: str,
    ) -> EpochProposal:
        """Record an approval from *signer_id* for the given proposal.

        The approval is authenticated via HMAC-SHA256(signer_secret,
        proposal_hash).  The secret is never stored or logged.

        Args:
            proposal_id:   ID of the proposal to approve.
            signer_id:     Identifier of the approving signer.
            signer_secret: The signer's private HMAC key.  Never logged.

        Returns:
            The updated EpochProposal (status may have changed to APPROVED).

        Raises:
            ValueError: If *signer_id* is not in the policy, or has already
                approved this proposal.
            KeyError:   If *proposal_id* does not exist.
            ValueError: If the proposal is not in PENDING status.
        """
        if signer_id not in self._policy.signers:
            raise ValueError(
                f"signer {signer_id!r} is not registered in the policy"
            )

        proposal = self._get_or_raise(proposal_id)

        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id!r} is not PENDING "
                f"(current status: {proposal.status.value})"
            )

        if proposal.has_approved(signer_id):
            raise ValueError(
                f"signer {signer_id!r} has already approved proposal {proposal_id!r}"
            )

        try:
            signature = _hmac_sign(signer_secret, proposal.proposal_hash)
        except ValueError:
            raise ValueError("invalid signer credentials")

        approval = EpochApproval(
            signer_id=signer_id,
            proposal_id=proposal_id,
            signature=signature,
        )
        proposal.approvals.append(approval)

        if self._policy.is_quorum_reached(proposal.approval_count()):
            proposal.status = ProposalStatus.APPROVED

        return proposal

    def reject(self, proposal_id: str, signer_id: str) -> EpochProposal:
        """Mark a proposal as REJECTED.

        Any registered signer may reject a proposal while it is still PENDING.

        Args:
            proposal_id: ID of the proposal to reject.
            signer_id:   Identifier of the rejecting signer.

        Returns:
            The updated EpochProposal with status REJECTED.

        Raises:
            ValueError: If *signer_id* is not in the policy.
            KeyError:   If *proposal_id* does not exist.
            ValueError: If the proposal is not in PENDING status.
        """
        if signer_id not in self._policy.signers:
            raise ValueError(
                f"signer {signer_id!r} is not registered in the policy"
            )

        proposal = self._get_or_raise(proposal_id)

        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(
                f"proposal {proposal_id!r} is not PENDING "
                f"(current status: {proposal.status.value})"
            )

        proposal.status = ProposalStatus.REJECTED
        return proposal

    def execute(self, proposal_id: str) -> dict[str, Any]:
        """Execute an APPROVED proposal.

        If an ``epoch_manager`` was provided at construction time, dispatches
        the appropriate call:
          - OPEN_EPOCH  → ``epoch_manager.open_epoch(**payload)``
          - CLOSE_EPOCH → ``epoch_manager.close_epoch(**payload)``
          - ROTATE_KEY  → no epoch_manager call (key rotation is out-of-band)

        The caller is responsible for awaiting coroutines returned by an async
        ``epoch_manager``.

        Args:
            proposal_id: ID of the proposal to execute.

        Returns:
            dict with keys ``executed``, ``proposal_id``, ``action``,
            ``executed_at``.

        Raises:
            KeyError:   If *proposal_id* does not exist.
            ValueError: If the proposal status is not APPROVED.
        """
        proposal = self._get_or_raise(proposal_id)

        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError(
                f"proposal {proposal_id!r} is not APPROVED "
                f"(current status: {proposal.status.value})"
            )

        proposal.status = ProposalStatus.EXECUTED
        proposal.executed_at = datetime.now(timezone.utc)

        if self._epoch_manager is not None:
            if proposal.action == ProposalAction.OPEN_EPOCH:
                self._epoch_manager.open_epoch(**proposal.payload)
            elif proposal.action == ProposalAction.CLOSE_EPOCH:
                self._epoch_manager.close_epoch(**proposal.payload)
            # ROTATE_KEY: no epoch_manager method — handled out-of-band.

        return {
            "executed": True,
            "proposal_id": proposal.proposal_id,
            "action": proposal.action.value,
            "executed_at": proposal.executed_at.isoformat(),
        }

    def get_pending(self) -> list[EpochProposal]:
        """Return all proposals currently in PENDING status."""
        return [
            p for p in self._proposals.values()
            if p.status == ProposalStatus.PENDING
        ]

    def get_proposal(self, proposal_id: str) -> EpochProposal | None:
        """Return the proposal with *proposal_id*, or None if not found."""
        return self._proposals.get(proposal_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, proposal_id: str) -> EpochProposal:
        """Return the proposal or raise KeyError."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"proposal {proposal_id!r} not found")
        return proposal
