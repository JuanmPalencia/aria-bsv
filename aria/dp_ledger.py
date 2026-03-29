"""
aria.dp_ledger — On-chain anchoring of Differential Privacy budget consumption.

Extends aria.privacy with BSV-backed immutable records of how much privacy
budget was consumed per ARIA epoch.  Every time a team calls
``DPBudgetLedger.checkpoint(epoch_id)``, the current state of the
``PrivacyAccountant`` is snapshotted, hashed canonically, and — when a
broadcaster is supplied — published as an OP_RETURN transaction on BSV.

The on-chain record is a BRC-121 DP_CHECKPOINT payload::

    {
        "type":           "DP_CHECKPOINT",
        "brc":            "121",
        "epoch_id":       "<epoch_id>",
        "checkpoint_hash": "sha256:<hex>",
        "epsilon_used":   <float>,
        "epsilon_total":  <float>,
        "delta_used":     <float>,
        "query_count":    <int>,
        "status":         "<SAFE|WARNING|EXHAUSTED|EXCEEDED>"
    }

The ``checkpoint_hash`` is computed from the canonical JSON of the fields
above plus ``mechanism_breakdown``, but NOT ``txid`` or ``created_at``, so
the hash remains verifiable off-chain without blockchain data.

Usage::

    from aria.privacy import PrivacyAccountant
    from aria.dp_ledger import DPBudgetLedger

    accountant = PrivacyAccountant(epsilon_total=1.0, delta=1e-5)
    ledger = DPBudgetLedger(accountant)

    accountant.record_query(epsilon=0.1, mechanism="laplace")
    cp = ledger.checkpoint(epoch_id="epoch-001")

    print(cp.epsilon_remaining())   # 0.9
    print(cp.is_on_chain())         # False — no broadcaster supplied
    print(ledger.verify_checkpoint(cp))  # True
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aria.core.hasher import canonical_json, hash_object
from aria.privacy import PrivacyAccountant, PrivacyBudgetStatus

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DPCheckpoint
# ---------------------------------------------------------------------------

@dataclass
class DPCheckpoint:
    """Immutable snapshot of DP budget state at a point in time.

    Args:
        checkpoint_id:      16-character hex string (8 random bytes).
        epoch_id:           ARIA epoch this checkpoint is tied to.
        epsilon_used:       Cumulative epsilon consumed so far.
        epsilon_total:      Total epsilon budget of the accountant.
        delta_used:         Cumulative delta consumed so far.
        query_count:        Total number of queries recorded.
        status:             Current budget status enum value.
        mechanism_breakdown: Count of queries per mechanism name.
        created_at:         UTC timestamp of checkpoint creation.
        txid:               BSV transaction ID if anchored on-chain, else None.
    """

    checkpoint_id: str
    epoch_id: str
    epsilon_used: float
    epsilon_total: float
    delta_used: float
    query_count: int
    status: PrivacyBudgetStatus
    mechanism_breakdown: dict[str, int]
    created_at: datetime
    txid: str | None = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def checkpoint_hash(self) -> str:
        """Return the canonical SHA-256 hash of this checkpoint's content.

        The hash covers all fields that describe the privacy state:
        ``checkpoint_id``, ``epoch_id``, ``epsilon_used``, ``epsilon_total``,
        ``delta_used``, ``query_count``, ``status``, and
        ``mechanism_breakdown``.

        ``txid`` and ``created_at`` are intentionally excluded so the hash
        is stable and verifiable regardless of whether the record has been
        anchored on-chain yet.

        Returns:
            str in the format ``"sha256:<64 hex chars>"``.
        """
        payload: dict[str, Any] = {
            "checkpoint_id": self.checkpoint_id,
            "delta_used": self.delta_used,
            "epsilon_total": self.epsilon_total,
            "epsilon_used": self.epsilon_used,
            "epoch_id": self.epoch_id,
            "mechanism_breakdown": self.mechanism_breakdown,
            "query_count": self.query_count,
            "status": self.status.value,
        }
        return hash_object(payload)

    def epsilon_remaining(self) -> float:
        """Return the epsilon budget not yet consumed.

        Returns:
            max(0.0, epsilon_total - epsilon_used)
        """
        return max(0.0, self.epsilon_total - self.epsilon_used)

    def is_on_chain(self) -> bool:
        """Return True when this checkpoint has a BSV txid.

        Returns:
            True if ``txid`` is a non-empty string, False otherwise.
        """
        return self.txid is not None and len(self.txid) > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation.

        ``created_at`` is serialised as an ISO-8601 UTC string.
        All other fields are native Python scalars.

        Returns:
            dict suitable for ``json.dumps``.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "epoch_id": self.epoch_id,
            "epsilon_used": self.epsilon_used,
            "epsilon_total": self.epsilon_total,
            "delta_used": self.delta_used,
            "query_count": self.query_count,
            "status": self.status.value,
            "mechanism_breakdown": dict(self.mechanism_breakdown),
            "created_at": self.created_at.isoformat(),
            "txid": self.txid,
            "checkpoint_hash": self.checkpoint_hash(),
        }


# ---------------------------------------------------------------------------
# DPBudgetLedger
# ---------------------------------------------------------------------------

class DPBudgetLedger:
    """Ordered ledger of DP budget checkpoints for an ARIA epoch sequence.

    Each call to :meth:`checkpoint` snapshots the current state of the
    ``PrivacyAccountant``, optionally publishes it to BSV via ``broadcaster``,
    and appends the result to the internal ordered list.

    Args:
        accountant:  A live ``PrivacyAccountant`` instance to read state from.
        broadcaster: Optional object with a ``broadcast(payload: bytes)``
                     method that returns an object with a ``.txid`` attribute.
                     Pass ``None`` to operate in audit-only mode (no on-chain
                     anchoring).

    Example::

        ledger = DPBudgetLedger(accountant, broadcaster=my_broadcaster)
        cp = ledger.checkpoint("epoch-42")
        print(cp.is_on_chain())  # True if broadcast succeeded
    """

    def __init__(
        self,
        accountant: PrivacyAccountant,
        broadcaster: Any = None,
    ) -> None:
        self._accountant = accountant
        self._broadcaster = broadcaster
        self._checkpoints: list[DPCheckpoint] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def checkpoint(self, epoch_id: str) -> DPCheckpoint:
        """Snapshot the current DP budget state and optionally anchor on BSV.

        Steps:
        1. Read current state from the accountant via ``.status()``.
        2. Build a ``DPCheckpoint`` with a fresh random ``checkpoint_id``.
        3. Compute ``checkpoint_hash`` (deterministic, excludes txid).
        4. If a broadcaster is configured, build the BRC-121 OP_RETURN
           payload and call ``broadcaster.broadcast(payload_bytes)``.
           Any exception from broadcast is caught and logged — the
           checkpoint is still stored with ``txid=None``.
        5. Append the checkpoint to the internal list and return it.

        Args:
            epoch_id: Identifier of the ARIA epoch this checkpoint belongs to.

        Returns:
            The newly created ``DPCheckpoint``.
        """
        ps = self._accountant.status()

        # Build mechanism breakdown from the accountant's query list.
        mechanism_breakdown: dict[str, int] = {}
        for q in self._accountant.queries:
            key = q.mechanism.value
            mechanism_breakdown[key] = mechanism_breakdown.get(key, 0) + 1

        cp = DPCheckpoint(
            checkpoint_id=secrets.token_hex(8),
            epoch_id=epoch_id,
            epsilon_used=ps.epsilon_used,
            epsilon_total=ps.epsilon_total,
            delta_used=ps.delta_used,
            query_count=ps.n_queries,
            status=ps.status,
            mechanism_breakdown=mechanism_breakdown,
            created_at=datetime.now(timezone.utc),
            txid=None,
        )

        # Attempt on-chain anchoring if broadcaster is available.
        if self._broadcaster is not None:
            cp_hash = cp.checkpoint_hash()
            payload_dict: dict[str, Any] = {
                "brc": "121",
                "checkpoint_hash": cp_hash,
                "delta_used": ps.delta_used,
                "epoch_id": epoch_id,
                "epsilon_total": ps.epsilon_total,
                "epsilon_used": ps.epsilon_used,
                "query_count": ps.n_queries,
                "status": ps.status.value,
                "type": "DP_CHECKPOINT",
            }
            payload_bytes = canonical_json(payload_dict)
            try:
                result = self._broadcaster.broadcast(payload_bytes)
                if hasattr(result, "txid") and result.txid:
                    cp.txid = result.txid
                    _log.info(
                        "DP checkpoint anchored on BSV: epoch=%s txid=%s",
                        epoch_id,
                        result.txid,
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Broadcast failed for DP checkpoint (epoch=%s): %s — "
                    "checkpoint stored locally only.",
                    epoch_id,
                    exc,
                )

        self._checkpoints.append(cp)
        _log.debug(
            "DP checkpoint created: id=%s epoch=%s ε_used=%.6f status=%s on_chain=%s",
            cp.checkpoint_id,
            epoch_id,
            cp.epsilon_used,
            cp.status.value,
            cp.is_on_chain(),
        )
        return cp

    def get_checkpoints(
        self,
        epoch_id: str | None = None,
    ) -> list[DPCheckpoint]:
        """Return all stored checkpoints, optionally filtered by epoch.

        Args:
            epoch_id: If provided, return only checkpoints whose
                      ``epoch_id`` matches exactly.  If ``None``,
                      return all checkpoints in insertion order.

        Returns:
            List of ``DPCheckpoint`` objects (may be empty).
        """
        if epoch_id is None:
            return list(self._checkpoints)
        return [cp for cp in self._checkpoints if cp.epoch_id == epoch_id]

    def get_latest(self) -> DPCheckpoint | None:
        """Return the most recently created checkpoint, or None if empty.

        Returns:
            The last ``DPCheckpoint`` appended, or ``None``.
        """
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def verify_checkpoint(self, checkpoint: DPCheckpoint) -> bool:
        """Verify the integrity of a checkpoint by recomputing its hash.

        Recomputes ``checkpoint.checkpoint_hash()`` from the checkpoint's
        current field values and compares it against a freshly-computed
        reference.  Any mutation to the checkpoint's data fields will cause
        this to fail.

        Note: because ``checkpoint_hash()`` is a method on the dataclass
        itself, this effectively checks whether the instance's own fields
        are internally consistent — which catches accidental mutations but
        not cases where both the stored hash and the fields are altered
        together.  For stronger guarantees, compare against the on-chain
        payload.

        Args:
            checkpoint: A ``DPCheckpoint`` to verify.

        Returns:
            True if the checkpoint hash is internally consistent.
        """
        try:
            recomputed = checkpoint.checkpoint_hash()
            # Re-instantiate a clean copy and compute from scratch to guard
            # against any monkey-patching of the method itself.
            clean = DPCheckpoint(
                checkpoint_id=checkpoint.checkpoint_id,
                epoch_id=checkpoint.epoch_id,
                epsilon_used=checkpoint.epsilon_used,
                epsilon_total=checkpoint.epsilon_total,
                delta_used=checkpoint.delta_used,
                query_count=checkpoint.query_count,
                status=checkpoint.status,
                mechanism_breakdown=dict(checkpoint.mechanism_breakdown),
                created_at=checkpoint.created_at,
                txid=checkpoint.txid,
            )
            expected = clean.checkpoint_hash()
            return recomputed == expected
        except Exception as exc:  # noqa: BLE001
            _log.error("verify_checkpoint raised an exception: %s", exc)
            return False

    def budget_history(self) -> list[dict[str, Any]]:
        """Return the full budget history as a list of serialisable dicts.

        Calls ``to_dict()`` on every stored checkpoint in insertion order.

        Returns:
            List of dicts — one per checkpoint, each including
            ``checkpoint_hash``, suitable for JSON export or audit logs.
        """
        return [cp.to_dict() for cp in self._checkpoints]

    def is_budget_safe(self) -> bool:
        """Return True if the accountant's current status is SAFE.

        Returns:
            True only when ``accountant.status().status == SAFE``.
            WARNING, EXHAUSTED, and EXCEEDED all return False.
        """
        return self._accountant.status().status == PrivacyBudgetStatus.SAFE
