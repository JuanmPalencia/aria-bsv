"""
aria/sampling.py

Cryptographically verifiable audit sampling for high-throughput AI systems.

For systems processing 1M+ inferences per day, recording every inference
on-chain is impractical. This module implements audit sampling: record only
N% of inferences, but with a verifiable random seed anchored on BSV so any
third party can independently verify which inferences were selected.

Key guarantees:
- Deterministic: same seed always produces the same selection sequence.
- Verifiable: a BSV txid as seed means any party can reproduce the sequence.
- Tamper-evident: VerifiableSamplingProof binds seed + selections together.
- Three methods: Bernoulli (probabilistic), Systematic (deterministic
  interval), Reservoir (approximate streaming).

BRC-121 compliance: seed_txid must reference an on-chain EPOCH OPEN transaction
to make the sampling auditable under EU AI Act Art. 9 (risk management) and
Art. 12 (record-keeping).
"""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

from aria.core.hasher import canonical_json, hash_object


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SamplingMethod(str, Enum):
    """Strategy used to decide whether each inference is selected for audit.

    Attributes:
        BERNOULLI:  Independent coin flip per inference.  ``rate`` is the
                    probability of selection.  The actual selected fraction
                    converges to ``rate`` over large streams but varies for
                    small ones.
        RESERVOIR:  Classically selects exactly *k* items from a stream of
                    unknown size.  In streaming mode we approximate this with
                    Bernoulli because the stream end is unknown at call time.
        SYSTEMATIC: Deterministic interval sampling.  Every
                    ``round(1/rate)``-th inference is selected.  Exactly
                    ``rate`` fraction for streams whose length is a multiple
                    of the interval.
    """

    BERNOULLI = "bernoulli"
    RESERVOIR = "reservoir"
    SYSTEMATIC = "systematic"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Configuration for an audit sampling session.

    Attributes:
        rate:        Fraction of inferences to audit.  Must be in (0.0, 1.0].
        method:      Sampling strategy to use (default: BERNOULLI).
        seed_txid:   BSV transaction ID used as the primary entropy source.
                     When set the sampling is verifiable by any third party
                     who can look up the transaction.
        seed_block:  BSV block height at the time the epoch opened.  Used as
                     secondary entropy alongside *seed_txid*.

    Raises:
        ValueError: if *rate* is not in the range (0.0, 1.0].
    """

    rate: float
    method: SamplingMethod = SamplingMethod.BERNOULLI
    seed_txid: str | None = None
    seed_block: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.rate <= 1.0):
            raise ValueError(
                f"Sampling rate must be in the range (0.0, 1.0], got {self.rate!r}."
            )

    @property
    def seed_material(self) -> bytes:
        """Canonical bytes encoding of all seed fields.

        This is the deterministic input to the PRNG.  It encodes every field
        that influences the seed so that two configs with different seeds
        always produce different byte strings.

        Returns:
            UTF-8 encoded canonical JSON bytes of the seed descriptor dict.
        """
        seed_dict: dict[str, Any] = {
            "method": self.method.value,
            "rate": self.rate,
            "seed_block": self.seed_block,
            "seed_txid": self.seed_txid,
        }
        return canonical_json(seed_dict)

    @property
    def is_verifiable(self) -> bool:
        """True when a BSV txid was supplied as the entropy source.

        A verifiable sampler can be independently audited by any party that
        has access to the BSV blockchain: they look up *seed_txid*, derive
        the same PRNG seed, and replay the selection sequence.
        """
        return self.seed_txid is not None


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass
class SampleDecision:
    """Immutable record of a single sampling decision.

    Attributes:
        inference_id:  Unique identifier for the inference being evaluated.
        selected:      Whether this inference was chosen for audit.
        method:        The sampling strategy that produced this decision.
        seed_hash:     SHA-256 of the seed material used, in ``sha256:<hex>``
                       format.  Proves which seed was in effect.
        position:      Zero-based sequential index of this inference in the
                       stream (0 = first call to ``should_record``).
        decided_at:    UTC timestamp at the moment the decision was made.
    """

    inference_id: str
    selected: bool
    method: SamplingMethod
    seed_hash: str
    position: int
    decided_at: datetime


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------


class AuditSampler:
    """Stateful, deterministic audit sampler.

    Maintains a sequential position counter and a PRNG whose state is derived
    entirely from the ``SamplingConfig`` seed material.  Calling
    ``should_record`` in the same order with the same seed always produces the
    same sequence of decisions, making the audit trail reproducible.

    Args:
        config: ``SamplingConfig`` describing the sampling policy and seed.
    """

    # LCG parameters (Knuth, TAOCP Vol.2):
    # a = 6364136223846793005, c = 1442695040888963407, m = 2^64
    _LCG_A: int = 6364136223846793005
    _LCG_C: int = 1442695040888963407
    _LCG_M: int = 2**64

    def __init__(self, config: SamplingConfig) -> None:
        self._config = config
        self._position: int = 0
        self._selected_count: int = 0
        self._seed_hash: str = hashlib.sha256(config.seed_material).hexdigest()
        # Derive initial PRNG state deterministically from seed material.
        self._prng_state: int = int.from_bytes(
            hashlib.sha256(config.seed_material + b"prng_v1").digest(), "big"
        )

    # ------------------------------------------------------------------
    # Internal PRNG
    # ------------------------------------------------------------------

    def _next_pseudo_random(self) -> float:
        """Advance the LCG and return a float in [0, 1).

        Uses the 64-bit LCG with well-known constants (same family as
        PCG/Knuth).  The state is updated in-place so consecutive calls
        produce an independent pseudo-random sequence seeded by the config.
        """
        self._prng_state = (
            self._LCG_A * self._prng_state + self._LCG_C
        ) % self._LCG_M
        return self._prng_state / self._LCG_M

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_record(self, inference_id: str) -> SampleDecision:
        """Decide whether to audit this inference.

        Must be called once per inference in arrival order to maintain the
        correct sequential position.  The decision is deterministic: the same
        *inference_id* at the same *position* with the same seed always
        produces the same ``selected`` value.

        Args:
            inference_id: Unique identifier for the inference (e.g. a UUID
                          or content-addressed hash of the request).

        Returns:
            A ``SampleDecision`` recording whether the inference was selected
            and the metadata needed to verify the decision later.
        """
        pos = self._position
        self._position += 1

        if self._config.method == SamplingMethod.BERNOULLI:
            selected = self._next_pseudo_random() < self._config.rate
        elif self._config.method == SamplingMethod.SYSTEMATIC:
            interval = max(1, round(1.0 / self._config.rate))
            selected = (pos % interval) == 0
        else:
            # RESERVOIR — approximate with Bernoulli for streaming use.
            selected = self._next_pseudo_random() < self._config.rate

        if selected:
            self._selected_count += 1

        return SampleDecision(
            inference_id=inference_id,
            selected=selected,
            method=self._config.method,
            seed_hash="sha256:" + self._seed_hash,
            position=pos,
            decided_at=datetime.now(timezone.utc),
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Current sampling statistics.

        Returns a dict with the following keys:

        - ``total_seen``: number of ``should_record`` calls made so far.
        - ``total_selected``: number of inferences selected for audit.
        - ``actual_rate``: ``total_selected / total_seen`` (0.0 if no calls).
        - ``target_rate``: the configured sampling rate.
        - ``method``: the method enum value string.
        - ``seed_hash``: ``sha256:<hex>`` of the seed material.
        - ``is_verifiable``: whether the sampler has a BSV txid seed.
        """
        total = self._position
        return {
            "total_seen": total,
            "total_selected": self._selected_count,
            "actual_rate": self._selected_count / total if total > 0 else 0.0,
            "target_rate": self._config.rate,
            "method": self._config.method.value,
            "seed_hash": "sha256:" + self._seed_hash,
            "is_verifiable": self._config.is_verifiable,
        }

    def verify_decision(self, decision: SampleDecision) -> bool:
        """Check that *decision* was produced using this sampler's seed.

        Compares the ``seed_hash`` embedded in *decision* against the hash
        of this sampler's seed material.  A mismatch means the decision was
        produced by a sampler with a different configuration.

        Args:
            decision: A ``SampleDecision`` to authenticate.

        Returns:
            True if *decision.seed_hash* matches this sampler's seed hash.
        """
        return decision.seed_hash == "sha256:" + self._seed_hash

    def reset(self) -> None:
        """Reset the sampler to its initial state.

        After calling ``reset()``, the next ``should_record`` call will
        produce the same decision as the very first call after construction.
        This allows replaying a stream to verify decisions recorded earlier.
        """
        self._position = 0
        self._selected_count = 0
        self._prng_state = int.from_bytes(
            hashlib.sha256(self._config.seed_material + b"prng_v1").digest(), "big"
        )


# ---------------------------------------------------------------------------
# Verifiable proof
# ---------------------------------------------------------------------------


@dataclass
class VerifiableSamplingProof:
    """Proof that a set of inferences was sampled according to a declared policy.

    Binds together the sampling configuration, the selected inference IDs,
    and a proof hash so that any recipient can verify the proof was not
    tampered with after generation.

    Attributes:
        config:                  The ``SamplingConfig`` used during sampling.
        total_inferences:        Total number of inferences seen in the stream.
        selected_inference_ids:  IDs of inferences that were selected for audit.
        selected_count:          Number of selected inferences (redundant with
                                 ``len(selected_inference_ids)`` but explicit
                                 for integrity checking).
        actual_rate:             Observed selection fraction.
        proof_hash:              SHA-256 of all proof fields in canonical form,
                                 prefixed with ``sha256:``.
        generated_at:            UTC timestamp when the proof was created.
    """

    config: SamplingConfig
    total_inferences: int
    selected_inference_ids: list[str]
    selected_count: int
    actual_rate: float
    proof_hash: str
    generated_at: datetime

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def generate(
        cls,
        sampler: AuditSampler,
        selected_ids: list[str],
    ) -> "VerifiableSamplingProof":
        """Create a proof from a completed sampling session.

        Computes a canonical hash over all material fields so the proof
        can be verified later with ``verify()``.

        Args:
            sampler:      The ``AuditSampler`` that produced the decisions.
            selected_ids: List of inference IDs that were marked ``selected``.

        Returns:
            A ``VerifiableSamplingProof`` whose ``proof_hash`` commits to all
            fields, including the seed and the sorted set of selected IDs.
        """
        stats = sampler.stats
        fields: dict[str, Any] = {
            "seed_hash": stats["seed_hash"],
            "total_inferences": stats["total_seen"],
            "selected_count": stats["total_selected"],
            "actual_rate": stats["actual_rate"],
            "target_rate": stats["target_rate"],
            "method": stats["method"],
            "selected_ids_hash": hashlib.sha256(
                "|".join(sorted(selected_ids)).encode()
            ).hexdigest(),
        }
        proof_hash = "sha256:" + hash_object(fields)
        return cls(
            config=sampler._config,
            total_inferences=stats["total_seen"],
            selected_inference_ids=selected_ids,
            selected_count=stats["total_selected"],
            actual_rate=stats["actual_rate"],
            proof_hash=proof_hash,
            generated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> bool:
        """Verify the proof has not been tampered with.

        Recomputes the ``proof_hash`` from the stored fields and compares it
        against ``self.proof_hash``.  Any modification to ``total_inferences``,
        ``selected_count``, ``actual_rate``, ``config``, or the list of
        selected IDs will cause verification to fail.

        Returns:
            True if the proof is internally consistent; False otherwise.
        """
        seed_hash_value: str | None = (
            "sha256:" + hashlib.sha256(self.config.seed_material).hexdigest()
        )
        fields: dict[str, Any] = {
            "seed_hash": seed_hash_value,
            "total_inferences": self.total_inferences,
            "selected_count": self.selected_count,
            "actual_rate": self.actual_rate,
            "target_rate": self.config.rate,
            "method": self.config.method.value,
            "selected_ids_hash": hashlib.sha256(
                "|".join(sorted(self.selected_inference_ids)).encode()
            ).hexdigest(),
        }
        expected = "sha256:" + hash_object(fields)
        return self.proof_hash == expected
