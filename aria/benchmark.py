"""
aria.benchmark — Benchmark Proof Registry for ARIA.

When an AI team claims "our model scores 87% on MMLU", that claim can be
anchored on BSV so it is verifiable by anyone.  This fights benchmark
gaming and fabrication by creating a cryptographic, timestamped record of
benchmark results that is independent of the team reporting them.

The on-chain payload follows the BRC-121 convention:

.. code-block:: json

    {
        "type": "BENCHMARK_ANCHOR",
        "brc": "121",
        "result_hash": "sha256:<64hex>",
        "model_id": "<human-readable name>",
        "suite": "<BenchmarkSuite value>",
        "score": 0.87,
        "num_samples": 14042
    }

The ``result_hash`` is the SHA-256 of the canonical JSON of all
:class:`BenchmarkResult` fields, making every claim tamper-evident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aria.core.hasher import canonical_json, hash_object


# ---------------------------------------------------------------------------
# Enum: benchmark suites
# ---------------------------------------------------------------------------


class BenchmarkSuite(str, Enum):
    """Well-known AI benchmark suites supported by the ARIA registry.

    Values are lowercase strings that match the canonical name used in
    published leaderboards so that cross-team comparisons are unambiguous.
    """

    MMLU = "mmlu"
    HUMANEVAL = "humaneval"
    BIGBENCH = "bigbench"
    HELLASWAG = "hellaswag"
    ARC = "arc"
    TRUTHFULQA = "truthfulqa"
    GSMATH = "gsmath"
    MBPP = "mbpp"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """A single benchmark evaluation result to be anchored on BSV.

    Attributes:
        model_hash:    SHA-256 of the model file (use
                       :func:`aria.core.hasher.hash_file`).
        model_id:      Human-readable model name, e.g. ``"gpt-4-turbo"``.
        suite:         Which benchmark suite was run.
        score:         Normalised score in [0.0, 1.0].
        num_samples:   Number of questions / tasks evaluated.
        subset:        Optional sub-category, e.g. ``"STEM"`` for MMLU.
        metadata:      Arbitrary extra key-value pairs (must be JSON-safe).
        evaluated_at:  UTC timestamp of the evaluation (auto-set if omitted).
    """

    model_hash: str
    model_id: str
    suite: BenchmarkSuite
    score: float
    num_samples: int
    subset: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"score must be in [0.0, 1.0], got {self.score}"
            )
        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be > 0, got {self.num_samples}"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def result_hash(self) -> str:
        """Return the SHA-256 hash of this result's canonical representation.

        The hash covers every field so that any post-hoc modification of
        score, model_hash, subset, or metadata is detectable.

        Returns:
            str in the format ``"sha256:<64 hex chars>"``.
        """
        return hash_object(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this result.

        The ``evaluated_at`` timestamp is serialised as an ISO-8601 string
        in UTC so the dict can be round-tripped through ``json.dumps``.

        Returns:
            dict with all BenchmarkResult fields.
        """
        return {
            "evaluated_at": self.evaluated_at.isoformat(),
            "metadata": self.metadata,
            "model_hash": self.model_hash,
            "model_id": self.model_id,
            "num_samples": self.num_samples,
            "score": self.score,
            "subset": self.subset,
            "suite": self.suite.value,
        }


# ---------------------------------------------------------------------------
# BenchmarkAnchor
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkAnchor:
    """An anchored benchmark claim — the output of :meth:`BenchmarkRegistry.anchor`.

    Attributes:
        result:       The original :class:`BenchmarkResult`.
        result_hash:  SHA-256 hash of the result at anchoring time.
        txid:         BSV transaction ID, or ``None`` if local-only.
        anchored_at:  UTC timestamp of the anchoring operation.
    """

    result: BenchmarkResult
    result_hash: str
    txid: str | None
    anchored_at: datetime

    def is_on_chain(self) -> bool:
        """Return ``True`` if this anchor has been broadcast to BSV.

        Returns:
            bool — ``True`` iff ``txid`` is a non-empty string.
        """
        return self.txid is not None and bool(self.txid)


# ---------------------------------------------------------------------------
# BenchmarkRegistry
# ---------------------------------------------------------------------------


class BenchmarkRegistry:
    """In-memory registry for anchored benchmark claims.

    Anchors can optionally be broadcast to BSV via any object that exposes a
    ``broadcast(payload_bytes: bytes)`` method and returns an object with a
    ``txid`` attribute, or a plain string txid.

    Args:
        broadcaster: Optional broadcaster (sync or async-capable object).
                     Pass ``None`` (default) for local-only operation.
        db_path:     Reserved for future persistent storage.  Currently unused;
                     all data lives in an in-memory dict keyed by result_hash.
    """

    def __init__(
        self,
        broadcaster: Any | None = None,
        db_path: str = ":memory:",
    ) -> None:
        self._broadcaster = broadcaster
        self._db_path = db_path
        # Primary store: result_hash → BenchmarkAnchor
        self._store: dict[str, BenchmarkAnchor] = {}
        # Secondary index: model_hash → list[result_hash]
        self._model_index: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def anchor(self, result: BenchmarkResult) -> BenchmarkAnchor:
        """Hash *result*, optionally broadcast an OP_RETURN payload, and store.

        Args:
            result: The :class:`BenchmarkResult` to anchor.

        Returns:
            :class:`BenchmarkAnchor` with ``result_hash`` set and ``txid``
            populated if a broadcaster is configured.
        """
        rh = result.result_hash()

        txid: str | None = None
        if self._broadcaster is not None:
            payload = {
                "type": "BENCHMARK_ANCHOR",
                "brc": "121",
                "result_hash": rh,
                "model_id": result.model_id,
                "suite": result.suite.value,
                "score": result.score,
                "num_samples": result.num_samples,
            }
            payload_bytes = canonical_json(payload)
            broadcast_result = self._broadcaster.broadcast(payload_bytes)
            if broadcast_result is not None:
                if hasattr(broadcast_result, "txid"):
                    txid = broadcast_result.txid
                elif isinstance(broadcast_result, str) and broadcast_result:
                    txid = broadcast_result

        anchor = BenchmarkAnchor(
            result=result,
            result_hash=rh,
            txid=txid,
            anchored_at=datetime.now(timezone.utc),
        )

        # Store
        self._store[rh] = anchor
        self._model_index.setdefault(result.model_hash, [])
        if rh not in self._model_index[result.model_hash]:
            self._model_index[result.model_hash].append(rh)

        return anchor

    def verify(self, anchor: BenchmarkAnchor) -> bool:
        """Verify the integrity of *anchor* by recomputing its result hash.

        Args:
            anchor: The :class:`BenchmarkAnchor` to verify.

        Returns:
            ``True`` iff the stored ``result_hash`` matches a fresh
            computation over ``anchor.result``.
        """
        return anchor.result.result_hash() == anchor.result_hash

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_anchors(self, model_hash: str) -> list[BenchmarkAnchor]:
        """Return all anchors for a given model hash.

        Args:
            model_hash: SHA-256 hash of the model file.

        Returns:
            List of :class:`BenchmarkAnchor` objects (may be empty).
        """
        hashes = self._model_index.get(model_hash, [])
        return [self._store[h] for h in hashes if h in self._store]

    def get_best(
        self,
        suite: BenchmarkSuite,
        model_hash: str | None = None,
    ) -> BenchmarkAnchor | None:
        """Return the anchor with the highest score for a given suite.

        Args:
            suite:       The :class:`BenchmarkSuite` to filter by.
            model_hash:  If given, restrict the search to this model.

        Returns:
            The :class:`BenchmarkAnchor` with the highest ``score``, or
            ``None`` if no matching anchors exist.
        """
        if model_hash is not None:
            candidates = self.get_anchors(model_hash)
        else:
            candidates = list(self._store.values())

        filtered = [a for a in candidates if a.result.suite == suite]
        if not filtered:
            return None

        return max(filtered, key=lambda a: a.result.score)

    def compare(
        self,
        anchor_a: BenchmarkAnchor,
        anchor_b: BenchmarkAnchor,
    ) -> dict[str, Any]:
        """Compare two anchors and return a summary dict.

        Args:
            anchor_a: First anchor.
            anchor_b: Second anchor.

        Returns:
            dict with keys:

            - ``score_delta`` (float): ``anchor_a.score - anchor_b.score``
            - ``same_model`` (bool): whether both anchors share the same
              ``model_hash``
            - ``suite_match`` (bool): whether both anchors are for the same
              suite
            - ``suite_a`` (str): suite value of anchor_a
            - ``suite_b`` (str): suite value of anchor_b
            - ``score_a`` (float): score of anchor_a
            - ``score_b`` (float): score of anchor_b
        """
        score_a = anchor_a.result.score
        score_b = anchor_b.result.score
        return {
            "score_delta": round(score_a - score_b, 10),
            "same_model": anchor_a.result.model_hash == anchor_b.result.model_hash,
            "suite_match": anchor_a.result.suite == anchor_b.result.suite,
            "suite_a": anchor_a.result.suite.value,
            "suite_b": anchor_b.result.suite.value,
            "score_a": score_a,
            "score_b": score_b,
        }
