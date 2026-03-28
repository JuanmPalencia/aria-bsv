"""Proof aggregation — combine N inference proofs into a single on-chain commitment.

Aggregation strategies
----------------------

MerkleAggregator (production-ready):
  Builds a Merkle tree over individual proof digests.  The root is a single
  32-byte hash that commits to all proofs.  Inclusion of any individual proof
  can be verified in O(log N) time.

  Suitable for: all production use cases today.
  On-chain footprint: one sha256 hash.

NovaAggregator (remote proving service):
  Delegates Nova folding to a remote HTTP proving service.  Nova (Kothapalli
  et al., 2022) generates a single succinct proof that verifies N inference
  proofs simultaneously in O(1) verifier time.

  Requires a running Nova proving service (see docs/nova-service.md).
  Falls back gracefully with ARIAZKError if the service is unreachable.

PlonkRecursiveAggregator (remote proving service):
  Delegates recursive Plonk aggregation to a remote HTTP service.
  Each fold reduces N proofs to a single proof of constant size.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from ..core.hasher import canonical_json
from ..core.merkle import ARIAMerkleTree, verify_proof as verify_merkle_proof
from .base import ZKProof


# ---------------------------------------------------------------------------
# AggregateProof
# ---------------------------------------------------------------------------


@dataclass
class AggregateProof:
    """Aggregation of N individual ZKProofs into a single compact commitment.

    Included in the EPOCH_CLOSE BSV payload so regulators can independently
    verify that the operator committed to all N inference proofs at close time.

    Attributes:
        proofs_merkle_root: Merkle root of all individual proof digests.
        n_proofs:           Number of proofs aggregated.
        aggregation_scheme: ``"merkle"``, ``"nova"``, or ``"plonk_recursive"``.
        aggregate_bytes:    Serialized aggregate (scheme-dependent).
        epoch_id:           The epoch these proofs belong to.
    """

    proofs_merkle_root: str
    n_proofs: int
    aggregation_scheme: str
    aggregate_bytes: bytes
    epoch_id: str

    def digest(self) -> str:
        """SHA-256 fingerprint of the aggregate commitment."""
        return "sha256:" + hashlib.sha256(self.aggregate_bytes).hexdigest()

    def to_dict(self) -> dict:
        return {
            "proofs_merkle_root": self.proofs_merkle_root,
            "n_proofs": self.n_proofs,
            "aggregation_scheme": self.aggregation_scheme,
            "aggregate_digest": self.digest(),
            "epoch_id": self.epoch_id,
        }


# ---------------------------------------------------------------------------
# Aggregator interface
# ---------------------------------------------------------------------------


class ProofAggregatorInterface(ABC):
    """Abstract aggregator interface."""

    @abstractmethod
    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        """Aggregate N proofs into a single AggregateProof."""

    @abstractmethod
    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Verify that an AggregateProof commits to exactly the given proofs."""

    def verify_membership(
        self,
        agg: AggregateProof,
        proof: ZKProof,
        all_proofs: list[ZKProof],
    ) -> bool:
        """Verify that a single proof is a member of the aggregate.

        Default implementation recomputes the aggregate and checks membership.
        Override for more efficient implementations (e.g., Merkle path lookup).
        """
        digests = [p.digest() for p in all_proofs]
        return proof.digest() in digests and self.verify_aggregate(agg, all_proofs)


# ---------------------------------------------------------------------------
# MerkleAggregator
# ---------------------------------------------------------------------------


class MerkleAggregator(ProofAggregatorInterface):
    """Production-ready Merkle-based proof aggregation.

    Builds an ARIAMerkleTree (RFC 6962, second-preimage protected) over the
    digest of each individual ZKProof.  The root is stored in the EPOCH_CLOSE
    BSV transaction.

    Membership proof: use ``aria.core.merkle.verify_proof`` with the Merkle
    path for any individual proof digest.

    On-chain footprint: one SHA-256 hash per epoch (no matter how many
    inferences: 10, 10_000, or 10_000_000).
    """

    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        if not proofs:
            empty_root = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            return AggregateProof(
                proofs_merkle_root=empty_root,
                n_proofs=0,
                aggregation_scheme="merkle",
                aggregate_bytes=bytes.fromhex(empty_root[7:]),
                epoch_id=epoch_id,
            )

        tree = ARIAMerkleTree()
        for p in sorted(proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())

        root = tree.root()
        root_hex = bytes.fromhex(root[7:])  # strip "sha256:" prefix

        return AggregateProof(
            proofs_merkle_root=root,
            n_proofs=len(proofs),
            aggregation_scheme="merkle",
            aggregate_bytes=root_hex,
            epoch_id=epoch_id,
        )

    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Recompute the Merkle root and compare to the aggregate."""
        if agg.aggregation_scheme != "merkle":
            return False

        recomputed = self.aggregate(proofs, agg.epoch_id)
        return recomputed.proofs_merkle_root == agg.proofs_merkle_root

    def membership_path(self, proof: ZKProof, all_proofs: list[ZKProof]):  # type: ignore[return]
        """Return the Merkle path for a specific proof (for independent verification)."""
        tree = ARIAMerkleTree()
        for p in sorted(all_proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())
        return tree.proof(proof.digest())


# ---------------------------------------------------------------------------
# NovaAggregator — remote HTTP proving service
# ---------------------------------------------------------------------------


class NovaAggregator(ProofAggregatorInterface):
    """Nova folding scheme aggregator via remote HTTP proving service.

    Nova (Kothapalli, Setty, Tzialla, 2022) is an incrementally verifiable
    computation (IVC) scheme that folds N proof instances into a single
    succinct proof with O(1) verifier cost:

      - One ~2KB proof verifies all N inference proofs simultaneously
      - Sub-millisecond verification regardless of N
      - No trusted setup required

    This implementation delegates proof generation to a remote Nova proving
    service, following the same pattern as BRC-100 wallet delegation.
    The service receives proof digests and returns an aggregated Nova proof.

    Args:
        service_url:  Base URL of a Nova proving service.
                      Expected endpoints:
                        POST /aggregate  — accepts {proofs: [...], epoch_id: str}
                        POST /verify     — accepts {aggregate: {...}, proofs: [...]}
        api_key:      Optional Bearer token for service authentication.
        timeout:      HTTP timeout in seconds (default 120 for large proof sets).

    References:
        https://eprint.iacr.org/2021/370 (Nova paper)
        https://github.com/microsoft/nova (Rust reference implementation)

    Example::

        agg = NovaAggregator(service_url="https://nova.aria-bsv.io")
        result = agg.aggregate(proofs, epoch_id="ep_123")
    """

    def __init__(
        self,
        service_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._url = service_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout

    def _proofs_to_payload(self, proofs: list[ZKProof], epoch_id: str) -> dict[str, Any]:
        return {
            "epoch_id": epoch_id,
            "proofs": [
                {
                    "record_id": p.record_id,
                    "digest": p.digest(),
                    "proof_bytes": p.proof_bytes.hex() if p.proof_bytes else "",
                    "tier": p.tier,
                }
                for p in proofs
            ],
        }

    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        """Delegate proof aggregation to the Nova proving service.

        Raises:
            ARIAZKError: If the service is unreachable or returns an error.
        """
        from ..core.errors import ARIAZKError

        payload = self._proofs_to_payload(proofs, epoch_id)
        # Build a Merkle root locally so we have it regardless of service result
        tree = ARIAMerkleTree()
        for p in sorted(proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())
        merkle_root = tree.root() if proofs else "sha256:" + hashlib.sha256(b"").hexdigest()

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._url}/aggregate",
                    headers=self._headers,
                    content=json.dumps(payload),
                )
                if resp.status_code != 200:
                    raise ARIAZKError(
                        f"Nova service error {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
        except ARIAZKError:
            raise
        except Exception as exc:
            raise ARIAZKError(f"Nova service unreachable: {exc}") from exc

        aggregate_hex: str = data.get("aggregate_bytes", "")
        aggregate_bytes = bytes.fromhex(aggregate_hex) if aggregate_hex else b""
        return AggregateProof(
            proofs_merkle_root=data.get("proofs_merkle_root", merkle_root),
            n_proofs=len(proofs),
            aggregation_scheme="nova",
            aggregate_bytes=aggregate_bytes,
            epoch_id=epoch_id,
        )

    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Verify an aggregate proof via the Nova proving service.

        Returns False (not raises) on service errors to allow graceful degradation.
        """
        from ..core.errors import ARIAZKError

        payload = {
            "aggregate": {
                "proofs_merkle_root": agg.proofs_merkle_root,
                "n_proofs": agg.n_proofs,
                "aggregation_scheme": agg.aggregation_scheme,
                "aggregate_bytes": agg.aggregate_bytes.hex(),
                "epoch_id": agg.epoch_id,
            },
            "proofs": self._proofs_to_payload(proofs, agg.epoch_id)["proofs"],
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._url}/verify",
                    headers=self._headers,
                    content=json.dumps(payload),
                )
                if resp.status_code != 200:
                    return False
                return bool(resp.json().get("valid", False))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PlonkRecursiveAggregator — remote HTTP proving service
# ---------------------------------------------------------------------------


class PlonkRecursiveAggregator(ProofAggregatorInterface):
    """Recursive Plonk aggregator via remote HTTP proving service.

    Each aggregation step folds the previous aggregate proof with the next
    batch of inference proofs, producing a single constant-size proof that
    verifies the entire epoch history.  Uses Halo2 recursive Plonk circuits.

    This implementation delegates to a remote proving service, allowing the
    heavy proof generation work to run on dedicated hardware without blocking
    the ARIA audit pipeline.

    Args:
        service_url:  Base URL of a Plonk recursive proving service.
                      Expected endpoints:
                        POST /aggregate  — {proofs: [...], epoch_id: str, prev_aggregate: {...}|null}
                        POST /verify     — {aggregate: {...}, proofs: [...]}
        api_key:      Optional Bearer token.
        timeout:      HTTP timeout in seconds (default 180).

    Example::

        agg = PlonkRecursiveAggregator(service_url="https://plonk.aria-bsv.io")
        result = agg.aggregate(proofs, epoch_id="ep_123")
    """

    def __init__(
        self,
        service_url: str,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self._url = service_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        self._prev_aggregate: AggregateProof | None = None

    def _proofs_to_list(self, proofs: list[ZKProof]) -> list[dict[str, Any]]:
        return [
            {
                "record_id": p.record_id,
                "digest": p.digest(),
                "proof_bytes": p.proof_bytes.hex() if p.proof_bytes else "",
                "tier": p.tier,
            }
            for p in proofs
        ]

    def aggregate(self, proofs: list[ZKProof], epoch_id: str) -> AggregateProof:
        """Aggregate *proofs* recursively via the Plonk service.

        If a previous aggregate exists (from a prior call), it is folded into
        the new aggregate, providing incremental proof accumulation.

        Raises:
            ARIAZKError: If the service is unreachable or returns an error.
        """
        from ..core.errors import ARIAZKError

        tree = ARIAMerkleTree()
        for p in sorted(proofs, key=lambda x: x.record_id or ""):
            tree.add(p.digest())
        merkle_root = tree.root() if proofs else "sha256:" + hashlib.sha256(b"").hexdigest()

        payload: dict[str, Any] = {
            "epoch_id": epoch_id,
            "proofs": self._proofs_to_list(proofs),
            "prev_aggregate": None,
        }
        if self._prev_aggregate is not None:
            payload["prev_aggregate"] = {
                "proofs_merkle_root": self._prev_aggregate.proofs_merkle_root,
                "n_proofs": self._prev_aggregate.n_proofs,
                "aggregate_bytes": self._prev_aggregate.aggregate_bytes.hex(),
                "epoch_id": self._prev_aggregate.epoch_id,
            }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._url}/aggregate",
                    headers=self._headers,
                    content=json.dumps(payload),
                )
                if resp.status_code != 200:
                    raise ARIAZKError(
                        f"Plonk service error {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
        except ARIAZKError:
            raise
        except Exception as exc:
            raise ARIAZKError(f"Plonk service unreachable: {exc}") from exc

        agg_hex: str = data.get("aggregate_bytes", "")
        agg = AggregateProof(
            proofs_merkle_root=data.get("proofs_merkle_root", merkle_root),
            n_proofs=len(proofs),
            aggregation_scheme="plonk_recursive",
            aggregate_bytes=bytes.fromhex(agg_hex) if agg_hex else b"",
            epoch_id=epoch_id,
        )
        self._prev_aggregate = agg
        return agg

    def verify_aggregate(self, agg: AggregateProof, proofs: list[ZKProof]) -> bool:
        """Verify a recursive Plonk aggregate via the proving service."""
        payload = {
            "aggregate": {
                "proofs_merkle_root": agg.proofs_merkle_root,
                "n_proofs": agg.n_proofs,
                "aggregation_scheme": agg.aggregation_scheme,
                "aggregate_bytes": agg.aggregate_bytes.hex(),
                "epoch_id": agg.epoch_id,
            },
            "proofs": self._proofs_to_list(proofs),
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._url}/verify",
                    headers=self._headers,
                    content=json.dumps(payload),
                )
                if resp.status_code != 200:
                    return False
                return bool(resp.json().get("valid", False))
        except Exception:
            return False
