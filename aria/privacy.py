"""
aria.privacy — Differential Privacy budget tracking for AI inference.

Tracks the cumulative privacy budget (epsilon) consumed by inference
queries under differential privacy guarantees. Alerts when the budget
is approaching exhaustion and enforces hard stops when exceeded.

Compatible with any DP mechanism — Laplace, Gaussian, or custom noise.
Integrates with ARIA's BRC-121 epoch system to anchor budget consumption
records on BSV.

Usage::

    from aria.privacy import PrivacyBudget, DPMechanism, PrivacyAccountant

    accountant = PrivacyAccountant(
        epsilon_total=1.0,    # Total epsilon budget (e.g. ε=1.0)
        delta=1e-5,           # δ for (ε,δ)-DP
        warn_at=0.8,          # Warn when 80% consumed
    )

    # Record each query's privacy cost
    accountant.record_query(epsilon=0.01, delta=1e-6, mechanism="laplace")

    status = accountant.status()
    print(status)
    # PrivacyStatus: ε_used=0.01/1.0 (1.0%)  δ_used=1e-06  SAFE
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

_log = logging.getLogger(__name__)


class DPMechanism(str, Enum):
    LAPLACE   = "laplace"
    GAUSSIAN  = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMISED_RESPONSE = "randomised_response"
    CUSTOM    = "custom"


class PrivacyBudgetStatus(str, Enum):
    SAFE      = "SAFE"       # < warn_at threshold
    WARNING   = "WARNING"    # ≥ warn_at, < 100%
    EXHAUSTED = "EXHAUSTED"  # ≥ 100%
    EXCEEDED  = "EXCEEDED"   # > 100% (hard enforcement triggered)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DPQuery:
    """A single differentially-private query that consumed budget."""
    query_id:    str
    epsilon:     float
    delta:       float
    mechanism:   DPMechanism
    sensitivity: float = 1.0
    timestamp:   str = ""
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class PrivacyStatus:
    """Current state of the privacy budget."""
    epsilon_used:    float
    epsilon_total:   float
    delta_used:      float
    delta_total:     float
    n_queries:       int
    status:          PrivacyBudgetStatus
    warn_threshold:  float

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.epsilon_total - self.epsilon_used)

    @property
    def epsilon_pct(self) -> float:
        if self.epsilon_total <= 0:
            return 100.0
        return (self.epsilon_used / self.epsilon_total) * 100

    def __str__(self) -> str:
        return (
            f"PrivacyStatus: ε_used={self.epsilon_used:.4f}/{self.epsilon_total}  "
            f"({self.epsilon_pct:.1f}%)  "
            f"δ_used={self.delta_used:.2e}  "
            f"queries={self.n_queries}  "
            f"{self.status.value}"
        )


# ---------------------------------------------------------------------------
# Composition theorems
# ---------------------------------------------------------------------------

def compose_basic(queries: list[DPQuery]) -> tuple[float, float]:
    """Basic composition: ε = Σεᵢ, δ = Σδᵢ."""
    eps = sum(q.epsilon for q in queries)
    delta = sum(q.delta for q in queries)
    return eps, delta


def compose_advanced(
    queries: list[DPQuery],
    delta_prime: float = 1e-6,
) -> tuple[float, float]:
    """Advanced composition (Dwork et al.): tighter epsilon bound for many queries.

    For k queries each (εᵢ, δᵢ)-DP:
    (ε', δ + k*δ')-DP where ε' = sqrt(2k*ln(1/δ')) * max_ε + k*max_ε*(e^max_ε - 1)
    """
    if not queries:
        return 0.0, 0.0
    k = len(queries)
    max_eps = max(q.epsilon for q in queries)
    delta_sum = sum(q.delta for q in queries)
    delta_total = delta_sum + k * delta_prime

    eps_advanced = (
        math.sqrt(2 * k * math.log(1 / delta_prime)) * max_eps
        + k * max_eps * (math.exp(max_eps) - 1)
    )
    return eps_advanced, delta_total


def laplace_epsilon(sensitivity: float, noise_scale: float) -> float:
    """Compute ε for Laplace mechanism: ε = sensitivity / scale."""
    if noise_scale <= 0:
        return float("inf")
    return sensitivity / noise_scale


def gaussian_epsilon(sensitivity: float, sigma: float, delta: float) -> float:
    """Compute ε for Gaussian mechanism (approximate): ε ≈ sensitivity*sqrt(2*ln(1.25/δ)) / σ."""
    if sigma <= 0 or delta <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma


# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------

class PrivacyAccountant:
    """Tracks cumulative privacy budget consumption.

    Args:
        epsilon_total:  Total epsilon budget available.
        delta:          Maximum allowed delta (δ) for (ε,δ)-DP.
        warn_at:        Fraction of budget at which to warn (default 0.8 = 80%).
        hard_stop:      If True, raise PrivacyBudgetExhaustedError when exceeded.
        composition:    Composition theorem: "basic" or "advanced".
    """

    def __init__(
        self,
        epsilon_total: float,
        delta: float = 1e-5,
        warn_at: float = 0.8,
        hard_stop: bool = False,
        composition: str = "basic",
    ) -> None:
        if epsilon_total <= 0:
            raise ValueError("epsilon_total must be > 0")
        self._epsilon_total = epsilon_total
        self._delta_total   = delta
        self._warn_at       = warn_at
        self._hard_stop     = hard_stop
        self._composition   = composition
        self._queries:      list[DPQuery] = []
        self._query_counter = 0

    def record_query(
        self,
        epsilon: float,
        delta: float = 0.0,
        mechanism: DPMechanism | str = DPMechanism.LAPLACE,
        sensitivity: float = 1.0,
        metadata: dict | None = None,
    ) -> DPQuery:
        """Record a privacy-consuming query.

        Args:
            epsilon:     Privacy cost of this query (ε).
            delta:       Privacy cost of this query (δ).
            mechanism:   DP mechanism used.
            sensitivity: Query sensitivity (for documentation).
            metadata:    Optional extra metadata.

        Returns:
            DPQuery record.

        Raises:
            PrivacyBudgetExhaustedError: If hard_stop=True and budget exceeded.
        """
        self._query_counter += 1
        mech = DPMechanism(mechanism) if isinstance(mechanism, str) else mechanism
        query = DPQuery(
            query_id=f"q-{self._query_counter:06d}",
            epsilon=epsilon,
            delta=delta,
            mechanism=mech,
            sensitivity=sensitivity,
            metadata=metadata or {},
        )
        self._queries.append(query)

        # Check hard stop
        if self._hard_stop:
            eps_used, _ = self._compute_used()
            if eps_used > self._epsilon_total:
                raise PrivacyBudgetExhaustedError(
                    f"Privacy budget exhausted: ε_used={eps_used:.4f} > ε_total={self._epsilon_total}"
                )

        return query

    def status(self) -> PrivacyStatus:
        """Return current budget status."""
        eps_used, delta_used = self._compute_used()
        pct = eps_used / self._epsilon_total if self._epsilon_total > 0 else 1.0

        if eps_used > self._epsilon_total:
            budget_status = PrivacyBudgetStatus.EXCEEDED
        elif pct >= 1.0:
            budget_status = PrivacyBudgetStatus.EXHAUSTED
        elif pct >= self._warn_at:
            budget_status = PrivacyBudgetStatus.WARNING
        else:
            budget_status = PrivacyBudgetStatus.SAFE

        return PrivacyStatus(
            epsilon_used=round(eps_used, 8),
            epsilon_total=self._epsilon_total,
            delta_used=delta_used,
            delta_total=self._delta_total,
            n_queries=len(self._queries),
            status=budget_status,
            warn_threshold=self._warn_at,
        )

    def is_safe(self) -> bool:
        """Return True if budget is not yet exhausted."""
        s = self.status()
        return s.status not in (PrivacyBudgetStatus.EXHAUSTED, PrivacyBudgetStatus.EXCEEDED)

    def reset(self) -> None:
        """Reset the accountant (start a new privacy epoch)."""
        self._queries.clear()
        self._query_counter = 0

    @property
    def queries(self) -> list[DPQuery]:
        return list(self._queries)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_used(self) -> tuple[float, float]:
        if not self._queries:
            return 0.0, 0.0
        if self._composition == "advanced":
            return compose_advanced(self._queries)
        return compose_basic(self._queries)


class PrivacyBudgetExhaustedError(Exception):
    """Raised when the privacy budget is exceeded and hard_stop=True."""
