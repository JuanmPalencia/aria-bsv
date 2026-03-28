"""
aria.cost_tracker — Token-based cost estimation for AI inference auditing.

Tracks cost per inference record using configurable token price tables.
Aggregates per epoch, model, and system. Budget alerts when thresholds exceeded.

Usage::

    from aria.cost_tracker import CostTracker

    tracker = CostTracker(storage, price_table={
        "gpt-4o":             {"input": 5.00,  "output": 15.00},  # per 1M tokens
        "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    })

    report = tracker.epoch_cost("epoch-model-v1")
    print(report)
    # CostReport: epoch=epoch-model-v1  total=$0.2340  records=120

    budget_ok = tracker.within_budget("epoch-model-v1", budget_usd=1.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)

# Default price table: USD per 1 million tokens
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o":                {"input": 5.00,   "output": 15.00},
    "gpt-4o-mini":           {"input": 0.15,   "output": 0.60},
    "gpt-4-turbo":           {"input": 10.00,  "output": 30.00},
    "gpt-3.5-turbo":         {"input": 0.50,   "output": 1.50},
    # Anthropic
    "claude-opus-4-6":       {"input": 15.00,  "output": 75.00},
    "claude-sonnet-4-6":     {"input": 3.00,   "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    # Fallback
    "__default__":           {"input": 1.00,   "output": 3.00},
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InferenceCost:
    """Cost breakdown for a single inference record."""
    record_id:     str
    model_id:      str
    input_tokens:  int
    output_tokens: int
    input_cost:    float    # USD
    output_cost:   float    # USD

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class CostReport:
    """Aggregated cost report for an epoch."""
    epoch_id:      str
    records:       list[InferenceCost] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(r.total_cost for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def cost_by_model(self) -> dict[str, float]:
        acc: dict[str, float] = {}
        for r in self.records:
            acc[r.model_id] = acc.get(r.model_id, 0.0) + r.total_cost
        return acc

    def __str__(self) -> str:
        return (
            f"CostReport: epoch={self.epoch_id}  "
            f"total=${self.total_usd:.4f}  "
            f"records={len(self.records)}  "
            f"tokens(in={self.total_input_tokens}, out={self.total_output_tokens})"
        )


@dataclass
class SystemCostReport:
    """Aggregated cost report across multiple epochs."""
    system_id:   str
    epoch_costs: list[CostReport] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(e.total_usd for e in self.epoch_costs)

    @property
    def cost_by_model(self) -> dict[str, float]:
        acc: dict[str, float] = {}
        for ep in self.epoch_costs:
            for model, cost in ep.cost_by_model.items():
                acc[model] = acc.get(model, 0.0) + cost
        return acc


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Estimates and aggregates inference cost from ARIA storage records.

    Args:
        storage:     Any StorageInterface implementation.
        price_table: Dict mapping model_id → {input: float, output: float}
                     (USD per 1M tokens). Merged with DEFAULT_PRICES.
        default_input_tokens:  Token count assumed when not in metadata (default 500).
        default_output_tokens: Token count assumed when not in metadata (default 150).
    """

    def __init__(
        self,
        storage: "StorageInterface",
        price_table: dict[str, dict[str, float]] | None = None,
        default_input_tokens: int = 500,
        default_output_tokens: int = 150,
    ) -> None:
        self._storage = storage
        self._prices: dict[str, dict[str, float]] = {**DEFAULT_PRICES}
        if price_table:
            self._prices.update(price_table)
        self._default_in = default_input_tokens
        self._default_out = default_output_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def epoch_cost(self, epoch_id: str) -> CostReport:
        """Compute cost for all records in an epoch."""
        records = self._storage.list_records_by_epoch(epoch_id)
        costs = [self._record_cost(r) for r in records]
        return CostReport(epoch_id=epoch_id, records=costs)

    def system_cost(self, epoch_ids: list[str], system_id: str = "") -> SystemCostReport:
        """Aggregate cost across multiple epochs."""
        epoch_costs = [self.epoch_cost(eid) for eid in epoch_ids]
        return SystemCostReport(system_id=system_id, epoch_costs=epoch_costs)

    def within_budget(self, epoch_id: str, budget_usd: float) -> bool:
        """Return True if epoch cost is within the specified USD budget."""
        report = self.epoch_cost(epoch_id)
        return report.total_usd <= budget_usd

    def price_for(self, model_id: str) -> dict[str, float]:
        """Return price dict for a model, falling back to __default__."""
        if model_id in self._prices:
            return self._prices[model_id]
        # Partial key match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
        for key in self._prices:
            if key != "__default__" and model_id.startswith(key):
                return self._prices[key]
        return self._prices["__default__"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_cost(self, record: Any) -> InferenceCost:
        model_id = str(getattr(record, "model_id", "") or "")
        metadata = getattr(record, "metadata", {}) or {}

        # Extract token counts from metadata (OpenAI / Anthropic format)
        usage = metadata.get("usage") or {}
        in_tok = (
            _coerce_int(usage.get("prompt_tokens"))
            or _coerce_int(usage.get("input_tokens"))
            or self._default_in
        )
        out_tok = (
            _coerce_int(usage.get("completion_tokens"))
            or _coerce_int(usage.get("output_tokens"))
            or self._default_out
        )

        prices = self.price_for(model_id)
        in_cost  = (in_tok  / 1_000_000) * prices["input"]
        out_cost = (out_tok / 1_000_000) * prices["output"]

        return InferenceCost(
            record_id=str(getattr(record, "record_id", "") or ""),
            model_id=model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            input_cost=in_cost,
            output_cost=out_cost,
        )


def _coerce_int(value: Any) -> int:
    try:
        v = int(value)
        return v if v > 0 else 0
    except (TypeError, ValueError):
        return 0
