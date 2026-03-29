"""
aria.cost_estimator — Estimate BSV transaction costs for ARIA auditing.

Provides cost projections for epoch anchoring based on record count,
transaction size estimates, and current BSV fee rates.

Usage::

    from aria.cost_estimator import CostEstimator

    est = CostEstimator(network="mainnet")
    result = est.estimate(records=10_000, epochs=1)
    print(result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# BSV fee rate: 1 sat/byte (standard BSV policy — extremely cheap)
DEFAULT_FEE_RATE_SAT_PER_BYTE = 1

# Approximate transaction sizes in bytes
EPOCH_OPEN_TX_SIZE = 310   # 1 input + 1 OP_RETURN (hashes) + 1 change
EPOCH_CLOSE_TX_SIZE = 330  # 1 input + 1 OP_RETURN (root + count) + 1 change

# BSV to USD conversion placeholder (user should provide real rate)
DEFAULT_BSV_USD = 50.0

# Satoshis per BSV
SATS_PER_BSV = 100_000_000


@dataclass
class CostEstimate:
    """Result of a cost estimation."""

    records: int
    epochs: int
    records_per_epoch: int
    bytes_per_epoch: int
    sats_per_epoch: int
    total_sats: int
    total_bsv: float
    total_usd: float
    bsv_usd_rate: float
    fee_rate_sat_byte: int
    breakdown: dict

    def __str__(self) -> str:
        lines = [
            f"ARIA Cost Estimate",
            f"──────────────────────────",
            f"Records:           {self.records:,}",
            f"Epochs:            {self.epochs:,}",
            f"Records/epoch:     {self.records_per_epoch:,}",
            f"",
            f"Per epoch:",
            f"  OPEN tx:         ~{self.breakdown['open_bytes']} bytes = {self.breakdown['open_sats']} sats",
            f"  CLOSE tx:        ~{self.breakdown['close_bytes']} bytes = {self.breakdown['close_sats']} sats",
            f"  Total/epoch:     {self.sats_per_epoch} sats",
            f"",
            f"Total cost:",
            f"  {self.total_sats:,} sats = {self.total_bsv:.8f} BSV",
            f"  ≈ ${self.total_usd:.4f} USD (@ ${self.bsv_usd_rate}/BSV)",
            f"  Fee rate: {self.fee_rate_sat_byte} sat/byte",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "records": self.records,
            "epochs": self.epochs,
            "records_per_epoch": self.records_per_epoch,
            "bytes_per_epoch": self.bytes_per_epoch,
            "sats_per_epoch": self.sats_per_epoch,
            "total_sats": self.total_sats,
            "total_bsv": self.total_bsv,
            "total_usd": self.total_usd,
            "bsv_usd_rate": self.bsv_usd_rate,
            "fee_rate_sat_byte": self.fee_rate_sat_byte,
        }


class CostEstimator:
    """Estimates the BSV transaction cost of ARIA epoch anchoring.

    Args:
        fee_rate:  Fee rate in satoshis/byte (default: 1 sat/byte on BSV).
        bsv_usd:  BSV/USD exchange rate for fiat estimate.
        network:   "mainnet" or "testnet" (testnet costs are zero).
    """

    def __init__(
        self,
        fee_rate: int = DEFAULT_FEE_RATE_SAT_PER_BYTE,
        bsv_usd: float = DEFAULT_BSV_USD,
        network: str = "mainnet",
    ) -> None:
        self._fee_rate = fee_rate
        self._bsv_usd = bsv_usd
        self._network = network

    def estimate(
        self,
        records: int,
        epochs: int | None = None,
        records_per_epoch: int = 500,
    ) -> CostEstimate:
        """Estimate cost for anchoring a given number of records.

        Args:
            records:           Total number of inference records.
            epochs:            Number of epochs (auto-calculated if None).
            records_per_epoch: Records per epoch (used if epochs not given).

        Returns:
            CostEstimate with breakdown.
        """
        if records <= 0:
            raise ValueError("records must be positive")
        if records_per_epoch <= 0:
            raise ValueError("records_per_epoch must be positive")

        if epochs is None:
            epochs = max(1, math.ceil(records / records_per_epoch))

        actual_rpe = max(1, records // epochs) if epochs > 0 else records

        if self._network == "testnet":
            return CostEstimate(
                records=records,
                epochs=epochs,
                records_per_epoch=actual_rpe,
                bytes_per_epoch=0,
                sats_per_epoch=0,
                total_sats=0,
                total_bsv=0.0,
                total_usd=0.0,
                bsv_usd_rate=0.0,
                fee_rate_sat_byte=0,
                breakdown={"open_bytes": 0, "open_sats": 0, "close_bytes": 0, "close_sats": 0, "note": "testnet — free"},
            )

        open_sats = EPOCH_OPEN_TX_SIZE * self._fee_rate
        close_sats = EPOCH_CLOSE_TX_SIZE * self._fee_rate
        sats_per_epoch = open_sats + close_sats
        bytes_per_epoch = EPOCH_OPEN_TX_SIZE + EPOCH_CLOSE_TX_SIZE

        total_sats = sats_per_epoch * epochs
        total_bsv = total_sats / SATS_PER_BSV
        total_usd = total_bsv * self._bsv_usd

        return CostEstimate(
            records=records,
            epochs=epochs,
            records_per_epoch=actual_rpe,
            bytes_per_epoch=bytes_per_epoch,
            sats_per_epoch=sats_per_epoch,
            total_sats=total_sats,
            total_bsv=total_bsv,
            total_usd=total_usd,
            bsv_usd_rate=self._bsv_usd,
            fee_rate_sat_byte=self._fee_rate,
            breakdown={
                "open_bytes": EPOCH_OPEN_TX_SIZE,
                "open_sats": open_sats,
                "close_bytes": EPOCH_CLOSE_TX_SIZE,
                "close_sats": close_sats,
            },
        )

    def monthly_estimate(
        self,
        inferences_per_day: int,
        epochs_per_day: int = 24,
    ) -> CostEstimate:
        """Convenience for estimating monthly cost.

        Args:
            inferences_per_day: Average daily inference count.
            epochs_per_day:     How many epochs to open per day.

        Returns:
            CostEstimate for a 30-day month.
        """
        total_records = inferences_per_day * 30
        total_epochs = epochs_per_day * 30
        rpe = max(1, total_records // total_epochs) if total_epochs > 0 else total_records
        return self.estimate(records=total_records, epochs=total_epochs, records_per_epoch=rpe)
