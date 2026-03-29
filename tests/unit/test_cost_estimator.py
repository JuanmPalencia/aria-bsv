"""Tests for aria.cost_estimator — BSV transaction cost estimation."""

from __future__ import annotations

import pytest

from aria.cost_estimator import (
    CostEstimate,
    CostEstimator,
    DEFAULT_BSV_USD,
    DEFAULT_FEE_RATE_SAT_PER_BYTE,
    EPOCH_CLOSE_TX_SIZE,
    EPOCH_OPEN_TX_SIZE,
    SATS_PER_BSV,
)


class TestCostEstimator:
    """Tests for the CostEstimator class."""

    def test_basic_estimate(self):
        est = CostEstimator()
        result = est.estimate(records=1000)
        assert isinstance(result, CostEstimate)
        assert result.records == 1000
        assert result.epochs == 2  # 1000 / 500 = 2
        assert result.total_sats > 0
        assert result.total_bsv > 0
        assert result.total_usd > 0

    def test_single_epoch(self):
        est = CostEstimator()
        result = est.estimate(records=100, epochs=1)
        assert result.epochs == 1
        expected_sats = (EPOCH_OPEN_TX_SIZE + EPOCH_CLOSE_TX_SIZE) * DEFAULT_FEE_RATE_SAT_PER_BYTE
        assert result.total_sats == expected_sats
        assert result.sats_per_epoch == expected_sats

    def test_multiple_epochs(self):
        est = CostEstimator()
        result = est.estimate(records=10000, epochs=10)
        assert result.epochs == 10
        assert result.total_sats == result.sats_per_epoch * 10

    def test_auto_epoch_calculation(self):
        est = CostEstimator()
        result = est.estimate(records=2500, records_per_epoch=500)
        assert result.epochs == 5

    def test_testnet_is_free(self):
        est = CostEstimator(network="testnet")
        result = est.estimate(records=1000000)
        assert result.total_sats == 0
        assert result.total_bsv == 0.0
        assert result.total_usd == 0.0
        assert result.breakdown.get("note") == "testnet — free"

    def test_custom_fee_rate(self):
        est = CostEstimator(fee_rate=2)
        result = est.estimate(records=100, epochs=1)
        expected = (EPOCH_OPEN_TX_SIZE + EPOCH_CLOSE_TX_SIZE) * 2
        assert result.total_sats == expected

    def test_custom_bsv_usd(self):
        est = CostEstimator(bsv_usd=100.0)
        result = est.estimate(records=100, epochs=1)
        assert result.bsv_usd_rate == 100.0
        assert result.total_usd > 0

    def test_bsv_conversion(self):
        est = CostEstimator(bsv_usd=100.0)
        result = est.estimate(records=100, epochs=1)
        assert abs(result.total_bsv - result.total_sats / SATS_PER_BSV) < 1e-12
        assert abs(result.total_usd - result.total_bsv * 100.0) < 1e-8

    def test_invalid_records_raises(self):
        est = CostEstimator()
        with pytest.raises(ValueError, match="records must be positive"):
            est.estimate(records=0)

    def test_invalid_records_per_epoch_raises(self):
        est = CostEstimator()
        with pytest.raises(ValueError, match="records_per_epoch must be positive"):
            est.estimate(records=100, records_per_epoch=0)

    def test_breakdown(self):
        est = CostEstimator()
        result = est.estimate(records=100, epochs=1)
        assert "open_bytes" in result.breakdown
        assert "close_bytes" in result.breakdown
        assert "open_sats" in result.breakdown
        assert "close_sats" in result.breakdown
        assert result.breakdown["open_bytes"] == EPOCH_OPEN_TX_SIZE
        assert result.breakdown["close_bytes"] == EPOCH_CLOSE_TX_SIZE


class TestCostEstimate:
    """Tests for the CostEstimate dataclass."""

    def test_to_dict(self):
        result = CostEstimate(
            records=100,
            epochs=1,
            records_per_epoch=100,
            bytes_per_epoch=640,
            sats_per_epoch=640,
            total_sats=640,
            total_bsv=0.0000064,
            total_usd=0.00032,
            bsv_usd_rate=50.0,
            fee_rate_sat_byte=1,
            breakdown={"open_bytes": 310, "close_bytes": 330, "open_sats": 310, "close_sats": 330},
        )
        d = result.to_dict()
        assert d["records"] == 100
        assert d["total_sats"] == 640
        assert "breakdown" not in d  # breakdown not in to_dict for simplicity

    def test_str_output(self):
        est = CostEstimator()
        result = est.estimate(records=1000)
        s = str(result)
        assert "ARIA Cost Estimate" in s
        assert "Records:" in s
        assert "Total cost:" in s
        assert "sats" in s


class TestMonthlyEstimate:
    """Tests for monthly_estimate."""

    def test_monthly_estimate(self):
        est = CostEstimator()
        result = est.monthly_estimate(inferences_per_day=10000, epochs_per_day=24)
        assert result.records == 300000
        assert result.epochs == 720
        assert result.total_sats > 0

    def test_monthly_estimate_testnet(self):
        est = CostEstimator(network="testnet")
        result = est.monthly_estimate(inferences_per_day=1000)
        assert result.total_sats == 0
