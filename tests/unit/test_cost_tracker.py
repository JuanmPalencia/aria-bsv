"""Tests for aria.cost_tracker."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aria.cost_tracker import (
    DEFAULT_PRICES,
    CostReport,
    CostTracker,
    InferenceCost,
    SystemCostReport,
    _coerce_int,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(model_id: str, in_tokens: int = 0, out_tokens: int = 0, metadata: dict | None = None):
    r = MagicMock()
    r.record_id = f"rec-{model_id}"
    r.model_id = model_id
    r.metadata = metadata or {}
    return r


def _record_with_usage(model_id: str, in_tok: int, out_tok: int, provider: str = "openai"):
    if provider == "openai":
        usage = {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok}
    else:
        usage = {"input_tokens": in_tok, "output_tokens": out_tok}
    return _record(model_id, metadata={"usage": usage})


def _storage(epoch_records: dict[str, list]):
    storage = MagicMock()
    storage.list_records_by_epoch.side_effect = lambda eid: epoch_records.get(eid, [])
    return storage


# ---------------------------------------------------------------------------
# _coerce_int
# ---------------------------------------------------------------------------

class TestCoerceInt:
    def test_int(self):
        assert _coerce_int(42) == 42

    def test_float(self):
        assert _coerce_int(3.7) == 3

    def test_string(self):
        assert _coerce_int("100") == 100

    def test_none(self):
        assert _coerce_int(None) == 0

    def test_zero(self):
        assert _coerce_int(0) == 0

    def test_negative(self):
        assert _coerce_int(-5) == 0

    def test_invalid(self):
        assert _coerce_int("abc") == 0


# ---------------------------------------------------------------------------
# CostTracker.price_for
# ---------------------------------------------------------------------------

class TestPriceFor:
    def _tracker(self, extra: dict | None = None):
        return CostTracker(_storage({}), price_table=extra)

    def test_exact_match(self):
        t = self._tracker()
        prices = t.price_for("gpt-4o")
        assert prices["input"] == DEFAULT_PRICES["gpt-4o"]["input"]

    def test_prefix_match(self):
        t = self._tracker()
        # "gpt-4o-2024-08-06" should match "gpt-4o"
        prices = t.price_for("gpt-4o-2024-08-06")
        assert prices["input"] == DEFAULT_PRICES["gpt-4o"]["input"]

    def test_default_fallback(self):
        t = self._tracker()
        prices = t.price_for("some-unknown-model")
        assert prices["input"] == DEFAULT_PRICES["__default__"]["input"]

    def test_custom_price_table(self):
        t = self._tracker(extra={"my-model": {"input": 99.0, "output": 199.0}})
        prices = t.price_for("my-model")
        assert prices["input"] == 99.0

    def test_custom_overrides_default(self):
        t = self._tracker(extra={"gpt-4o": {"input": 0.01, "output": 0.01}})
        prices = t.price_for("gpt-4o")
        assert prices["input"] == 0.01


# ---------------------------------------------------------------------------
# CostTracker.epoch_cost
# ---------------------------------------------------------------------------

class TestEpochCost:
    def test_empty_epoch(self):
        storage = _storage({"ep-1": []})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-1")
        assert report.total_usd == 0.0
        assert len(report.records) == 0

    def test_single_record_openai(self):
        r = _record_with_usage("gpt-4o", in_tok=1000, out_tok=500)
        storage = _storage({"ep-1": [r]})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-1")
        # gpt-4o: $5/1M input, $15/1M output
        expected_in  = (1000 / 1_000_000) * 5.0
        expected_out = (500  / 1_000_000) * 15.0
        assert report.total_usd == pytest.approx(expected_in + expected_out)

    def test_single_record_anthropic(self):
        r = _record_with_usage("claude-opus-4-6", in_tok=2000, out_tok=1000, provider="anthropic")
        storage = _storage({"ep-1": [r]})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-1")
        # claude-opus-4-6: $15/1M input, $75/1M output
        expected = (2000 / 1_000_000) * 15.0 + (1000 / 1_000_000) * 75.0
        assert report.total_usd == pytest.approx(expected)

    def test_default_tokens_when_no_metadata(self):
        r = _record("gpt-4o")
        storage = _storage({"ep-1": [r]})
        t = CostTracker(storage, default_input_tokens=500, default_output_tokens=150)
        report = t.epoch_cost("ep-1")
        expected = (500 / 1_000_000) * 5.0 + (150 / 1_000_000) * 15.0
        assert report.total_usd == pytest.approx(expected)

    def test_multiple_records(self):
        records = [
            _record_with_usage("gpt-4o-mini", 100, 50),
            _record_with_usage("gpt-4o-mini", 200, 100),
        ]
        storage = _storage({"ep-1": records})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-1")
        assert len(report.records) == 2
        assert report.total_usd > 0

    def test_epoch_id_preserved(self):
        storage = _storage({"my-epoch": []})
        t = CostTracker(storage)
        report = t.epoch_cost("my-epoch")
        assert report.epoch_id == "my-epoch"

    def test_cost_by_model(self):
        records = [
            _record_with_usage("gpt-4o", 1000, 500),
            _record_with_usage("gpt-4o-mini", 1000, 500),
        ]
        storage = _storage({"ep-1": records})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-1")
        cbm = report.cost_by_model
        assert "gpt-4o" in cbm
        assert "gpt-4o-mini" in cbm
        assert cbm["gpt-4o"] > cbm["gpt-4o-mini"]  # gpt-4o is more expensive

    def test_str_representation(self):
        r = _record_with_usage("gpt-4o", 1000, 500)
        storage = _storage({"ep-x": [r]})
        t = CostTracker(storage)
        report = t.epoch_cost("ep-x")
        s = str(report)
        assert "ep-x" in s
        assert "$" in s


# ---------------------------------------------------------------------------
# CostTracker.within_budget
# ---------------------------------------------------------------------------

class TestWithinBudget:
    def test_within_budget(self):
        r = _record_with_usage("gpt-4o-mini", 100, 50)
        storage = _storage({"ep-1": [r]})
        t = CostTracker(storage)
        assert t.within_budget("ep-1", budget_usd=10.0) is True

    def test_over_budget(self):
        r = _record_with_usage("claude-opus-4-6", 1_000_000, 500_000)
        storage = _storage({"ep-1": [r]})
        t = CostTracker(storage)
        # $15 + $37.5 = $52.5 >> $1
        assert t.within_budget("ep-1", budget_usd=1.0) is False


# ---------------------------------------------------------------------------
# CostTracker.system_cost
# ---------------------------------------------------------------------------

class TestSystemCost:
    def test_aggregates_epochs(self):
        r1 = _record_with_usage("gpt-4o", 1000, 500)
        r2 = _record_with_usage("gpt-4o", 2000, 1000)
        storage = _storage({"ep-1": [r1], "ep-2": [r2]})
        t = CostTracker(storage)
        sys_report = t.system_cost(["ep-1", "ep-2"], system_id="my-system")
        assert sys_report.system_id == "my-system"
        assert len(sys_report.epoch_costs) == 2
        total = sys_report.total_usd
        assert total > 0

    def test_cost_by_model_across_epochs(self):
        records1 = [_record_with_usage("gpt-4o", 1000, 500)]
        records2 = [_record_with_usage("gpt-4o-mini", 1000, 500)]
        storage = _storage({"ep-1": records1, "ep-2": records2})
        t = CostTracker(storage)
        sys_report = t.system_cost(["ep-1", "ep-2"])
        cbm = sys_report.cost_by_model
        assert "gpt-4o" in cbm
        assert "gpt-4o-mini" in cbm


# ---------------------------------------------------------------------------
# InferenceCost
# ---------------------------------------------------------------------------

class TestInferenceCost:
    def test_total_cost(self):
        ic = InferenceCost(
            record_id="r1",
            model_id="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.005,
            output_cost=0.0075,
        )
        assert ic.total_cost == pytest.approx(0.0125)
