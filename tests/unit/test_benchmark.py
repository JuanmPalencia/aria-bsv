"""Tests for aria.benchmark — BenchmarkResult, BenchmarkAnchor, BenchmarkRegistry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aria.benchmark import (
    BenchmarkAnchor,
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSuite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MODEL_HASH = "sha256:" + "ab" * 32
_FAKE_TXID = "cd" * 32


def _result(
    score: float = 0.87,
    suite: BenchmarkSuite = BenchmarkSuite.MMLU,
    model_hash: str = _FAKE_MODEL_HASH,
    model_id: str = "test-model",
    num_samples: int = 1000,
    subset: str = "",
    metadata: dict | None = None,
    evaluated_at: datetime | None = None,
) -> BenchmarkResult:
    kwargs: dict = dict(
        model_hash=model_hash,
        model_id=model_id,
        suite=suite,
        score=score,
        num_samples=num_samples,
        subset=subset,
        metadata=metadata or {},
    )
    if evaluated_at is not None:
        kwargs["evaluated_at"] = evaluated_at
    return BenchmarkResult(**kwargs)


def _mock_broadcaster(txid: str = _FAKE_TXID) -> MagicMock:
    """Broadcaster whose broadcast() returns an object with a .txid attribute."""
    status = MagicMock()
    status.txid = txid
    b = MagicMock()
    b.broadcast = MagicMock(return_value=status)
    return b


def _registry(broadcaster=None) -> BenchmarkRegistry:
    return BenchmarkRegistry(broadcaster=broadcaster)


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    def test_mmlu_value(self):
        assert BenchmarkSuite.MMLU.value == "mmlu"

    def test_humaneval_value(self):
        assert BenchmarkSuite.HUMANEVAL.value == "humaneval"

    def test_bigbench_value(self):
        assert BenchmarkSuite.BIGBENCH.value == "bigbench"

    def test_all_expected_members(self):
        names = {m.name for m in BenchmarkSuite}
        expected = {
            "MMLU", "HUMANEVAL", "BIGBENCH", "HELLASWAG",
            "ARC", "TRUTHFULQA", "GSMATH", "MBPP", "CUSTOM",
        }
        assert expected.issubset(names)

    def test_custom_value(self):
        assert BenchmarkSuite.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# BenchmarkResult — basic construction and validation
# ---------------------------------------------------------------------------


class TestBenchmarkResultValidation:
    def test_valid_result_creates_ok(self):
        r = _result(score=0.5)
        assert r.score == 0.5

    def test_score_zero_allowed(self):
        r = _result(score=0.0)
        assert r.score == 0.0

    def test_score_one_allowed(self):
        r = _result(score=1.0)
        assert r.score == 1.0

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="score must be in"):
            _result(score=1.01)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="score must be in"):
            _result(score=-0.01)

    def test_num_samples_zero_raises(self):
        with pytest.raises(ValueError, match="num_samples must be"):
            _result(num_samples=0)

    def test_num_samples_negative_raises(self):
        with pytest.raises(ValueError, match="num_samples must be"):
            _result(num_samples=-5)

    def test_num_samples_one_allowed(self):
        r = _result(num_samples=1)
        assert r.num_samples == 1


# ---------------------------------------------------------------------------
# BenchmarkResult.result_hash
# ---------------------------------------------------------------------------


class TestBenchmarkResultHash:
    def test_returns_sha256_prefixed(self):
        r = _result()
        assert r.result_hash().startswith("sha256:")

    def test_hex_part_is_64_chars(self):
        r = _result()
        assert len(r.result_hash().split("sha256:")[1]) == 64

    def test_deterministic(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r1 = _result(evaluated_at=dt)
        r2 = _result(evaluated_at=dt)
        assert r1.result_hash() == r2.result_hash()

    def test_idempotent_same_instance(self):
        r = _result()
        assert r.result_hash() == r.result_hash()

    def test_different_scores_different_hash(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r1 = _result(score=0.80, evaluated_at=dt)
        r2 = _result(score=0.90, evaluated_at=dt)
        assert r1.result_hash() != r2.result_hash()

    def test_different_subsets_different_hash(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r1 = _result(subset="", evaluated_at=dt)
        r2 = _result(subset="STEM", evaluated_at=dt)
        assert r1.result_hash() != r2.result_hash()

    def test_metadata_included_in_hash(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r1 = _result(metadata={}, evaluated_at=dt)
        r2 = _result(metadata={"extra": "info"}, evaluated_at=dt)
        assert r1.result_hash() != r2.result_hash()

    def test_different_model_hash_different_result_hash(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r1 = _result(model_hash="sha256:" + "aa" * 32, evaluated_at=dt)
        r2 = _result(model_hash="sha256:" + "bb" * 32, evaluated_at=dt)
        assert r1.result_hash() != r2.result_hash()


# ---------------------------------------------------------------------------
# BenchmarkResult.to_dict
# ---------------------------------------------------------------------------


class TestBenchmarkResultToDict:
    def test_to_dict_is_json_serialisable(self):
        r = _result()
        d = r.to_dict()
        # Must not raise
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_to_dict_contains_expected_keys(self):
        r = _result()
        d = r.to_dict()
        for key in ("model_hash", "model_id", "suite", "score", "num_samples",
                    "subset", "metadata", "evaluated_at"):
            assert key in d

    def test_to_dict_suite_is_string(self):
        r = _result(suite=BenchmarkSuite.HUMANEVAL)
        assert r.to_dict()["suite"] == "humaneval"

    def test_to_dict_evaluated_at_is_string(self):
        r = _result()
        assert isinstance(r.to_dict()["evaluated_at"], str)

    def test_to_dict_metadata_preserved(self):
        r = _result(metadata={"foo": "bar", "n": 42})
        assert r.to_dict()["metadata"] == {"foo": "bar", "n": 42}


# ---------------------------------------------------------------------------
# BenchmarkAnchor
# ---------------------------------------------------------------------------


class TestBenchmarkAnchor:
    def test_is_on_chain_true_when_txid_set(self):
        r = _result()
        anchor = BenchmarkAnchor(
            result=r,
            result_hash=r.result_hash(),
            txid=_FAKE_TXID,
            anchored_at=datetime.now(timezone.utc),
        )
        assert anchor.is_on_chain() is True

    def test_is_on_chain_false_when_txid_none(self):
        r = _result()
        anchor = BenchmarkAnchor(
            result=r,
            result_hash=r.result_hash(),
            txid=None,
            anchored_at=datetime.now(timezone.utc),
        )
        assert anchor.is_on_chain() is False

    def test_is_on_chain_false_when_txid_empty_string(self):
        r = _result()
        anchor = BenchmarkAnchor(
            result=r,
            result_hash=r.result_hash(),
            txid="",
            anchored_at=datetime.now(timezone.utc),
        )
        assert anchor.is_on_chain() is False


# ---------------------------------------------------------------------------
# BenchmarkRegistry.anchor
# ---------------------------------------------------------------------------


class TestBenchmarkRegistryAnchor:
    def test_anchor_returns_benchmark_anchor(self):
        reg = _registry()
        anchor = reg.anchor(_result())
        assert isinstance(anchor, BenchmarkAnchor)

    def test_anchor_result_hash_set(self):
        reg = _registry()
        r = _result()
        anchor = reg.anchor(r)
        assert anchor.result_hash == r.result_hash()

    def test_anchor_without_broadcaster_txid_none(self):
        reg = _registry(broadcaster=None)
        anchor = reg.anchor(_result())
        assert anchor.txid is None
        assert anchor.is_on_chain() is False

    def test_anchor_with_broadcaster_calls_broadcast(self):
        b = _mock_broadcaster()
        reg = _registry(broadcaster=b)
        reg.anchor(_result())
        b.broadcast.assert_called_once()

    def test_anchor_with_broadcaster_sets_txid(self):
        b = _mock_broadcaster(txid=_FAKE_TXID)
        reg = _registry(broadcaster=b)
        anchor = reg.anchor(_result())
        assert anchor.txid == _FAKE_TXID
        assert anchor.is_on_chain() is True

    def test_anchor_broadcaster_string_return(self):
        """Broadcaster that returns a raw string txid."""
        b = MagicMock()
        b.broadcast = MagicMock(return_value=_FAKE_TXID)
        reg = _registry(broadcaster=b)
        anchor = reg.anchor(_result())
        assert anchor.txid == _FAKE_TXID

    def test_anchor_broadcaster_none_return(self):
        """Broadcaster that returns None leaves txid as None."""
        b = MagicMock()
        b.broadcast = MagicMock(return_value=None)
        reg = _registry(broadcaster=b)
        anchor = reg.anchor(_result())
        assert anchor.txid is None

    def test_anchor_stored_in_registry(self):
        reg = _registry()
        r = _result()
        anchor = reg.anchor(r)
        found = reg.get_anchors(r.model_hash)
        assert anchor in found


# ---------------------------------------------------------------------------
# BenchmarkRegistry.verify
# ---------------------------------------------------------------------------


class TestBenchmarkRegistryVerify:
    def test_verify_valid_anchor_returns_true(self):
        reg = _registry()
        anchor = reg.anchor(_result())
        assert reg.verify(anchor) is True

    def test_verify_tampered_score_returns_false(self):
        reg = _registry()
        r = _result(score=0.80)
        anchor = reg.anchor(r)
        # Tamper: mutate the result's score after anchoring
        object.__setattr__(anchor.result, "score", 0.99) if hasattr(
            anchor.result, "__dataclass_fields__"
        ) else None
        anchor.result.__dict__["score"] = 0.99
        assert reg.verify(anchor) is False

    def test_verify_mismatched_hash_returns_false(self):
        reg = _registry()
        r = _result()
        anchor = BenchmarkAnchor(
            result=r,
            result_hash="sha256:" + "00" * 32,  # wrong hash
            txid=None,
            anchored_at=datetime.now(timezone.utc),
        )
        assert reg.verify(anchor) is False


# ---------------------------------------------------------------------------
# BenchmarkRegistry.get_anchors
# ---------------------------------------------------------------------------


class TestBenchmarkRegistryGetAnchors:
    def test_returns_empty_for_unknown_model(self):
        reg = _registry()
        assert reg.get_anchors("sha256:" + "ff" * 32) == []

    def test_returns_correct_anchors_for_model(self):
        reg = _registry()
        mh = "sha256:" + "aa" * 32
        r1 = _result(model_hash=mh, score=0.70)
        r2 = _result(model_hash=mh, score=0.85)
        r_other = _result(model_hash="sha256:" + "bb" * 32, score=0.90)
        a1 = reg.anchor(r1)
        a2 = reg.anchor(r2)
        reg.anchor(r_other)
        found = reg.get_anchors(mh)
        assert a1 in found
        assert a2 in found
        assert len(found) == 2

    def test_anchors_not_duplicated(self):
        reg = _registry()
        r = _result()
        reg.anchor(r)
        reg.anchor(r)  # same result_hash — should not duplicate
        found = reg.get_anchors(r.model_hash)
        assert len(found) == 1


# ---------------------------------------------------------------------------
# BenchmarkRegistry.get_best
# ---------------------------------------------------------------------------


class TestBenchmarkRegistryGetBest:
    def test_returns_none_when_no_anchors(self):
        reg = _registry()
        assert reg.get_best(BenchmarkSuite.MMLU) is None

    def test_returns_highest_score(self):
        reg = _registry()
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r_low = _result(score=0.70, suite=BenchmarkSuite.MMLU,
                        model_id="model-a", evaluated_at=dt)
        r_high = _result(score=0.92, suite=BenchmarkSuite.MMLU,
                         model_id="model-b",
                         evaluated_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
        reg.anchor(r_low)
        best = reg.anchor(r_high)
        assert reg.get_best(BenchmarkSuite.MMLU) is best

    def test_filters_by_suite(self):
        reg = _registry()
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r_mmlu = _result(score=0.90, suite=BenchmarkSuite.MMLU, evaluated_at=dt)
        r_arc = _result(score=0.95, suite=BenchmarkSuite.ARC,
                        evaluated_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
        best_mmlu = reg.anchor(r_mmlu)
        reg.anchor(r_arc)
        assert reg.get_best(BenchmarkSuite.MMLU) is best_mmlu

    def test_filters_by_model_hash(self):
        reg = _registry()
        mh_a = "sha256:" + "aa" * 32
        mh_b = "sha256:" + "bb" * 32
        dt_a = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dt_b = datetime(2025, 1, 2, tzinfo=timezone.utc)
        r_a = _result(score=0.80, model_hash=mh_a,
                      suite=BenchmarkSuite.MMLU, evaluated_at=dt_a)
        r_b = _result(score=0.95, model_hash=mh_b,
                      suite=BenchmarkSuite.MMLU, evaluated_at=dt_b)
        best_a = reg.anchor(r_a)
        reg.anchor(r_b)
        assert reg.get_best(BenchmarkSuite.MMLU, model_hash=mh_a) is best_a

    def test_returns_none_for_unknown_suite(self):
        reg = _registry()
        reg.anchor(_result(suite=BenchmarkSuite.MMLU))
        assert reg.get_best(BenchmarkSuite.ARC) is None


# ---------------------------------------------------------------------------
# BenchmarkRegistry.compare
# ---------------------------------------------------------------------------


class TestBenchmarkRegistryCompare:
    def _make_anchors(
        self,
        score_a: float,
        score_b: float,
        suite_a: BenchmarkSuite = BenchmarkSuite.MMLU,
        suite_b: BenchmarkSuite = BenchmarkSuite.MMLU,
        model_hash_a: str = _FAKE_MODEL_HASH,
        model_hash_b: str = _FAKE_MODEL_HASH,
    ) -> tuple[BenchmarkAnchor, BenchmarkAnchor]:
        reg = _registry()
        dt_a = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dt_b = datetime(2025, 1, 2, tzinfo=timezone.utc)
        r_a = _result(score=score_a, suite=suite_a,
                      model_hash=model_hash_a, evaluated_at=dt_a)
        r_b = _result(score=score_b, suite=suite_b,
                      model_hash=model_hash_b, evaluated_at=dt_b)
        return reg.anchor(r_a), reg.anchor(r_b)

    def test_score_delta_positive(self):
        a, b = self._make_anchors(0.90, 0.80)
        cmp = _registry().compare(a, b)
        assert abs(cmp["score_delta"] - 0.10) < 1e-9

    def test_score_delta_negative(self):
        a, b = self._make_anchors(0.70, 0.90)
        cmp = _registry().compare(a, b)
        assert cmp["score_delta"] < 0

    def test_score_delta_zero(self):
        a, b = self._make_anchors(0.85, 0.85)
        cmp = _registry().compare(a, b)
        assert cmp["score_delta"] == 0.0

    def test_same_model_true(self):
        a, b = self._make_anchors(0.80, 0.90,
                                  model_hash_a=_FAKE_MODEL_HASH,
                                  model_hash_b=_FAKE_MODEL_HASH)
        cmp = _registry().compare(a, b)
        assert cmp["same_model"] is True

    def test_same_model_false(self):
        a, b = self._make_anchors(
            0.80, 0.90,
            model_hash_a="sha256:" + "aa" * 32,
            model_hash_b="sha256:" + "bb" * 32,
        )
        cmp = _registry().compare(a, b)
        assert cmp["same_model"] is False

    def test_suite_match_true(self):
        a, b = self._make_anchors(0.80, 0.90,
                                  suite_a=BenchmarkSuite.MMLU,
                                  suite_b=BenchmarkSuite.MMLU)
        cmp = _registry().compare(a, b)
        assert cmp["suite_match"] is True

    def test_suite_match_false(self):
        a, b = self._make_anchors(0.80, 0.90,
                                  suite_a=BenchmarkSuite.MMLU,
                                  suite_b=BenchmarkSuite.ARC)
        cmp = _registry().compare(a, b)
        assert cmp["suite_match"] is False

    def test_compare_includes_score_a_and_b(self):
        a, b = self._make_anchors(0.75, 0.88)
        cmp = _registry().compare(a, b)
        assert cmp["score_a"] == pytest.approx(0.75)
        assert cmp["score_b"] == pytest.approx(0.88)

    def test_compare_includes_suite_values(self):
        a, b = self._make_anchors(0.80, 0.90,
                                  suite_a=BenchmarkSuite.MMLU,
                                  suite_b=BenchmarkSuite.HUMANEVAL)
        cmp = _registry().compare(a, b)
        assert cmp["suite_a"] == "mmlu"
        assert cmp["suite_b"] == "humaneval"
