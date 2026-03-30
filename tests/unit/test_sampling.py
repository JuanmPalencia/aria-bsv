"""
tests/unit/test_sampling.py

Unit tests for aria/sampling.py — cryptographically verifiable audit sampling.

Coverage targets:
- SamplingConfig validation and properties
- AuditSampler determinism, statistics, verification, and reset
- SampleDecision fields
- VerifiableSamplingProof generation and tamper detection
"""

from __future__ import annotations

import hashlib
from datetime import timezone

import pytest

from aria.sampling import (
    AuditSampler,
    SampleDecision,
    SamplingConfig,
    SamplingMethod,
    VerifiableSamplingProof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ids(n: int, prefix: str = "inf") -> list[str]:
    """Return a list of n inference IDs."""
    return [f"{prefix}_{i:06d}" for i in range(n)]


def _run_sampler(config: SamplingConfig, n: int) -> tuple[AuditSampler, list[SampleDecision]]:
    """Run *n* should_record calls and return (sampler, decisions)."""
    sampler = AuditSampler(config)
    decisions = [sampler.should_record(f"inf_{i}") for i in range(n)]
    return sampler, decisions


# ---------------------------------------------------------------------------
# SamplingConfig — validation
# ---------------------------------------------------------------------------


class TestSamplingConfigValidation:
    def test_rate_zero_raises(self):
        with pytest.raises(ValueError, match="rate"):
            SamplingConfig(rate=0.0)

    def test_rate_negative_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(rate=-0.1)

    def test_rate_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            SamplingConfig(rate=1.01)

    def test_rate_exactly_one_is_valid(self):
        cfg = SamplingConfig(rate=1.0)
        assert cfg.rate == 1.0

    def test_rate_small_positive_is_valid(self):
        cfg = SamplingConfig(rate=0.001)
        assert cfg.rate == 0.001


# ---------------------------------------------------------------------------
# SamplingConfig — is_verifiable
# ---------------------------------------------------------------------------


class TestSamplingConfigIsVerifiable:
    def test_is_verifiable_true_when_seed_txid_set(self):
        cfg = SamplingConfig(rate=0.1, seed_txid="abc123")
        assert cfg.is_verifiable is True

    def test_is_verifiable_false_when_no_seed(self):
        cfg = SamplingConfig(rate=0.1)
        assert cfg.is_verifiable is False

    def test_is_verifiable_false_when_seed_txid_none_explicitly(self):
        cfg = SamplingConfig(rate=0.1, seed_txid=None)
        assert cfg.is_verifiable is False


# ---------------------------------------------------------------------------
# SamplingConfig — seed_material
# ---------------------------------------------------------------------------


class TestSamplingConfigSeedMaterial:
    def test_seed_material_is_bytes(self):
        cfg = SamplingConfig(rate=0.1, seed_txid="tx1")
        assert isinstance(cfg.seed_material, bytes)

    def test_seed_material_differs_with_different_seed_txid(self):
        cfg_a = SamplingConfig(rate=0.1, seed_txid="abc")
        cfg_b = SamplingConfig(rate=0.1, seed_txid="xyz")
        assert cfg_a.seed_material != cfg_b.seed_material

    def test_seed_material_same_for_same_config(self):
        cfg_a = SamplingConfig(rate=0.2, seed_txid="same_tx", seed_block=100)
        cfg_b = SamplingConfig(rate=0.2, seed_txid="same_tx", seed_block=100)
        assert cfg_a.seed_material == cfg_b.seed_material

    def test_seed_material_differs_with_different_block(self):
        cfg_a = SamplingConfig(rate=0.1, seed_txid="tx", seed_block=10)
        cfg_b = SamplingConfig(rate=0.1, seed_txid="tx", seed_block=99)
        assert cfg_a.seed_material != cfg_b.seed_material

    def test_seed_material_differs_when_no_txid_vs_txid(self):
        cfg_a = SamplingConfig(rate=0.1)
        cfg_b = SamplingConfig(rate=0.1, seed_txid="tx")
        assert cfg_a.seed_material != cfg_b.seed_material


# ---------------------------------------------------------------------------
# AuditSampler — Bernoulli rate accuracy
# ---------------------------------------------------------------------------


class TestAuditSamplerBernoulli:
    def test_bernoulli_rate_01_approx_10_percent(self):
        cfg = SamplingConfig(rate=0.1, seed_txid="bernoulli_seed_A")
        sampler, decisions = _run_sampler(cfg, 10_000)
        selected = sum(1 for d in decisions if d.selected)
        rate = selected / 10_000
        assert abs(rate - 0.1) < 0.03, f"Bernoulli rate {rate:.4f} too far from 0.10"

    def test_bernoulli_rate_05_approx_50_percent(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="bernoulli_seed_B")
        sampler, decisions = _run_sampler(cfg, 1_000)
        selected = sum(1 for d in decisions if d.selected)
        rate = selected / 1_000
        assert abs(rate - 0.5) < 0.05, f"Bernoulli rate {rate:.4f} too far from 0.50"

    def test_rate_1_all_selected(self):
        cfg = SamplingConfig(rate=1.0, seed_txid="full")
        sampler, decisions = _run_sampler(cfg, 100)
        assert all(d.selected for d in decisions)

    def test_bernoulli_rate_001_low_volume(self):
        """rate=0.01 over 10000 should select between 50 and 150."""
        cfg = SamplingConfig(rate=0.01, seed_txid="low_rate_seed")
        sampler, decisions = _run_sampler(cfg, 10_000)
        selected = sum(1 for d in decisions if d.selected)
        assert 50 <= selected <= 150, f"Got {selected} selections at rate=0.01 over 10000"


# ---------------------------------------------------------------------------
# AuditSampler — Systematic
# ---------------------------------------------------------------------------


class TestAuditSamplerSystematic:
    def test_systematic_every_10th(self):
        cfg = SamplingConfig(rate=0.1, method=SamplingMethod.SYSTEMATIC, seed_txid="sys")
        sampler, decisions = _run_sampler(cfg, 100)
        selected = sum(1 for d in decisions if d.selected)
        # rate=0.1 → interval=10, so exactly 10 from 100
        assert selected == 10

    def test_systematic_every_4th(self):
        cfg = SamplingConfig(rate=0.25, method=SamplingMethod.SYSTEMATIC, seed_txid="sys4")
        sampler, decisions = _run_sampler(cfg, 100)
        # positions 0, 4, 8, … → exactly 25 from 100
        selected_positions = [d.position for d in decisions if d.selected]
        assert len(selected_positions) == 25
        for pos in selected_positions:
            assert pos % 4 == 0

    def test_systematic_rate_1_all_selected(self):
        cfg = SamplingConfig(rate=1.0, method=SamplingMethod.SYSTEMATIC)
        sampler, decisions = _run_sampler(cfg, 50)
        assert all(d.selected for d in decisions)


# ---------------------------------------------------------------------------
# AuditSampler — determinism
# ---------------------------------------------------------------------------


class TestAuditSamplerDeterminism:
    def test_same_seed_same_decisions(self):
        cfg_a = SamplingConfig(rate=0.3, seed_txid="deterministic_tx_1")
        cfg_b = SamplingConfig(rate=0.3, seed_txid="deterministic_tx_1")
        ids = _make_ids(200)
        _, decisions_a = _run_sampler(cfg_a, 200)
        sampler_b = AuditSampler(cfg_b)
        decisions_b = [sampler_b.should_record(i) for i in ids]
        selected_a = [d.selected for d in decisions_a]
        selected_b = [d.selected for d in decisions_b]
        assert selected_a == selected_b

    def test_different_seeds_different_patterns(self):
        cfg_a = SamplingConfig(rate=0.5, seed_txid="seed_alpha")
        cfg_b = SamplingConfig(rate=0.5, seed_txid="seed_beta")
        _, decisions_a = _run_sampler(cfg_a, 100)
        _, decisions_b = _run_sampler(cfg_b, 100)
        selections_a = [d.selected for d in decisions_a]
        selections_b = [d.selected for d in decisions_b]
        # Different seeds must produce different patterns
        assert selections_a != selections_b

    def test_reset_replays_same_decisions(self):
        cfg = SamplingConfig(rate=0.4, seed_txid="replay_seed")
        sampler = AuditSampler(cfg)
        first_pass = [sampler.should_record(f"i_{k}") for k in range(50)]
        sampler.reset()
        second_pass = [sampler.should_record(f"i_{k}") for k in range(50)]
        assert [d.selected for d in first_pass] == [d.selected for d in second_pass]

    def test_reset_restores_position_to_zero(self):
        cfg = SamplingConfig(rate=0.2, seed_txid="pos_seed")
        sampler = AuditSampler(cfg)
        for k in range(10):
            sampler.should_record(f"x_{k}")
        sampler.reset()
        d = sampler.should_record("after_reset")
        assert d.position == 0


# ---------------------------------------------------------------------------
# AuditSampler — statistics
# ---------------------------------------------------------------------------


class TestAuditSamplerStats:
    def test_stats_total_seen_equals_call_count(self):
        cfg = SamplingConfig(rate=0.2, seed_txid="stats_tx")
        sampler, _ = _run_sampler(cfg, 42)
        assert sampler.stats["total_seen"] == 42

    def test_stats_actual_rate_close_to_target(self):
        cfg = SamplingConfig(rate=0.15, seed_txid="rate_check_tx")
        sampler, decisions = _run_sampler(cfg, 5_000)
        actual = sampler.stats["actual_rate"]
        assert abs(actual - 0.15) < 0.03

    def test_stats_is_verifiable_matches_config(self):
        cfg_yes = SamplingConfig(rate=0.1, seed_txid="vtx")
        sampler_yes, _ = _run_sampler(cfg_yes, 10)
        assert sampler_yes.stats["is_verifiable"] is True

        cfg_no = SamplingConfig(rate=0.1)
        sampler_no, _ = _run_sampler(cfg_no, 10)
        assert sampler_no.stats["is_verifiable"] is False

    def test_stats_selected_count_matches_decision_list(self):
        cfg = SamplingConfig(rate=0.3, seed_txid="count_check")
        sampler, decisions = _run_sampler(cfg, 200)
        expected = sum(1 for d in decisions if d.selected)
        assert sampler.stats["total_selected"] == expected

    def test_stats_actual_rate_zero_when_no_calls(self):
        cfg = SamplingConfig(rate=0.5)
        sampler = AuditSampler(cfg)
        assert sampler.stats["actual_rate"] == 0.0

    def test_stats_method_is_string(self):
        cfg = SamplingConfig(rate=0.1, method=SamplingMethod.SYSTEMATIC)
        sampler = AuditSampler(cfg)
        assert sampler.stats["method"] == "systematic"


# ---------------------------------------------------------------------------
# AuditSampler — should_record output fields
# ---------------------------------------------------------------------------


class TestSampleDecisionFields:
    def test_positions_are_sequential(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="seq_seed")
        sampler = AuditSampler(cfg)
        for expected_pos in range(20):
            d = sampler.should_record(f"inf_{expected_pos}")
            assert d.position == expected_pos

    def test_decision_selected_is_bool(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="bool_seed")
        sampler = AuditSampler(cfg)
        d = sampler.should_record("test_inf")
        assert isinstance(d.selected, bool)

    def test_decision_seed_hash_starts_with_sha256(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="hash_prefix_seed")
        sampler = AuditSampler(cfg)
        d = sampler.should_record("inf_0")
        assert d.seed_hash.startswith("sha256:")

    def test_decision_decided_at_is_utc(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="tz_seed")
        sampler = AuditSampler(cfg)
        d = sampler.should_record("inf_tz")
        assert d.decided_at.tzinfo == timezone.utc

    def test_decision_inference_id_preserved(self):
        cfg = SamplingConfig(rate=1.0)
        sampler = AuditSampler(cfg)
        d = sampler.should_record("my_unique_id")
        assert d.inference_id == "my_unique_id"

    def test_decision_method_matches_config(self):
        cfg = SamplingConfig(rate=0.1, method=SamplingMethod.RESERVOIR)
        sampler = AuditSampler(cfg)
        d = sampler.should_record("inf_method")
        assert d.method == SamplingMethod.RESERVOIR


# ---------------------------------------------------------------------------
# AuditSampler — verify_decision
# ---------------------------------------------------------------------------


class TestAuditSamplerVerifyDecision:
    def test_verify_decision_true_for_same_sampler(self):
        cfg = SamplingConfig(rate=0.5, seed_txid="verify_same")
        sampler = AuditSampler(cfg)
        d = sampler.should_record("inf_v")
        assert sampler.verify_decision(d) is True

    def test_verify_decision_false_for_different_seed(self):
        cfg_a = SamplingConfig(rate=0.5, seed_txid="seed_one")
        cfg_b = SamplingConfig(rate=0.5, seed_txid="seed_two")
        sampler_a = AuditSampler(cfg_a)
        sampler_b = AuditSampler(cfg_b)
        d = sampler_a.should_record("inf_v")
        assert sampler_b.verify_decision(d) is False


# ---------------------------------------------------------------------------
# VerifiableSamplingProof
# ---------------------------------------------------------------------------


class TestVerifiableSamplingProof:
    def _proof(self) -> tuple[VerifiableSamplingProof, list[str]]:
        cfg = SamplingConfig(rate=0.2, seed_txid="proof_seed_tx", seed_block=777)
        sampler = AuditSampler(cfg)
        ids = _make_ids(500)
        decisions = [sampler.should_record(i) for i in ids]
        selected_ids = [d.inference_id for d in decisions if d.selected]
        proof = VerifiableSamplingProof.generate(sampler, selected_ids)
        return proof, selected_ids

    def test_generate_produces_valid_proof(self):
        proof, _ = self._proof()
        assert proof.verify() is True

    def test_proof_hash_starts_with_sha256(self):
        proof, _ = self._proof()
        assert proof.proof_hash.startswith("sha256:")

    def test_verify_false_when_selected_count_tampered(self):
        proof, _ = self._proof()
        proof.selected_count += 1
        assert proof.verify() is False

    def test_verify_false_when_total_inferences_tampered(self):
        proof, _ = self._proof()
        proof.total_inferences += 1
        assert proof.verify() is False

    def test_verify_false_when_selected_ids_tampered(self):
        proof, _ = self._proof()
        proof.selected_inference_ids.append("malicious_id")
        assert proof.verify() is False

    def test_verify_false_when_actual_rate_tampered(self):
        proof, _ = self._proof()
        proof.actual_rate = 0.9999
        assert proof.verify() is False

    def test_proof_total_inferences_matches_sampler(self):
        cfg = SamplingConfig(rate=0.1, seed_txid="total_check")
        sampler = AuditSampler(cfg)
        decisions = [sampler.should_record(f"x_{i}") for i in range(300)]
        selected_ids = [d.inference_id for d in decisions if d.selected]
        proof = VerifiableSamplingProof.generate(sampler, selected_ids)
        assert proof.total_inferences == 300

    def test_proof_selected_count_equals_selected_ids_length(self):
        proof, selected_ids = self._proof()
        assert proof.selected_count == len(selected_ids)

    def test_proof_actual_rate_in_valid_range(self):
        proof, _ = self._proof()
        assert 0.0 <= proof.actual_rate <= 1.0

    def test_proof_generated_at_is_utc(self):
        proof, _ = self._proof()
        assert proof.generated_at.tzinfo == timezone.utc

    def test_proof_config_preserved(self):
        cfg = SamplingConfig(rate=0.2, seed_txid="proof_seed_tx", seed_block=777)
        sampler = AuditSampler(cfg)
        decisions = [sampler.should_record(f"i_{k}") for k in range(50)]
        selected_ids = [d.inference_id for d in decisions if d.selected]
        proof = VerifiableSamplingProof.generate(sampler, selected_ids)
        assert proof.config.seed_txid == "proof_seed_tx"
        assert proof.config.seed_block == 777
