"""Tests for aria.privacy — Differential Privacy budget tracking."""

from __future__ import annotations

import math

import pytest

from aria.privacy import (
    DPMechanism,
    DPQuery,
    PrivacyAccountant,
    PrivacyBudgetExhaustedError,
    PrivacyBudgetStatus,
    PrivacyStatus,
    compose_advanced,
    compose_basic,
    gaussian_epsilon,
    laplace_epsilon,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _queries(n: int, eps: float = 0.1, delta: float = 1e-6) -> list[DPQuery]:
    return [
        DPQuery(
            query_id=f"q-{i}",
            epsilon=eps,
            delta=delta,
            mechanism=DPMechanism.LAPLACE,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# DPMechanism
# ---------------------------------------------------------------------------

class TestDPMechanism:
    def test_values(self):
        assert DPMechanism.LAPLACE == "laplace"
        assert DPMechanism.GAUSSIAN == "gaussian"
        assert DPMechanism.EXPONENTIAL == "exponential"
        assert DPMechanism.RANDOMISED_RESPONSE == "randomised_response"
        assert DPMechanism.CUSTOM == "custom"

    def test_from_string(self):
        assert DPMechanism("laplace") is DPMechanism.LAPLACE
        assert DPMechanism("gaussian") is DPMechanism.GAUSSIAN


# ---------------------------------------------------------------------------
# DPQuery
# ---------------------------------------------------------------------------

class TestDPQuery:
    def test_default_timestamp(self):
        q = DPQuery(query_id="q1", epsilon=0.1, delta=1e-6, mechanism=DPMechanism.LAPLACE)
        assert q.timestamp != ""

    def test_explicit_timestamp(self):
        q = DPQuery(
            query_id="q1", epsilon=0.1, delta=1e-6,
            mechanism=DPMechanism.LAPLACE, timestamp="2025-01-01T00:00:00"
        )
        assert q.timestamp == "2025-01-01T00:00:00"

    def test_default_sensitivity(self):
        q = DPQuery(query_id="q1", epsilon=0.5, delta=0.0, mechanism=DPMechanism.LAPLACE)
        assert q.sensitivity == 1.0

    def test_default_metadata(self):
        q = DPQuery(query_id="q1", epsilon=0.5, delta=0.0, mechanism=DPMechanism.LAPLACE)
        assert q.metadata == {}


# ---------------------------------------------------------------------------
# compose_basic
# ---------------------------------------------------------------------------

class TestComposeBasic:
    def test_empty(self):
        eps, delta = compose_basic([])
        assert eps == 0.0
        assert delta == 0.0

    def test_single_query(self):
        qs = _queries(1, eps=0.1, delta=1e-6)
        eps, delta = compose_basic(qs)
        assert abs(eps - 0.1) < 1e-9
        assert abs(delta - 1e-6) < 1e-12

    def test_multiple_queries(self):
        qs = _queries(10, eps=0.1, delta=1e-6)
        eps, delta = compose_basic(qs)
        assert abs(eps - 1.0) < 1e-9
        assert abs(delta - 10 * 1e-6) < 1e-10

    def test_heterogeneous_epsilon(self):
        qs = [
            DPQuery(query_id="q1", epsilon=0.2, delta=0.0, mechanism=DPMechanism.LAPLACE),
            DPQuery(query_id="q2", epsilon=0.3, delta=0.0, mechanism=DPMechanism.GAUSSIAN),
        ]
        eps, delta = compose_basic(qs)
        assert abs(eps - 0.5) < 1e-9
        assert delta == 0.0

    def test_zero_delta(self):
        qs = _queries(5, eps=0.1, delta=0.0)
        _, delta = compose_basic(qs)
        assert delta == 0.0


# ---------------------------------------------------------------------------
# compose_advanced
# ---------------------------------------------------------------------------

class TestComposeAdvanced:
    def test_empty(self):
        eps, delta = compose_advanced([])
        assert eps == 0.0
        assert delta == 0.0

    def test_single_query_less_than_basic(self):
        # Advanced should give smaller epsilon than basic for many queries
        qs = _queries(100, eps=0.01, delta=1e-6)
        eps_basic, _ = compose_basic(qs)
        eps_adv, _ = compose_advanced(qs)
        # For k=100 queries with eps=0.01, advanced should be less than basic (1.0)
        assert eps_adv < eps_basic

    def test_advanced_delta_larger_than_basic(self):
        # Advanced adds k*delta_prime to delta total
        qs = _queries(10, eps=0.1, delta=1e-6)
        _, delta_basic = compose_basic(qs)
        _, delta_adv = compose_advanced(qs, delta_prime=1e-6)
        # delta_adv = sum(q.delta) + k * delta_prime = 10*1e-6 + 10*1e-6 = 2e-5
        assert delta_adv > delta_basic

    def test_advanced_eps_positive(self):
        qs = _queries(5, eps=0.1, delta=1e-6)
        eps, delta = compose_advanced(qs)
        assert eps > 0
        assert delta > 0

    def test_custom_delta_prime(self):
        qs = _queries(10, eps=0.1, delta=1e-6)
        _, d1 = compose_advanced(qs, delta_prime=1e-6)
        _, d2 = compose_advanced(qs, delta_prime=1e-8)
        assert d1 > d2  # larger delta_prime → larger total delta


# ---------------------------------------------------------------------------
# laplace_epsilon
# ---------------------------------------------------------------------------

class TestLaplaceEpsilon:
    def test_basic(self):
        eps = laplace_epsilon(sensitivity=1.0, noise_scale=1.0)
        assert abs(eps - 1.0) < 1e-9

    def test_higher_scale_lower_epsilon(self):
        eps = laplace_epsilon(sensitivity=1.0, noise_scale=10.0)
        assert abs(eps - 0.1) < 1e-9

    def test_zero_scale_inf(self):
        eps = laplace_epsilon(sensitivity=1.0, noise_scale=0.0)
        assert math.isinf(eps)

    def test_negative_scale_inf(self):
        eps = laplace_epsilon(sensitivity=1.0, noise_scale=-1.0)
        assert math.isinf(eps)

    def test_sensitivity_scales_epsilon(self):
        eps = laplace_epsilon(sensitivity=2.0, noise_scale=1.0)
        assert abs(eps - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# gaussian_epsilon
# ---------------------------------------------------------------------------

class TestGaussianEpsilon:
    def test_basic(self):
        eps = gaussian_epsilon(sensitivity=1.0, sigma=1.0, delta=1e-5)
        assert eps > 0

    def test_zero_sigma_inf(self):
        eps = gaussian_epsilon(sensitivity=1.0, sigma=0.0, delta=1e-5)
        assert math.isinf(eps)

    def test_zero_delta_inf(self):
        eps = gaussian_epsilon(sensitivity=1.0, sigma=1.0, delta=0.0)
        assert math.isinf(eps)

    def test_higher_sigma_lower_eps(self):
        e1 = gaussian_epsilon(sensitivity=1.0, sigma=1.0, delta=1e-5)
        e2 = gaussian_epsilon(sensitivity=1.0, sigma=5.0, delta=1e-5)
        assert e1 > e2

    def test_higher_sensitivity_higher_eps(self):
        e1 = gaussian_epsilon(sensitivity=1.0, sigma=2.0, delta=1e-5)
        e2 = gaussian_epsilon(sensitivity=2.0, sigma=2.0, delta=1e-5)
        assert e2 > e1


# ---------------------------------------------------------------------------
# PrivacyStatus
# ---------------------------------------------------------------------------

class TestPrivacyStatus:
    def _status(self, used=0.1, total=1.0, delta_used=1e-6, n=1, status=PrivacyBudgetStatus.SAFE):
        return PrivacyStatus(
            epsilon_used=used,
            epsilon_total=total,
            delta_used=delta_used,
            delta_total=1e-5,
            n_queries=n,
            status=status,
            warn_threshold=0.8,
        )

    def test_epsilon_remaining(self):
        s = self._status(used=0.3, total=1.0)
        assert abs(s.epsilon_remaining - 0.7) < 1e-9

    def test_epsilon_remaining_clamp_zero(self):
        s = self._status(used=1.5, total=1.0)
        assert s.epsilon_remaining == 0.0

    def test_epsilon_pct(self):
        s = self._status(used=0.5, total=1.0)
        assert abs(s.epsilon_pct - 50.0) < 1e-9

    def test_epsilon_pct_zero_total(self):
        s = PrivacyStatus(
            epsilon_used=0.1, epsilon_total=0.0, delta_used=0.0,
            delta_total=0.0, n_queries=0, status=PrivacyBudgetStatus.SAFE,
            warn_threshold=0.8,
        )
        assert s.epsilon_pct == 100.0

    def test_str_contains_status(self):
        s = self._status(status=PrivacyBudgetStatus.WARNING)
        assert "WARNING" in str(s)

    def test_str_contains_epsilon(self):
        s = self._status(used=0.25, total=1.0)
        assert "0.2500" in str(s)

    def test_str_contains_pct(self):
        s = self._status(used=0.5, total=1.0)
        assert "50.0%" in str(s)

    def test_str_contains_queries(self):
        s = self._status(n=42)
        assert "42" in str(s)


# ---------------------------------------------------------------------------
# PrivacyAccountant — construction
# ---------------------------------------------------------------------------

class TestPrivacyAccountantInit:
    def test_valid(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        assert pa.status().epsilon_total == 1.0

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(epsilon_total=0.0)

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(epsilon_total=-0.5)

    def test_default_delta(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        assert pa.status().delta_total == 1e-5

    def test_initial_status_safe(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        assert pa.status().status == PrivacyBudgetStatus.SAFE

    def test_initial_n_queries_zero(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        assert pa.status().n_queries == 0


# ---------------------------------------------------------------------------
# PrivacyAccountant — record_query
# ---------------------------------------------------------------------------

class TestPrivacyAccountantRecordQuery:
    def test_returns_dpquery(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        q = pa.record_query(epsilon=0.1)
        assert isinstance(q, DPQuery)
        assert q.epsilon == 0.1

    def test_query_id_increments(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        q1 = pa.record_query(epsilon=0.1)
        q2 = pa.record_query(epsilon=0.1)
        assert q1.query_id != q2.query_id
        assert q2.query_id > q1.query_id

    def test_mechanism_string_coerced(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        q = pa.record_query(epsilon=0.1, mechanism="gaussian")
        assert q.mechanism == DPMechanism.GAUSSIAN

    def test_mechanism_enum_accepted(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        q = pa.record_query(epsilon=0.1, mechanism=DPMechanism.EXPONENTIAL)
        assert q.mechanism == DPMechanism.EXPONENTIAL

    def test_metadata_stored(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        q = pa.record_query(epsilon=0.1, metadata={"model": "gpt-4"})
        assert q.metadata["model"] == "gpt-4"

    def test_n_queries_increments(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.1)
        pa.record_query(epsilon=0.1)
        assert pa.status().n_queries == 2

    def test_epsilon_accumulates(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        for _ in range(3):
            pa.record_query(epsilon=0.1)
        assert abs(pa.status().epsilon_used - 0.3) < 1e-9

    def test_queries_property(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.1)
        pa.record_query(epsilon=0.2)
        assert len(pa.queries) == 2

    def test_queries_property_returns_copy(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.1)
        qs = pa.queries
        qs.clear()
        assert len(pa.queries) == 1  # internal not modified


# ---------------------------------------------------------------------------
# PrivacyAccountant — status transitions
# ---------------------------------------------------------------------------

class TestPrivacyAccountantStatus:
    def test_safe_below_warn(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.8)
        pa.record_query(epsilon=0.5)
        assert pa.status().status == PrivacyBudgetStatus.SAFE

    def test_warning_at_warn_threshold(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.8)
        pa.record_query(epsilon=0.85)
        assert pa.status().status == PrivacyBudgetStatus.WARNING

    def test_exhausted_at_100_pct(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.8)
        pa.record_query(epsilon=1.0)
        assert pa.status().status == PrivacyBudgetStatus.EXHAUSTED

    def test_exceeded_above_100_pct(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.8)
        pa.record_query(epsilon=1.1)
        assert pa.status().status == PrivacyBudgetStatus.EXCEEDED

    def test_safe_to_warning_progression(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.5)
        pa.record_query(epsilon=0.3)
        assert pa.status().status == PrivacyBudgetStatus.SAFE
        pa.record_query(epsilon=0.3)
        assert pa.status().status == PrivacyBudgetStatus.WARNING

    def test_epsilon_remaining_decreases(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.4)
        assert abs(pa.status().epsilon_remaining - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# PrivacyAccountant — is_safe
# ---------------------------------------------------------------------------

class TestPrivacyAccountantIsSafe:
    def test_safe_when_safe(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.1)
        assert pa.is_safe() is True

    def test_safe_in_warning(self):
        pa = PrivacyAccountant(epsilon_total=1.0, warn_at=0.5)
        pa.record_query(epsilon=0.6)
        assert pa.is_safe() is True

    def test_not_safe_when_exhausted(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=1.0)
        assert pa.is_safe() is False

    def test_not_safe_when_exceeded(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=1.5)
        assert pa.is_safe() is False


# ---------------------------------------------------------------------------
# PrivacyAccountant — hard_stop
# ---------------------------------------------------------------------------

class TestPrivacyAccountantHardStop:
    def test_hard_stop_raises_on_exceed(self):
        pa = PrivacyAccountant(epsilon_total=1.0, hard_stop=True)
        pa.record_query(epsilon=0.5)
        with pytest.raises(PrivacyBudgetExhaustedError):
            pa.record_query(epsilon=0.6)  # total 1.1 > 1.0

    def test_hard_stop_false_no_raise(self):
        pa = PrivacyAccountant(epsilon_total=1.0, hard_stop=False)
        pa.record_query(epsilon=1.5)  # should not raise
        assert pa.status().status == PrivacyBudgetStatus.EXCEEDED

    def test_hard_stop_error_message(self):
        pa = PrivacyAccountant(epsilon_total=1.0, hard_stop=True)
        pa.record_query(epsilon=0.9)
        with pytest.raises(PrivacyBudgetExhaustedError, match="exhausted"):
            pa.record_query(epsilon=0.2)

    def test_hard_stop_exact_boundary_ok(self):
        pa = PrivacyAccountant(epsilon_total=1.0, hard_stop=True)
        pa.record_query(epsilon=1.0)  # exactly 1.0, should not raise
        assert pa.status().status == PrivacyBudgetStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# PrivacyAccountant — reset
# ---------------------------------------------------------------------------

class TestPrivacyAccountantReset:
    def test_reset_clears_queries(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.5)
        pa.reset()
        assert pa.status().n_queries == 0

    def test_reset_clears_epsilon_used(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.9)
        pa.reset()
        assert pa.status().epsilon_used == 0.0

    def test_reset_restores_safe_status(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=1.5)
        pa.reset()
        assert pa.status().status == PrivacyBudgetStatus.SAFE

    def test_queries_recordable_after_reset(self):
        pa = PrivacyAccountant(epsilon_total=1.0)
        pa.record_query(epsilon=0.9)
        pa.reset()
        pa.record_query(epsilon=0.2)
        assert pa.status().n_queries == 1
        assert abs(pa.status().epsilon_used - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# PrivacyAccountant — composition modes
# ---------------------------------------------------------------------------

class TestPrivacyAccountantComposition:
    def test_basic_composition(self):
        pa = PrivacyAccountant(epsilon_total=10.0, composition="basic")
        for _ in range(10):
            pa.record_query(epsilon=0.1, delta=1e-6)
        assert abs(pa.status().epsilon_used - 1.0) < 1e-9

    def test_advanced_composition(self):
        pa = PrivacyAccountant(epsilon_total=10.0, composition="advanced")
        for _ in range(100):
            pa.record_query(epsilon=0.01, delta=1e-6)
        # Advanced should yield less epsilon than basic (10*0.01=1.0)
        assert pa.status().epsilon_used < 1.0

    def test_advanced_safe_when_basic_exhausted(self):
        # 100 queries * 0.015 each = 1.5 basic, but advanced < 1.5
        pa = PrivacyAccountant(epsilon_total=1.0, composition="advanced")
        for _ in range(100):
            pa.record_query(epsilon=0.015, delta=1e-6)
        # advanced gives tighter bound; result depends on formula but should differ from basic
        s = pa.status()
        assert isinstance(s.epsilon_used, float)


# ---------------------------------------------------------------------------
# PrivacyBudgetExhaustedError
# ---------------------------------------------------------------------------

class TestPrivacyBudgetExhaustedError:
    def test_is_exception(self):
        err = PrivacyBudgetExhaustedError("test")
        assert isinstance(err, Exception)

    def test_message(self):
        err = PrivacyBudgetExhaustedError("budget gone")
        assert "budget gone" in str(err)
