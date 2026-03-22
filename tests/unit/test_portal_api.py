"""Tests for portal.backend.api — verification endpoint."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aria.verify import TxFetcher, VerificationResult, Verifier
from portal.backend.api import app, get_limiter, get_verifier
from portal.backend._rate_limit import _RateLimiter

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_OPEN_TXID = "a" * 64
_EPOCH_ID = "ep_1742848200000_0001"
_SYSTEM_ID = "kairos-v2"


def _valid_result(**kwargs) -> VerificationResult:
    base = dict(
        valid=True,
        epoch_id=_EPOCH_ID,
        system_id=_SYSTEM_ID,
        records_count=5,
        merkle_root="sha256:" + "f" * 64,
        decided_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
    )
    base.update(kwargs)
    return VerificationResult(**base)


class _StubVerifier(Verifier):
    """Verifier whose fetch result is pre-configured."""

    def __init__(self, result: VerificationResult) -> None:
        self._result = result

    async def verify_epoch(self, open_txid: str, close_txid: str | None = None) -> VerificationResult:
        return self._result

    async def verify_record(self, *args: Any, **kwargs: Any) -> VerificationResult:
        return self._result


def _unlimited_limiter() -> _RateLimiter:
    return _RateLimiter(max_requests=10_000, window=60.0)


def _make_client(result: VerificationResult) -> TestClient:
    app.dependency_overrides[get_verifier] = lambda: _StubVerifier(result)
    app.dependency_overrides[get_limiter] = _unlimited_limiter
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /verify/{open_txid}
# ---------------------------------------------------------------------------


class TestVerifyEndpoint:
    def test_valid_epoch_returns_200(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.status_code == 200

    def test_valid_true_in_body(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.json()["valid"] is True

    def test_epoch_id_in_body(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.json()["epoch_id"] == _EPOCH_ID

    def test_system_id_in_body(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.json()["system_id"] == _SYSTEM_ID

    def test_records_count_in_body(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.json()["records_count"] == 5

    def test_decided_at_iso_string(self):
        resp = _make_client(_valid_result()).get(f"/verify/{_OPEN_TXID}")
        assert resp.json()["decided_at"].startswith("2026-03-22")

    def test_tampered_result_propagated(self):
        result = VerificationResult(
            valid=False, tampered=True,
            epoch_id=_EPOCH_ID, system_id=_SYSTEM_ID,
            error="hash mismatch",
        )
        resp = _make_client(result).get(f"/verify/{_OPEN_TXID}")
        body = resp.json()
        assert body["valid"] is False
        assert body["tampered"] is True
        assert "hash mismatch" in body["error"]

    def test_error_result_not_found(self):
        result = VerificationResult(
            valid=False, error="EPOCH_OPEN txid not found"
        )
        resp = _make_client(result).get(f"/verify/{'z' * 64}")
        body = resp.json()
        assert body["valid"] is False
        assert "not found" in body["error"]

    def test_close_txid_query_param_accepted(self):
        resp = _make_client(_valid_result()).get(
            f"/verify/{_OPEN_TXID}?close_txid={'b' * 64}"
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self):
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = _RateLimiter(max_requests=3, window=60.0)
        assert limiter.is_allowed("1.2.3.4") is True
        assert limiter.is_allowed("1.2.3.4") is True
        assert limiter.is_allowed("1.2.3.4") is True

    def test_blocks_after_limit(self):
        limiter = _RateLimiter(max_requests=2, window=60.0)
        limiter.is_allowed("1.2.3.4")
        limiter.is_allowed("1.2.3.4")
        assert limiter.is_allowed("1.2.3.4") is False

    def test_different_ips_are_independent(self):
        limiter = _RateLimiter(max_requests=1, window=60.0)
        limiter.is_allowed("1.1.1.1")
        assert limiter.is_allowed("2.2.2.2") is True

    def test_rate_limit_endpoint_returns_429(self):
        tight_limiter = _RateLimiter(max_requests=0, window=60.0)
        app.dependency_overrides[get_verifier] = lambda: _StubVerifier(_valid_result())
        app.dependency_overrides[get_limiter] = lambda: tight_limiter
        c = TestClient(app, raise_server_exceptions=False)
        resp = c.get(f"/verify/{_OPEN_TXID}")
        assert resp.status_code == 429
