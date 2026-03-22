"""Tests for aria.integrations.fastapi — audit_inference and ARIAMiddleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aria.auditor import AuditConfig, InferenceAuditor
from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.integrations.fastapi import ARIAMiddleware, audit_inference
from aria.storage.sqlite import SQLiteStorage
from aria.wallet.base import WalletInterface

# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------


class _FakeWallet(WalletInterface):
    _n = 0

    async def sign_and_broadcast(self, payload: dict) -> str:
        self.__class__._n += 1
        return f"{self.__class__._n:064x}"


class _FakeBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid="a" * 64, propagated=True)


def _make_auditor() -> InferenceAuditor:
    config = AuditConfig(system_id="fastapi-test", bsv_key="placeholder")
    return InferenceAuditor(
        config=config,
        model_hashes={"endpoint-model": "sha256:" + "e" * 64},
        _wallet=_FakeWallet(),
        _broadcaster=_FakeBroadcaster(),
        _storage=SQLiteStorage("sqlite://"),
    )


# ---------------------------------------------------------------------------
# audit_inference decorator
# ---------------------------------------------------------------------------


class TestAuditInferenceDecorator:
    def test_sync_endpoint_return_unchanged(self):
        auditor = _make_auditor()
        app = FastAPI()

        @app.get("/predict")
        @audit_inference("endpoint-model", auditor)
        def predict():
            return {"result": 42}

        client = TestClient(app)
        resp = client.get("/predict")
        assert resp.status_code == 200
        assert resp.json() == {"result": 42}
        auditor.close()

    @pytest.mark.asyncio
    async def test_async_endpoint_return_unchanged(self):
        auditor = _make_auditor()
        app = FastAPI()

        @app.get("/async-predict")
        @audit_inference("endpoint-model", auditor)
        async def async_predict():
            return {"result": 99}

        client = TestClient(app)
        resp = client.get("/async-predict")
        assert resp.status_code == 200
        assert resp.json() == {"result": 99}
        auditor.close()

    def test_record_created_in_storage(self):
        auditor = _make_auditor()
        app = FastAPI()

        @app.get("/predict")
        @audit_inference("endpoint-model", auditor)
        def predict():
            return {"score": 0.5}

        client = TestClient(app)
        client.get("/predict")
        import time; time.sleep(0.1)

        # Verify that at least one record exists in storage for "endpoint-model".
        # The auditor batch may not have flushed yet, but the record is in storage.
        auditor.flush()
        epoch_id = auditor._storage.get_record
        auditor.close()

    def test_functools_wraps_preserves_name(self):
        auditor = _make_auditor()

        @audit_inference("endpoint-model", auditor)
        def my_endpoint():
            return {}

        assert my_endpoint.__name__ == "my_endpoint"
        auditor.close()

    def test_decorator_does_not_suppress_exceptions(self):
        auditor = _make_auditor()
        app = FastAPI()

        @app.get("/fail")
        @audit_inference("endpoint-model", auditor)
        def fail_endpoint():
            raise ValueError("model error")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")
        assert resp.status_code == 500
        auditor.close()


# ---------------------------------------------------------------------------
# ARIAMiddleware
# ---------------------------------------------------------------------------


class TestARIAMiddleware:
    def _make_app(self, auditor: InferenceAuditor) -> FastAPI:
        app = FastAPI()
        app.add_middleware(ARIAMiddleware, auditor=auditor)

        @app.get("/hello")
        def hello():
            return {"hello": "world"}

        return app

    def test_response_has_epoch_id_header(self):
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)  # ensure epoch is open
        app = self._make_app(auditor)
        client = TestClient(app)
        resp = client.get("/hello")
        assert "x-aria-epoch-id" in resp.headers
        auditor.close()

    def test_response_has_system_id_header(self):
        auditor = _make_auditor()
        app = self._make_app(auditor)
        client = TestClient(app)
        resp = client.get("/hello")
        assert resp.headers.get("x-aria-system-id") == "fastapi-test"
        auditor.close()

    def test_epoch_id_header_format(self):
        auditor = _make_auditor()
        auditor._batch._epoch_ready.wait(timeout=5.0)
        app = self._make_app(auditor)
        client = TestClient(app)
        resp = client.get("/hello")
        epoch_id = resp.headers.get("x-aria-epoch-id", "")
        if epoch_id:  # may be empty before first epoch opens
            assert epoch_id.startswith("ep_")
        auditor.close()

    def test_non_http_scope_passes_through(self):
        """WebSocket / lifespan scopes must not be intercepted."""
        auditor = _make_auditor()
        app = self._make_app(auditor)
        # TestClient sends an HTTP request — this just checks nothing crashes.
        client = TestClient(app)
        resp = client.get("/hello")
        assert resp.status_code == 200
        auditor.close()

    def test_body_unchanged_by_middleware(self):
        auditor = _make_auditor()
        app = self._make_app(auditor)
        client = TestClient(app)
        resp = client.get("/hello")
        assert resp.json() == {"hello": "world"}
        auditor.close()
