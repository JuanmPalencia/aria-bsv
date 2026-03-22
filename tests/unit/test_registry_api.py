"""Tests for registry.api — FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from registry.api import app, get_storage
from registry.schemas import EpochCreate, SystemCreate
from registry.storage import RegistryStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_KEY = "test-key-42"


def _fresh_store() -> RegistryStorage:
    return RegistryStorage("sqlite://")


def _client(store: RegistryStorage | None = None) -> TestClient:
    s = store or _fresh_store()
    app.dependency_overrides[get_storage] = lambda: s
    return TestClient(app)


def _system_payload(**kwargs) -> dict:
    base = {
        "system_id": "kairos-v2",
        "system_name": "KAIROS v2",
        "operator_name": "Palencia Research",
        "eu_ai_act_risk_level": "high",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# POST /systems
# ---------------------------------------------------------------------------


class TestRegisterSystem:
    def test_creates_system_201(self):
        resp = _client().post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        assert resp.status_code == 201
        assert resp.json()["system_id"] == "kairos-v2"

    def test_missing_api_key_400(self):
        resp = _client().post("/systems", json=_system_payload())
        assert resp.status_code == 400

    def test_duplicate_system_id_409(self):
        store = _fresh_store()
        c = _client(store)
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        resp = c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        assert resp.status_code == 409

    def test_invalid_system_id_422(self):
        resp = _client().post(
            "/systems",
            json=_system_payload(system_id="UPPER_CASE"),
            headers={"x-api-key": API_KEY},
        )
        assert resp.status_code == 422

    def test_eu_ai_act_fields_in_response(self):
        resp = _client().post(
            "/systems",
            json=_system_payload(eu_ai_act_risk_level="high", eu_ai_act_article="Annex III"),
            headers={"x-api-key": API_KEY},
        )
        data = resp.json()
        assert data["eu_ai_act_risk_level"] == "high"
        assert data["eu_ai_act_article"] == "Annex III"


# ---------------------------------------------------------------------------
# GET /systems
# ---------------------------------------------------------------------------


class TestListSystems:
    def test_empty_list(self):
        resp = _client().get("/systems")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_registered_system_appears(self):
        store = _fresh_store()
        c = _client(store)
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        data = c.get("/systems").json()
        assert len(data) == 1
        assert data[0]["system_id"] == "kairos-v2"


# ---------------------------------------------------------------------------
# GET /systems/{system_id}
# ---------------------------------------------------------------------------


class TestGetSystem:
    def test_returns_system(self):
        store = _fresh_store()
        c = _client(store)
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        resp = c.get("/systems/kairos-v2")
        assert resp.status_code == 200
        assert resp.json()["system_id"] == "kairos-v2"

    def test_not_found_404(self):
        resp = _client().get("/systems/does-not-exist")
        assert resp.status_code == 404

    def test_stats_present_in_response(self):
        store = _fresh_store()
        c = _client(store)
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        data = c.get("/systems/kairos-v2").json()
        assert "total_epochs" in data
        assert "total_records" in data


# ---------------------------------------------------------------------------
# GET /systems/{system_id}/history
# ---------------------------------------------------------------------------


class TestEpochHistory:
    def _register(self, c: TestClient) -> None:
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})

    def _epoch(self, **kwargs) -> dict:
        base = {
            "epoch_id": "ep_1742848200000_0001",
            "open_txid": "a" * 64,
            "close_txid": "b" * 64,
            "records_count": 5,
            "merkle_root": "sha256:" + "f" * 64,
            "opened_at": 1_742_848_200,
            "closed_at": 1_742_848_210,
        }
        base.update(kwargs)
        return base

    def test_empty_history(self):
        store = _fresh_store()
        c = _client(store)
        self._register(c)
        assert c.get("/systems/kairos-v2/history").json() == []

    def test_history_has_epoch(self):
        store = _fresh_store()
        c = _client(store)
        self._register(c)
        c.post("/systems/kairos-v2/epochs", json=self._epoch(), headers={"x-api-key": API_KEY})
        history = c.get("/systems/kairos-v2/history").json()
        assert len(history) == 1
        assert history[0]["epoch_id"] == "ep_1742848200000_0001"

    def test_history_system_not_found(self):
        resp = _client().get("/systems/ghost/history")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /systems/{system_id}/epochs
# ---------------------------------------------------------------------------


class TestRecordEpoch:
    def _setup(self) -> tuple[TestClient, dict]:
        store = _fresh_store()
        c = _client(store)
        c.post("/systems", json=_system_payload(), headers={"x-api-key": API_KEY})
        epoch = {
            "epoch_id": "ep_1742848200000_0001",
            "open_txid": "a" * 64,
            "close_txid": "b" * 64,
            "records_count": 3,
            "merkle_root": "sha256:" + "c" * 64,
            "opened_at": 1_742_848_200,
            "closed_at": 1_742_848_210,
        }
        return c, epoch

    def test_record_epoch_201(self):
        c, epoch = self._setup()
        resp = c.post("/systems/kairos-v2/epochs", json=epoch, headers={"x-api-key": API_KEY})
        assert resp.status_code == 201
        assert resp.json()["epoch_id"] == epoch["epoch_id"]

    def test_wrong_key_401(self):
        c, epoch = self._setup()
        resp = c.post("/systems/kairos-v2/epochs", json=epoch, headers={"x-api-key": "wrong"})
        assert resp.status_code == 401

    def test_system_not_found_404(self):
        c, epoch = self._setup()
        resp = c.post("/systems/ghost/epochs", json=epoch, headers={"x-api-key": API_KEY})
        assert resp.status_code == 404
