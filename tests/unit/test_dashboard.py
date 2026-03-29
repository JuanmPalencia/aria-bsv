"""Tests for aria.dashboard — Local web dashboard."""

from __future__ import annotations

import time
import pytest

from aria.core.record import AuditRecord
from aria.dashboard import create_dashboard_app, _gather_dashboard_data
from aria.storage.sqlite import SQLiteStorage


def _populated_storage():
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-dash",
        system_id="sys-dash",
        open_txid="tx_" + "a" * 60,
        model_hashes={"m1": "sha256:" + "a" * 64},
        state_hash="sha256:" + "b" * 64,
        opened_at=now,
    )
    for i in range(5):
        storage.save_record(AuditRecord(
            epoch_id="ep-dash",
            model_id="m1",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.85,
            latency_ms=120,
        ))
    return storage


class TestGatherDashboardData:
    def test_returns_expected_structure(self):
        storage = _populated_storage()
        data = _gather_dashboard_data(storage)
        assert "total_epochs" in data
        assert "total_records" in data
        assert "open_epochs" in data
        assert "avg_confidence" in data
        assert "avg_latency_ms" in data
        assert "models_count" in data
        assert "epochs" in data
        assert "records" in data

    def test_counts(self):
        storage = _populated_storage()
        data = _gather_dashboard_data(storage)
        assert data["total_epochs"] == 1
        assert data["total_records"] == 5
        assert data["models_count"] == 1

    def test_empty_storage(self):
        storage = SQLiteStorage(dsn="sqlite://")
        data = _gather_dashboard_data(storage)
        assert data["total_epochs"] == 0
        assert data["total_records"] == 0
        assert data["avg_confidence"] is None


class TestCreateDashboardApp:
    def test_creates_app(self):
        storage = _populated_storage()
        app = create_dashboard_app(storage)
        assert app is not None
        assert hasattr(app, "routes")

    @pytest.mark.anyio
    async def test_index_returns_html(self):
        from httpx import AsyncClient, ASGITransport

        storage = _populated_storage()
        app = create_dashboard_app(storage)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "ARIA Dashboard" in resp.text

    @pytest.mark.anyio
    async def test_api_dashboard(self):
        from httpx import AsyncClient, ASGITransport

        storage = _populated_storage()
        app = create_dashboard_app(storage)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/dashboard")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_epochs"] == 1
            assert data["total_records"] == 5
