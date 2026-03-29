"""Tests for aria.overlay — TopicManager, LookupService, OverlayClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aria.core.errors import ARIAError
from aria.overlay import (
    AdmittanceResult,
    LookupResult,
    LookupService,
    OverlayClient,
    OverlayError,
    TopicManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = "https://overlay.example.com"
_RAW_TX = "01000000" + "00" * 100  # minimal fake raw tx

def _ok_client(json_body: dict) -> MagicMock:
    """httpx.AsyncClient that returns json_body on any POST."""
    resp = MagicMock()
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = json_body
    resp.text = ""

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _err_client(status: int = 500) -> MagicMock:
    """httpx.AsyncClient that returns an HTTP error."""
    resp = MagicMock()
    resp.is_success = False
    resp.status_code = status
    resp.text = "Internal Server Error"

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=resp)
    return client


def _network_error_client() -> MagicMock:
    """httpx.AsyncClient that raises a network error."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    return client


# ---------------------------------------------------------------------------
# TopicManager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTopicManager:
    async def test_default_topic(self):
        tm = TopicManager(_BASE)
        assert tm.topic == "tm_aria_epochs"

    async def test_custom_topic(self):
        tm = TopicManager(_BASE, topic="my_topic")
        assert tm.topic == "my_topic"

    async def test_submit_admitted_response(self):
        body = {
            "txid": "aa" * 32,
            "topics": {
                "tm_aria_epochs": {"outputsToAdmit": [0], "coinsToRetain": []}
            },
        }
        client = _ok_client(body)
        tm = TopicManager(_BASE, client=client)
        result = await tm.submit(_RAW_TX)

        assert isinstance(result, AdmittanceResult)
        assert result.admitted is True
        assert 0 in result.admitted_outputs
        assert result.txid == "aa" * 32

    async def test_submit_rejected_response(self):
        body = {
            "txid": "bb" * 32,
            "topics": {
                "tm_aria_epochs": {"outputsToAdmit": [], "coinsToRetain": []}
            },
        }
        client = _ok_client(body)
        tm = TopicManager(_BASE, client=client)
        result = await tm.submit(_RAW_TX)

        assert result.admitted is False
        assert result.admitted_outputs == []

    async def test_http_error_raises_overlay_error(self):
        client = _err_client(500)
        tm = TopicManager(_BASE, client=client)
        with pytest.raises(OverlayError):
            await tm.submit(_RAW_TX)

    async def test_network_error_raises_overlay_error(self):
        client = _network_error_client()
        tm = TopicManager(_BASE, client=client)
        with pytest.raises(OverlayError):
            await tm.submit(_RAW_TX)

    async def test_api_key_sent_in_headers(self):
        body = {"txid": "cc" * 32, "topics": {"tm_aria_epochs": {"outputsToAdmit": [0]}}}
        client = _ok_client(body)
        tm = TopicManager(_BASE, api_key="test-key", client=client)
        await tm.submit(_RAW_TX)

        call_kwargs = client.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key"

    async def test_topic_sent_in_body(self):
        body = {"txid": "dd" * 32, "topics": {"custom_topic": {"outputsToAdmit": [0]}}}
        client = _ok_client(body)
        tm = TopicManager(_BASE, topic="custom_topic", client=client)
        await tm.submit(_RAW_TX)

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert "custom_topic" in sent_body["topics"]

    async def test_multiple_admitted_outputs(self):
        body = {
            "txid": "ee" * 32,
            "topics": {
                "tm_aria_epochs": {"outputsToAdmit": [0, 1, 2], "coinsToRetain": []}
            },
        }
        client = _ok_client(body)
        tm = TopicManager(_BASE, client=client)
        result = await tm.submit(_RAW_TX)
        assert result.admitted_outputs == [0, 1, 2]

    async def test_overlay_error_is_aria_error_subclass(self):
        assert issubclass(OverlayError, ARIAError)


# ---------------------------------------------------------------------------
# LookupService
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLookupService:
    async def test_default_service_name(self):
        ls = LookupService(_BASE)
        assert ls.service_name == "ls_aria"

    async def test_lookup_returns_results(self):
        body = {
            "results": [
                {
                    "txid": "aa" * 32,
                    "outputIndex": 0,
                    "beef": None,
                    "data": {"system_id": "my-system"},
                }
            ]
        }
        client = _ok_client(body)
        ls = LookupService(_BASE, client=client)
        results = await ls.lookup({"system_id": "my-system"})

        assert len(results) == 1
        assert isinstance(results[0], LookupResult)
        assert results[0].txid == "aa" * 32
        assert results[0].output_index == 0

    async def test_lookup_empty_results(self):
        client = _ok_client({"results": []})
        ls = LookupService(_BASE, client=client)
        results = await ls.lookup({"system_id": "unknown"})
        assert results == []

    async def test_lookup_multiple_results(self):
        body = {
            "results": [
                {"txid": "aa" * 32, "outputIndex": 0, "beef": None, "data": {}},
                {"txid": "bb" * 32, "outputIndex": 1, "beef": None, "data": {}},
                {"txid": "cc" * 32, "outputIndex": 0, "beef": None, "data": {}},
            ]
        }
        client = _ok_client(body)
        ls = LookupService(_BASE, client=client)
        results = await ls.lookup({})
        assert len(results) == 3

    async def test_beef_field_preserved(self):
        body = {
            "results": [
                {
                    "txid": "aa" * 32,
                    "outputIndex": 0,
                    "beef": "0100beef" + "ff" * 50,
                    "data": {},
                }
            ]
        }
        client = _ok_client(body)
        ls = LookupService(_BASE, client=client)
        results = await ls.lookup({})
        assert results[0].beef == "0100beef" + "ff" * 50

    async def test_data_field_preserved(self):
        body = {
            "results": [
                {"txid": "aa" * 32, "outputIndex": 0, "beef": None, "data": {"epoch_id": "epoch_1"}}
            ]
        }
        client = _ok_client(body)
        ls = LookupService(_BASE, client=client)
        results = await ls.lookup({})
        assert results[0].data == {"epoch_id": "epoch_1"}

    async def test_limit_sent_in_body(self):
        client = _ok_client({"results": []})
        ls = LookupService(_BASE, client=client)
        await ls.lookup({"x": 1}, limit=25)

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert sent_body["limit"] == 25

    async def test_service_name_sent_in_body(self):
        client = _ok_client({"results": []})
        ls = LookupService(_BASE, service_name="ls_custom", client=client)
        await ls.lookup({})

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert sent_body["service"] == "ls_custom"

    async def test_http_error_raises_overlay_error(self):
        ls = LookupService(_BASE, client=_err_client(404))
        with pytest.raises(OverlayError):
            await ls.lookup({})

    async def test_network_error_raises_overlay_error(self):
        ls = LookupService(_BASE, client=_network_error_client())
        with pytest.raises(OverlayError):
            await ls.lookup({})


# ---------------------------------------------------------------------------
# OverlayClient
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOverlayClient:
    def _client_with_responses(
        self,
        topic_body: dict,
        lookup_body: dict,
    ) -> tuple[OverlayClient, MagicMock]:
        """Return an OverlayClient with sequential mock responses."""
        responses = [
            _make_resp(topic_body),
            _make_resp(lookup_body),
        ]
        client = MagicMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(side_effect=responses)
        oc = OverlayClient(_BASE, client=client)
        return oc, client

    async def test_submit_epoch(self):
        body = {
            "txid": "aa" * 32,
            "topics": {"tm_aria_epochs": {"outputsToAdmit": [0]}},
        }
        oc = OverlayClient(_BASE, client=_ok_client(body))
        result = await oc.submit_epoch(_RAW_TX)
        assert result.admitted is True

    async def test_find_epochs_sends_system_id(self):
        client = _ok_client({"results": []})
        oc = OverlayClient(_BASE, client=client)
        await oc.find_epochs("my-system")

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert sent_body["query"]["system_id"] == "my-system"

    async def test_find_epochs_with_epoch_id(self):
        client = _ok_client({"results": []})
        oc = OverlayClient(_BASE, client=client)
        await oc.find_epochs("sys", epoch_id="ep_123")

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert sent_body["query"]["epoch_id"] == "ep_123"

    async def test_find_records_includes_type_filter(self):
        client = _ok_client({"results": []})
        oc = OverlayClient(_BASE, client=client)
        await oc.find_records("sys", "ep_123")

        call_kwargs = client.post.call_args[1]
        sent_body = call_kwargs.get("json", {})
        assert sent_body["query"]["type"] == "AUDIT_RECORD"

    async def test_topic_manager_and_lookup_accessible(self):
        oc = OverlayClient(_BASE)
        assert oc.topic_manager is not None
        assert oc.lookup_service is not None


def _make_resp(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = body
    resp.text = ""
    return resp
