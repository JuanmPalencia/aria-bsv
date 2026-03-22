"""Tests for aria.broadcaster — BroadcasterInterface contract and ARCBroadcaster."""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from aria.broadcaster.base import BroadcasterInterface, TxStatus
from aria.broadcaster.arc import ARCBroadcaster, _PROPAGATED_STATUSES
from aria.core.errors import ARIABroadcastError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TXID = "a" * 64


def _arc_response(status_code: int = 200, txid: str = FAKE_TXID, tx_status: str = "SEEN_ON_NETWORK", text: str = "") -> MagicMock:
    """Build a mock httpx.Response for ARC."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or f'{{"txid":"{txid}","txStatus":"{tx_status}"}}'
    resp.json.return_value = {"txid": txid, "txStatus": tx_status}
    return resp


# ---------------------------------------------------------------------------
# BroadcasterInterface — ABC contract
# ---------------------------------------------------------------------------


class TestBroadcasterInterfaceABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BroadcasterInterface()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_broadcast(self):
        class Incomplete(BroadcasterInterface):
            pass  # missing broadcast

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self):
        class Concrete(BroadcasterInterface):
            async def broadcast(self, raw_tx: str) -> TxStatus:
                return TxStatus(txid=FAKE_TXID, propagated=True)

        bc = Concrete()
        assert isinstance(bc, BroadcasterInterface)


# ---------------------------------------------------------------------------
# TxStatus
# ---------------------------------------------------------------------------


class TestTxStatus:
    def test_defaults(self):
        ts = TxStatus(txid=FAKE_TXID, propagated=True)
        assert ts.message == ""

    def test_fields(self):
        ts = TxStatus(txid=FAKE_TXID, propagated=False, message="error")
        assert ts.txid == FAKE_TXID
        assert not ts.propagated
        assert ts.message == "error"


# ---------------------------------------------------------------------------
# ARCBroadcaster — construction
# ---------------------------------------------------------------------------


class TestARCBroadcasterInit:
    def test_default_url(self):
        bc = ARCBroadcaster()
        assert "arc.taal.com" in bc._api_url

    def test_trailing_slash_stripped(self):
        bc = ARCBroadcaster(api_url="https://arc.example.com/")
        assert not bc._api_url.endswith("/")

    def test_no_api_key_no_auth_header(self):
        bc = ARCBroadcaster()
        assert "Authorization" not in bc._headers

    def test_api_key_adds_bearer_header(self):
        bc = ARCBroadcaster(api_key="secret")
        assert bc._headers["Authorization"] == "Bearer secret"

    def test_content_type_always_present(self):
        bc = ARCBroadcaster()
        assert bc._headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# ARCBroadcaster.broadcast — success paths
# ---------------------------------------------------------------------------


class TestARCBroadcastSuccess:
    @pytest.mark.asyncio
    async def test_returns_txstatus_on_200(self):
        bc = ARCBroadcaster(base_delay=0)
        mock_resp = _arc_response(200, txid=FAKE_TXID, tx_status="SEEN_ON_NETWORK")

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await bc.broadcast("deadbeef")

        assert result.txid == FAKE_TXID
        assert result.propagated is True
        assert result.message == "SEEN_ON_NETWORK"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", sorted(_PROPAGATED_STATUSES))
    async def test_all_propagated_statuses_return_true(self, status: str):
        bc = ARCBroadcaster(base_delay=0)
        mock_resp = _arc_response(200, txid=FAKE_TXID, tx_status=status)

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await bc.broadcast("deadbeef")

        assert result.propagated is True

    @pytest.mark.asyncio
    async def test_unknown_status_propagated_false(self):
        bc = ARCBroadcaster(base_delay=0)
        mock_resp = _arc_response(200, txid=FAKE_TXID, tx_status="UNKNOWN_STATUS")

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await bc.broadcast("deadbeef")

        assert result.propagated is False


# ---------------------------------------------------------------------------
# ARCBroadcaster.broadcast — error and retry paths
# ---------------------------------------------------------------------------


class TestARCBroadcastErrors:
    @pytest.mark.asyncio
    async def test_4xx_raises_immediately_no_retry(self):
        bc = ARCBroadcaster(max_retries=3, base_delay=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "bad request"

        call_count = 0

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_resp

            mock_client.post = post_side_effect
            mock_client_cls.return_value = mock_client

            with pytest.raises(ARIABroadcastError, match="rejected"):
                await bc.broadcast("deadbeef")

        assert call_count == 1  # no retry on 4xx

    @pytest.mark.asyncio
    async def test_transport_error_retries_then_raises(self):
        bc = ARCBroadcaster(max_retries=3, base_delay=0)

        call_count = 0

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise httpx.TransportError("connection refused")

            mock_client.post = post_side_effect
            mock_client_cls.return_value = mock_client

            with pytest.raises(ARIABroadcastError, match="failed after 3 attempts"):
                await bc.broadcast("deadbeef")

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        bc = ARCBroadcaster(max_retries=3, base_delay=0)
        ok_resp = _arc_response(200, txid=FAKE_TXID, tx_status="SEEN_ON_NETWORK")

        call_count = 0

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx.TransportError("timeout")
                return ok_resp

            mock_client.post = post_side_effect
            mock_client_cls.return_value = mock_client

            result = await bc.broadcast("deadbeef")

        assert result.propagated is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_5xx_retries_then_raises(self):
        bc = ARCBroadcaster(max_retries=2, base_delay=0)

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "service unavailable"

        call_count = 0

        with patch("aria.broadcaster.arc.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def post_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_resp

            mock_client.post = post_side_effect
            mock_client_cls.return_value = mock_client

            with pytest.raises(ARIABroadcastError, match="failed after 2 attempts"):
                await bc.broadcast("deadbeef")

        assert call_count == 2
