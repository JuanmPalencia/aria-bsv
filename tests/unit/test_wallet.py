"""Tests for aria.wallet — WalletInterface contract, DirectWallet, BRC100Wallet."""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from aria.wallet.base import WalletInterface
from aria.wallet.brc100 import BRC100Wallet
from aria.core.errors import ARIABroadcastError, ARIAWalletError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_TXID = "b" * 64
# Testnet WIF key generated from bsvlib (no funds, test only).
_TESTNET_WIF = "cRGDkFkRYioJNTRcWpTMm3EpAfej7ykQMiSM1vD6drWjh4oq8LZP"
# Deliberately invalid WIF.
_BAD_WIF = "notawif"


# ---------------------------------------------------------------------------
# WalletInterface — ABC contract
# ---------------------------------------------------------------------------


class TestWalletInterfaceABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            WalletInterface()  # type: ignore[abstract]

    def test_subclass_without_method_raises(self):
        class Incomplete(WalletInterface):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        class Concrete(WalletInterface):
            async def sign_and_broadcast(self, payload: dict) -> str:
                return FAKE_TXID

        w = Concrete()
        assert isinstance(w, WalletInterface)


# ---------------------------------------------------------------------------
# DirectWallet — construction and key safety
# ---------------------------------------------------------------------------


class TestDirectWalletInit:
    def test_invalid_wif_raises_wallet_error(self):
        from aria.wallet.direct import DirectWallet
        from aria.broadcaster.base import BroadcasterInterface, TxStatus

        class FakeBroadcaster(BroadcasterInterface):
            async def broadcast(self, raw_tx: str) -> TxStatus:
                return TxStatus(txid=FAKE_TXID, propagated=True)

        with pytest.raises(ARIAWalletError) as exc_info:
            DirectWallet(wif=_BAD_WIF, broadcaster=FakeBroadcaster())

        # Error message must NEVER contain the actual WIF string.
        assert _BAD_WIF not in str(exc_info.value)
        assert "invalid key material" in str(exc_info.value)

    def test_error_message_never_contains_wif(self):
        """Even a syntactically valid but wrong-checksum WIF must not leak."""
        from aria.wallet.direct import DirectWallet
        from aria.broadcaster.base import BroadcasterInterface, TxStatus

        class FakeBroadcaster(BroadcasterInterface):
            async def broadcast(self, raw_tx: str) -> TxStatus:
                return TxStatus(txid=FAKE_TXID, propagated=True)

        garbled = "5" + "K" * 50  # plausible-looking but invalid
        with pytest.raises(ARIAWalletError) as exc_info:
            DirectWallet(wif=garbled, broadcaster=FakeBroadcaster())

        assert garbled not in str(exc_info.value)


# ---------------------------------------------------------------------------
# DirectWallet — sign_and_broadcast (mocked UTXO + broadcaster)
# ---------------------------------------------------------------------------


class TestDirectWalletSignAndBroadcast:
    def _make_wallet(self, wif: str = _TESTNET_WIF, broadcast_result=None):
        from aria.wallet.direct import DirectWallet
        from aria.broadcaster.base import BroadcasterInterface, TxStatus

        if broadcast_result is None:
            broadcast_result = TxStatus(txid=FAKE_TXID, propagated=True)

        class FakeBroadcaster(BroadcasterInterface):
            async def broadcast(self, raw_tx: str) -> TxStatus:
                return broadcast_result

        w = DirectWallet(wif=wif, broadcaster=FakeBroadcaster(), network="testnet")
        return w

    @pytest.mark.asyncio
    async def test_returns_txid_on_success(self):
        wallet = self._make_wallet()
        fake_utxo = {
            "txid": "c" * 64,
            "vout": 0,
            "satoshi": 10_000,
            "private_keys": [wallet._key],
        }

        with patch.object(wallet, "_get_utxos", return_value=[fake_utxo]):
            txid = await wallet.sign_and_broadcast({"type": "EPOCH_OPEN", "epoch_id": "ep_1"})

        assert txid == FAKE_TXID

    @pytest.mark.asyncio
    async def test_no_utxos_raises_wallet_error(self):
        wallet = self._make_wallet()

        with patch.object(wallet, "_get_utxos", return_value=[]):
            with pytest.raises(ARIAWalletError) as exc_info:
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

        assert "invalid key material" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_broadcaster_failure_raises_broadcast_error(self):
        from aria.broadcaster.base import TxStatus

        wallet = self._make_wallet(
            broadcast_result=TxStatus(txid="", propagated=False, message="rejected")
        )
        fake_utxo = {
            "txid": "c" * 64,
            "vout": 0,
            "satoshi": 10_000,
            "private_keys": [wallet._key],
        }

        with patch.object(wallet, "_get_utxos", return_value=[fake_utxo]):
            with pytest.raises(ARIABroadcastError):
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

    @pytest.mark.asyncio
    async def test_exception_never_leaks_key(self):
        """Internal exceptions from bsvlib must not surface key material."""
        wallet = self._make_wallet()

        with patch.object(wallet, "_get_utxos", side_effect=RuntimeError("internal error")):
            with pytest.raises(ARIAWalletError) as exc_info:
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

        # The WIF string used to construct the wallet must not appear in the error.
        assert _TESTNET_WIF not in str(exc_info.value)


# ---------------------------------------------------------------------------
# BRC100Wallet — sign_and_broadcast (mocked HTTP)
# ---------------------------------------------------------------------------


class TestBRC100Wallet:
    def _mock_response(self, status_code: int = 200, body: dict | None = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        body = body or {"txid": FAKE_TXID}
        resp.json.return_value = body
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=resp
            )
        else:
            resp.raise_for_status.return_value = None
        return resp

    @pytest.mark.asyncio
    async def test_returns_txid_on_success(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local")
        mock_resp = self._mock_response(200, {"txid": FAKE_TXID})

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            txid = await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

        assert txid == FAKE_TXID

    @pytest.mark.asyncio
    async def test_short_txid_raises_wallet_error(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local")
        mock_resp = self._mock_response(200, {"txid": "tooshort"})

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            with pytest.raises(ARIAWalletError):
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

    @pytest.mark.asyncio
    async def test_missing_txid_raises_wallet_error(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local")
        mock_resp = self._mock_response(200, {"status": "ok"})

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            with pytest.raises(ARIAWalletError):
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

    @pytest.mark.asyncio
    async def test_http_error_raises_wallet_error(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local")

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TransportError("refused"))
            mock_cls.return_value = mock_client

            with pytest.raises(ARIAWalletError):
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

    @pytest.mark.asyncio
    async def test_4xx_raises_wallet_error(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local")
        mock_resp = self._mock_response(401, {"error": "unauthorized"})

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            with pytest.raises(ARIAWalletError):
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

    @pytest.mark.asyncio
    async def test_error_message_does_not_expose_endpoint_secrets(self):
        """The wallet endpoint URL may contain tokens — must not appear in error."""
        secret_url = "http://wallet.local/secret-token-abc123"
        wallet = BRC100Wallet(endpoint=secret_url)

        with patch("aria.wallet.brc100.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TransportError("refused"))
            mock_cls.return_value = mock_client

            with pytest.raises(ARIAWalletError) as exc_info:
                await wallet.sign_and_broadcast({"type": "EPOCH_OPEN"})

        assert "invalid key material" in str(exc_info.value)

    def test_trailing_slash_stripped(self):
        wallet = BRC100Wallet(endpoint="http://wallet.local/")
        assert not wallet._endpoint.endswith("/")
