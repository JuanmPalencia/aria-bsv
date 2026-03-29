"""Tests for aria.wallet.utxo — UTXOSet, FeeOracle, select_utxos."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aria.core.errors import ARIAWalletError
from aria.wallet.utxo import (
    UTXO,
    CoinSelection,
    FeeOracle,
    UTXOSet,
    estimate_tx_bytes,
    select_utxos,
)


# ---------------------------------------------------------------------------
# UTXO dataclass
# ---------------------------------------------------------------------------


class TestUTXO:
    def test_valid_utxo(self):
        u = UTXO(txid="a" * 64, vout=0, satoshis=1000, script_pubkey="76a914")
        assert u.satoshis == 1000

    def test_zero_satoshis_raises(self):
        with pytest.raises(ARIAWalletError):
            UTXO(txid="a" * 64, vout=0, satoshis=0, script_pubkey="")

    def test_negative_satoshis_raises(self):
        with pytest.raises(ARIAWalletError):
            UTXO(txid="a" * 64, vout=0, satoshis=-1, script_pubkey="")

    def test_short_txid_raises(self):
        with pytest.raises(ARIAWalletError):
            UTXO(txid="abc", vout=0, satoshis=100, script_pubkey="")

    def test_frozen(self):
        u = UTXO(txid="a" * 64, vout=0, satoshis=100, script_pubkey="")
        with pytest.raises((AttributeError, TypeError)):
            u.satoshis = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# estimate_tx_bytes
# ---------------------------------------------------------------------------


class TestEstimateTxBytes:
    def test_single_input_two_outputs(self):
        size = estimate_tx_bytes(1, 2)
        # 10 + 1*148 + 2*34 = 226
        assert size == 226

    def test_two_inputs_two_outputs(self):
        size = estimate_tx_bytes(2, 2)
        # 10 + 2*148 + 2*34 = 374
        assert size == 374

    def test_zero_inputs_zero_outputs(self):
        assert estimate_tx_bytes(0, 0) == 10


# ---------------------------------------------------------------------------
# select_utxos
# ---------------------------------------------------------------------------


def _utxos(*sats: int) -> list[UTXO]:
    return [
        UTXO(txid=f"{'a' * 62}{i:02d}", vout=0, satoshis=s, script_pubkey="76a914")
        for i, s in enumerate(sats)
    ]


class TestSelectUtxos:
    def test_single_utxo_sufficient(self):
        utxos = _utxos(100_000)
        sel = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        assert sel.sufficient
        assert len(sel.inputs) == 1
        assert sel.inputs[0].satoshis == 100_000

    def test_change_is_correct(self):
        utxos = _utxos(100_000)
        sel = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        assert sel.change_satoshis == sel.total_in - 1_000 - sel.fee_satoshis

    def test_fee_is_positive(self):
        utxos = _utxos(100_000)
        sel = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        assert sel.fee_satoshis > 0

    def test_multiple_utxos_selected_when_needed(self):
        utxos = _utxos(500, 500, 500)
        sel = select_utxos(utxos, target_satoshis=800, fee_rate=1.0)
        assert len(sel.inputs) >= 2

    def test_largest_first_selection(self):
        # Largest UTXO should be tried first
        utxos = _utxos(200, 100, 50, 10_000)
        sel = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        # The 10_000 sat UTXO should be selected alone
        assert len(sel.inputs) == 1
        assert sel.inputs[0].satoshis == 10_000

    def test_insufficient_funds_raises(self):
        utxos = _utxos(100, 100)
        with pytest.raises(ARIAWalletError, match="Insufficient"):
            select_utxos(utxos, target_satoshis=100_000, fee_rate=1.0)

    def test_empty_utxos_raises(self):
        with pytest.raises(ARIAWalletError):
            select_utxos([], target_satoshis=1_000, fee_rate=1.0)

    def test_zero_target_raises(self):
        utxos = _utxos(1_000)
        with pytest.raises(ARIAWalletError):
            select_utxos(utxos, target_satoshis=0, fee_rate=1.0)

    def test_negative_target_raises(self):
        utxos = _utxos(1_000)
        with pytest.raises(ARIAWalletError):
            select_utxos(utxos, target_satoshis=-1, fee_rate=1.0)

    def test_total_in_equals_sum_of_selected(self):
        utxos = _utxos(5_000, 3_000, 1_000)
        sel = select_utxos(utxos, target_satoshis=2_000, fee_rate=1.0)
        assert sel.total_in == sum(u.satoshis for u in sel.inputs)

    def test_higher_fee_rate_increases_fee(self):
        utxos = _utxos(100_000)
        sel_low = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        sel_high = select_utxos(utxos, target_satoshis=1_000, fee_rate=5.0)
        assert sel_high.fee_satoshis > sel_low.fee_satoshis

    def test_unsorted_input_still_selects_correctly(self):
        # Pass UTXOs in random order — function should sort internally
        utxos = _utxos(100, 50_000, 200, 300)
        sel = select_utxos(utxos, target_satoshis=1_000, fee_rate=1.0)
        # 50_000 sat UTXO should be selected first (largest)
        assert sel.inputs[0].satoshis == 50_000


# ---------------------------------------------------------------------------
# FeeOracle
# ---------------------------------------------------------------------------

def _mock_client(rate_response: dict) -> MagicMock:
    """Return a mock httpx.AsyncClient that returns *rate_response* as JSON."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = rate_response

    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=resp)
    # Support async context manager usage (not used here since we pass client directly)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
class TestFeeOracle:
    async def test_returns_default_rate_on_empty_response(self):
        client = _mock_client({})
        oracle = FeeOracle(client=client)
        rate = await oracle.get_rate()
        assert rate == 1.0  # _DEFAULT_FEE_RATE

    async def test_parses_arc_policy_response(self):
        payload = {"policy": {"miningFee": {"satoshis": 5, "bytes": 1}}}
        client = _mock_client(payload)
        oracle = FeeOracle(client=client)
        rate = await oracle.get_rate()
        assert rate == pytest.approx(5.0)

    async def test_caches_result(self):
        payload = {"policy": {"miningFee": {"satoshis": 2, "bytes": 1}}}
        client = _mock_client(payload)
        oracle = FeeOracle(client=client, ttl_seconds=60.0)
        await oracle.get_rate()
        await oracle.get_rate()  # second call uses cache
        assert client.get.call_count == 1

    async def test_invalidate_forces_refetch(self):
        payload = {"policy": {"miningFee": {"satoshis": 2, "bytes": 1}}}
        client = _mock_client(payload)
        oracle = FeeOracle(client=client, ttl_seconds=60.0)
        await oracle.get_rate()
        oracle.invalidate()
        await oracle.get_rate()
        assert client.get.call_count == 2

    async def test_falls_back_to_default_on_error(self):
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        oracle = FeeOracle(client=client)
        rate = await oracle.get_rate()
        assert rate == 1.0

    async def test_rate_per_byte_calculation(self):
        # 10 sat per 2 bytes → 5 sat/byte
        payload = {"policy": {"miningFee": {"satoshis": 10, "bytes": 2}}}
        client = _mock_client(payload)
        oracle = FeeOracle(client=client)
        rate = await oracle.get_rate()
        assert rate == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# UTXOSet
# ---------------------------------------------------------------------------

def _woc_response(*sats: int) -> list[dict]:
    return [
        {
            "tx_hash": "a" * 62 + f"{i:02d}",
            "tx_pos": 0,
            "value": s,
            "height": 800000 + i,
            "script": "76a914" + "00" * 20 + "88ac",
        }
        for i, s in enumerate(sats)
    ]


def _mock_woc_client(sats: tuple[int, ...]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _woc_response(*sats)

    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
class TestUTXOSet:
    async def test_get_returns_utxos(self):
        client = _mock_woc_client((10_000, 5_000))
        uset = UTXOSet("1A1zP...", client=client)
        utxos = await uset.get()
        assert len(utxos) == 2

    async def test_sorted_largest_first(self):
        client = _mock_woc_client((1_000, 50_000, 500))
        uset = UTXOSet("1A1zP...", client=client)
        utxos = await uset.get()
        assert utxos[0].satoshis == 50_000

    async def test_caches_result(self):
        client = _mock_woc_client((10_000,))
        uset = UTXOSet("1A1zP...", client=client, ttl_seconds=60.0)
        await uset.get()
        await uset.get()
        assert client.get.call_count == 1

    async def test_force_refresh_bypasses_cache(self):
        client = _mock_woc_client((10_000,))
        uset = UTXOSet("1A1zP...", client=client, ttl_seconds=60.0)
        await uset.get()
        await uset.get(force_refresh=True)
        assert client.get.call_count == 2

    async def test_invalidate_forces_refetch(self):
        client = _mock_woc_client((10_000,))
        uset = UTXOSet("1A1zP...", client=client, ttl_seconds=60.0)
        await uset.get()
        uset.invalidate()
        await uset.get()
        assert client.get.call_count == 2

    async def test_total_satoshis(self):
        client = _mock_woc_client((10_000, 5_000, 3_000))
        uset = UTXOSet("1A1zP...", client=client)
        total = await uset.total_satoshis()
        assert total == 18_000

    async def test_http_error_raises_aria_wallet_error(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        uset = UTXOSet("1A1zP...", client=client)
        with pytest.raises(ARIAWalletError):
            await uset.get()

    async def test_network_error_raises_aria_wallet_error(self):
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        uset = UTXOSet("1A1zP...", client=client)
        with pytest.raises(ARIAWalletError):
            await uset.get()

    async def test_address_property(self):
        uset = UTXOSet("1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf")
        assert uset.address == "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf"

    async def test_skips_malformed_utxos(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # Mix valid and malformed entries
        resp.json.return_value = [
            {"tx_hash": "a" * 64, "tx_pos": 0, "value": 1_000},
            {"tx_hash": "short", "tx_pos": 0, "value": 500},   # bad txid
            {"tx_hash": "b" * 64, "tx_pos": 0, "value": 2_000},
        ]
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(return_value=resp)
        uset = UTXOSet("1A1zP...", client=client)
        utxos = await uset.get()
        assert len(utxos) == 2  # malformed one skipped
