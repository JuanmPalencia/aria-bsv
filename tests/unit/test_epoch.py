"""Tests for aria.core.epoch — EpochManager Pre-Commitment Protocol."""

from __future__ import annotations

import re
import pytest
from unittest.mock import AsyncMock, patch

from aria.core.epoch import EpochConfig, EpochManager, EpochOpenResult, _EMPTY_ROOT
from aria.core.record import AuditRecord
from aria.core.hasher import hash_object
from aria.core.merkle import ARIAMerkleTree, verify_proof
from aria.core.errors import ARIAError, ARIAWalletError, ARIABroadcastError
from aria.wallet.base import WalletInterface
from aria.broadcaster.base import BroadcasterInterface, TxStatus

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

OPEN_TXID = "a" * 64
CLOSE_TXID = "b" * 64

_txid_sequence: list[str] = []


class _SequencedWallet(WalletInterface):
    """Returns txids from a pre-set sequence so open/close can be distinguished."""

    def __init__(self, txids: list[str]) -> None:
        self._txids = list(txids)
        self._calls: list[dict] = []

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        self._calls.append(payload)
        return self._txids.pop(0)


class _OkBroadcaster(BroadcasterInterface):
    async def broadcast(self, raw_tx: str) -> TxStatus:
        return TxStatus(txid=OPEN_TXID, propagated=True)


def _make_manager(txids: list[str] | None = None) -> tuple[EpochManager, _SequencedWallet]:
    config = EpochConfig(system_id="test-system-1")
    wallet = _SequencedWallet(txids or [OPEN_TXID, CLOSE_TXID])
    broadcaster = _OkBroadcaster()
    mgr = EpochManager(config=config, wallet=wallet, broadcaster=broadcaster)
    return mgr, wallet


def _make_record(epoch_id: str, sequence: int, model_id: str = "model-a") -> AuditRecord:
    return AuditRecord(
        epoch_id=epoch_id,
        model_id=model_id,
        input_hash=hash_object({"seq": sequence}),
        output_hash=hash_object({"result": sequence * 2}),
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# EpochConfig
# ---------------------------------------------------------------------------


class TestEpochConfig:
    def test_defaults(self):
        cfg = EpochConfig(system_id="sys-1")
        assert cfg.network == "mainnet"

    def test_custom_network(self):
        cfg = EpochConfig(system_id="sys-1", network="testnet")
        assert cfg.network == "testnet"


# ---------------------------------------------------------------------------
# EpochManager — open_epoch
# ---------------------------------------------------------------------------


class TestOpenEpoch:
    @pytest.mark.asyncio
    async def test_returns_epoch_open_result(self):
        mgr, _ = _make_manager()
        result = await mgr.open_epoch(
            model_hashes={"model-a": "sha256:" + "f" * 64},
            system_state={"version": "1.0"},
        )
        assert result.txid == OPEN_TXID
        assert result.epoch_id.startswith("ep_")

    @pytest.mark.asyncio
    async def test_epoch_id_format(self):
        mgr, _ = _make_manager()
        result = await mgr.open_epoch(
            model_hashes={},
            system_state={},
        )
        # Format: ep_{timestamp_ms}_{counter:04d}
        assert re.match(r"^ep_\d{13}_\d{4}$", result.epoch_id)

    @pytest.mark.asyncio
    async def test_epoch_ids_monotonically_increase(self):
        mgr, _ = _make_manager(txids=["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        r1 = await mgr.open_epoch(model_hashes={}, system_state={})
        r2 = await mgr.open_epoch(model_hashes={}, system_state={})
        # Counter suffix must be strictly increasing.
        counter1 = int(r1.epoch_id.split("_")[-1])
        counter2 = int(r2.epoch_id.split("_")[-1])
        assert counter2 > counter1

    @pytest.mark.asyncio
    async def test_payload_type_is_epoch_open(self):
        mgr, wallet = _make_manager()
        await mgr.open_epoch(
            model_hashes={"m": "sha256:" + "0" * 64},
            system_state={"k": "v"},
        )
        payload = wallet._calls[0]
        assert payload["type"] == "EPOCH_OPEN"
        assert payload["aria_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_payload_contains_system_id(self):
        mgr, wallet = _make_manager()
        await mgr.open_epoch(model_hashes={}, system_state={})
        assert wallet._calls[0]["system_id"] == "test-system-1"

    @pytest.mark.asyncio
    async def test_payload_model_hashes_match(self):
        mgr, wallet = _make_manager()
        mh = {"model-x": "sha256:" + "a" * 64, "model-y": "sha256:" + "b" * 64}
        await mgr.open_epoch(model_hashes=mh, system_state={})
        assert wallet._calls[0]["model_hashes"] == mh

    @pytest.mark.asyncio
    async def test_state_hash_is_deterministic(self):
        mgr, wallet = _make_manager(txids=["a" * 64, "b" * 64])
        state = {"fleet_size": 10, "active": True}
        await mgr.open_epoch(model_hashes={}, system_state=state)
        received_hash = wallet._calls[0]["state_hash"]
        assert received_hash == hash_object(state)

    @pytest.mark.asyncio
    async def test_nonce_is_16_byte_hex(self):
        mgr, wallet = _make_manager()
        await mgr.open_epoch(model_hashes={}, system_state={})
        nonce: str = wallet._calls[0]["nonce"]
        assert len(nonce) == 32  # 16 bytes = 32 hex chars
        assert all(c in "0123456789abcdef" for c in nonce)

    @pytest.mark.asyncio
    async def test_nonces_are_unique_across_epochs(self):
        mgr, wallet = _make_manager(txids=["a" * 64, "b" * 64])
        await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.open_epoch(model_hashes={}, system_state={})
        nonce1 = wallet._calls[0]["nonce"]
        nonce2 = wallet._calls[1]["nonce"]
        assert nonce1 != nonce2

    @pytest.mark.asyncio
    async def test_result_stored_in_open_epochs(self):
        mgr, _ = _make_manager()
        result = await mgr.open_epoch(model_hashes={}, system_state={})
        assert result.epoch_id in mgr._open_epochs


# ---------------------------------------------------------------------------
# EpochManager — close_epoch
# ---------------------------------------------------------------------------


class TestCloseEpoch:
    @pytest.mark.asyncio
    async def test_returns_epoch_close_result(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        assert close_r.txid == CLOSE_TXID
        assert close_r.epoch_id == open_r.epoch_id

    @pytest.mark.asyncio
    async def test_prev_txid_links_to_open(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        assert close_r.prev_txid == open_r.txid

    @pytest.mark.asyncio
    async def test_payload_type_is_epoch_close(self):
        mgr, wallet = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        close_payload = wallet._calls[1]
        assert close_payload["type"] == "EPOCH_CLOSE"
        assert close_payload["aria_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_close_payload_prev_txid_field(self):
        mgr, wallet = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        close_payload = wallet._calls[1]
        assert close_payload["prev_txid"] == OPEN_TXID

    @pytest.mark.asyncio
    async def test_empty_epoch_uses_empty_root(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        assert close_r.merkle_root == _EMPTY_ROOT
        assert close_r.records_count == 0

    @pytest.mark.asyncio
    async def test_records_count_is_accurate(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        records = [_make_record(open_r.epoch_id, i) for i in range(5)]
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=records)
        assert close_r.records_count == 5

    @pytest.mark.asyncio
    async def test_merkle_root_matches_manual_tree(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        records = [_make_record(open_r.epoch_id, i) for i in range(3)]
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=records)

        # Manually compute the expected root.
        tree = ARIAMerkleTree()
        for rec in sorted(records, key=lambda r: r.sequence):
            tree.add(rec.hash())
        assert close_r.merkle_root == tree.root()

    @pytest.mark.asyncio
    async def test_records_sorted_by_sequence_regardless_of_input_order(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        # Pass records in reverse order.
        records = [_make_record(open_r.epoch_id, i) for i in range(4)]
        shuffled = list(reversed(records))
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=shuffled)

        tree = ARIAMerkleTree()
        for rec in records:  # sorted order
            tree.add(rec.hash())
        assert close_r.merkle_root == tree.root()

    @pytest.mark.asyncio
    async def test_epoch_removed_from_open_epochs_after_close(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        assert open_r.epoch_id not in mgr._open_epochs

    @pytest.mark.asyncio
    async def test_close_unknown_epoch_raises_aria_error(self):
        mgr, _ = _make_manager()
        with pytest.raises(ARIAError, match="never opened"):
            await mgr.close_epoch(epoch_id="ep_9999999999999_9999", records=[])

    @pytest.mark.asyncio
    async def test_close_already_closed_epoch_raises(self):
        mgr, _ = _make_manager(txids=["a" * 64, "b" * 64, "c" * 64])
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        with pytest.raises(ARIAError):
            await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])

    @pytest.mark.asyncio
    async def test_duration_ms_is_non_negative(self):
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=[])
        assert close_r.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_merkle_proof_verifiable_from_close_root(self):
        """Records hashed into EPOCH_CLOSE must be individually verifiable via Merkle proof."""
        mgr, _ = _make_manager()
        open_r = await mgr.open_epoch(model_hashes={}, system_state={})
        records = [_make_record(open_r.epoch_id, i) for i in range(4)]
        close_r = await mgr.close_epoch(epoch_id=open_r.epoch_id, records=records)

        # Verify each record independently.
        tree = ARIAMerkleTree()
        for rec in records:
            tree.add(rec.hash())

        for rec in records:
            proof = tree.proof(rec.hash())
            assert verify_proof(close_r.merkle_root, proof, rec.hash())


# ---------------------------------------------------------------------------
# EpochManager — multiple concurrent open epochs
# ---------------------------------------------------------------------------


class TestConcurrentEpochs:
    @pytest.mark.asyncio
    async def test_two_epochs_open_simultaneously(self):
        mgr, _ = _make_manager(txids=["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        r1 = await mgr.open_epoch(model_hashes={}, system_state={"epoch": 1})
        r2 = await mgr.open_epoch(model_hashes={}, system_state={"epoch": 2})
        assert r1.epoch_id != r2.epoch_id
        assert len(mgr._open_epochs) == 2

    @pytest.mark.asyncio
    async def test_closing_one_does_not_affect_other(self):
        mgr, _ = _make_manager(txids=["a" * 64, "b" * 64, "c" * 64, "d" * 64])
        r1 = await mgr.open_epoch(model_hashes={}, system_state={})
        r2 = await mgr.open_epoch(model_hashes={}, system_state={})
        await mgr.close_epoch(epoch_id=r1.epoch_id, records=[])
        # r2 must still be open.
        assert r2.epoch_id in mgr._open_epochs
        assert r1.epoch_id not in mgr._open_epochs
