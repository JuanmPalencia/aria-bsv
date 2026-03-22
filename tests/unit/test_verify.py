"""Tests for aria.verify — VerificationResult, _parse_aria_script, Verifier."""

from __future__ import annotations

import json
from datetime import timezone
from typing import Any

import pytest

from aria.core.hasher import hash_object
from aria.core.record import AuditRecord
from aria.core.merkle import ARIAMerkleTree
from aria.storage.sqlite import SQLiteStorage
from aria.verify import (
    TxFetcher,
    VerificationResult,
    Verifier,
    WhatsOnChainFetcher,
    _parse_aria_script,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPEN_TXID = "a" * 64
CLOSE_TXID = "b" * 64
FAKE_HASH = "sha256:" + "f" * 64
SYSTEM_ID = "kairos-v2"
EPOCH_ID = "ep_1742848200000_0001"
MODEL_HASHES = {"triage": "sha256:" + "1" * 64, "dispatch": "sha256:" + "2" * 64}


def _open_payload(
    epoch_id: str = EPOCH_ID,
    system_id: str = SYSTEM_ID,
    model_hashes: dict | None = None,
    timestamp: int = 1742848200,
) -> dict:
    return {
        "aria_version": "1.0",
        "type": "EPOCH_OPEN",
        "epoch_id": epoch_id,
        "system_id": system_id,
        "model_hashes": model_hashes if model_hashes is not None else MODEL_HASHES,
        "state_hash": FAKE_HASH,
        "timestamp": timestamp,
        "nonce": "aa" * 16,
    }


def _close_payload(
    epoch_id: str = EPOCH_ID,
    prev_txid: str = OPEN_TXID,
    records_count: int = 3,
    merkle_root: str = FAKE_HASH,
) -> dict:
    return {
        "aria_version": "1.0",
        "type": "EPOCH_CLOSE",
        "epoch_id": epoch_id,
        "prev_txid": prev_txid,
        "records_merkle_root": merkle_root,
        "records_count": records_count,
        "duration_ms": 1498,
    }


def _make_record(seq: int, model_id: str = "triage") -> AuditRecord:
    return AuditRecord(
        epoch_id=EPOCH_ID,
        model_id=model_id,
        input_hash=hash_object({"seq": seq}),
        output_hash=hash_object({"result": seq}),
        sequence=seq,
    )


class _MockFetcher(TxFetcher):
    """TxFetcher backed by a dict of txid → payload."""

    def __init__(self, payloads: dict[str, dict], close_map: dict[str, str] | None = None) -> None:
        self._payloads = payloads
        self._close_map = close_map or {}

    async def fetch_payload(self, txid: str) -> dict | None:
        return self._payloads.get(txid)

    async def find_close_txid(self, epoch_id: str, open_txid: str) -> str | None:
        return self._close_map.get(open_txid)


def _make_verifier(
    open_payload: dict | None = None,
    close_payload: dict | None = None,
    close_map: dict[str, str] | None = None,
    storage: SQLiteStorage | None = None,
) -> Verifier:
    payloads = {}
    if open_payload is not None:
        payloads[OPEN_TXID] = open_payload
    if close_payload is not None:
        payloads[CLOSE_TXID] = close_payload
    fetcher = _MockFetcher(payloads, close_map)
    return Verifier(tx_fetcher=fetcher, storage=storage)


# ---------------------------------------------------------------------------
# _parse_aria_script — unit tests
# ---------------------------------------------------------------------------


def _build_script(payload: dict) -> str:
    """Reproduce the bsvlib OP_RETURN script for an ARIA payload."""
    from bsvlib.transaction.transaction import TxOutput
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output = TxOutput(out=[b"ARIA", data], satoshi=0)
    return output.locking_script.hex()


class TestParseAriaScript:
    def test_parses_epoch_open(self):
        script = _build_script(_open_payload())
        result = _parse_aria_script(script)
        assert result is not None
        assert result["type"] == "EPOCH_OPEN"
        assert result["epoch_id"] == EPOCH_ID

    def test_parses_epoch_close(self):
        script = _build_script(_close_payload())
        result = _parse_aria_script(script)
        assert result is not None
        assert result["type"] == "EPOCH_CLOSE"

    def test_non_aria_script_returns_none(self):
        # P2PKH script (not OP_RETURN)
        assert _parse_aria_script("76a914" + "00" * 20 + "88ac") is None

    def test_empty_script_returns_none(self):
        assert _parse_aria_script("") is None

    def test_invalid_hex_returns_none(self):
        assert _parse_aria_script("not-hex") is None

    def test_wrong_prefix_returns_none(self):
        # OP_RETURN with different prefix (not ARIA)
        assert _parse_aria_script("006a04424f4d42" + "00" * 10) is None

    def test_round_trip_preserves_all_fields(self):
        payload = _open_payload()
        script = _build_script(payload)
        result = _parse_aria_script(script)
        assert result["system_id"] == payload["system_id"]
        assert result["model_hashes"] == payload["model_hashes"]
        assert result["nonce"] == payload["nonce"]

    def test_large_payload_parsed_correctly(self):
        """Payloads > 75 bytes require OP_PUSHDATA1 encoding."""
        large_payload = _open_payload(
            model_hashes={f"model-{i}": "sha256:" + str(i) * 64 for i in range(10)}
        )
        script = _build_script(large_payload)
        result = _parse_aria_script(script)
        assert result is not None
        assert len(result["model_hashes"]) == 10


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_defaults(self):
        r = VerificationResult(valid=True)
        assert r.tampered is False
        assert r.error is None
        assert r.model_id is None

    def test_err_factory(self):
        r = VerificationResult._err("something failed")
        assert r.valid is False
        assert r.error == "something failed"
        assert r.tampered is False

    def test_err_factory_tampered(self):
        r = VerificationResult._err("hash mismatch", tampered=True)
        assert r.tampered is True


# ---------------------------------------------------------------------------
# TxFetcher — ABC contract
# ---------------------------------------------------------------------------


class TestTxFetcherABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TxFetcher()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        fetcher = _MockFetcher({})
        assert isinstance(fetcher, TxFetcher)


# ---------------------------------------------------------------------------
# Verifier — verify_epoch success path
# ---------------------------------------------------------------------------


class TestVerifyEpochSuccess:
    @pytest.mark.asyncio
    async def test_valid_epoch_returns_valid_true(self):
        v = _make_verifier(_open_payload(), _close_payload())
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.valid is True
        assert result.tampered is False

    @pytest.mark.asyncio
    async def test_epoch_id_in_result(self):
        v = _make_verifier(_open_payload(), _close_payload())
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.epoch_id == EPOCH_ID

    @pytest.mark.asyncio
    async def test_system_id_in_result(self):
        v = _make_verifier(_open_payload(), _close_payload())
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.system_id == SYSTEM_ID

    @pytest.mark.asyncio
    async def test_records_count_in_result(self):
        v = _make_verifier(_open_payload(), _close_payload(records_count=7))
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.records_count == 7

    @pytest.mark.asyncio
    async def test_decided_at_is_utc_datetime(self):
        v = _make_verifier(_open_payload(timestamp=1742848200), _close_payload())
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.decided_at is not None
        assert result.decided_at.tzinfo == timezone.utc
        assert result.decided_at.year == 2025

    @pytest.mark.asyncio
    async def test_merkle_root_in_result(self):
        v = _make_verifier(_open_payload(), _close_payload(merkle_root="sha256:" + "c" * 64))
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.merkle_root == "sha256:" + "c" * 64


# ---------------------------------------------------------------------------
# Verifier — verify_epoch error paths
# ---------------------------------------------------------------------------


class TestVerifyEpochErrors:
    @pytest.mark.asyncio
    async def test_open_txid_not_found(self):
        v = _make_verifier()
        result = await v.verify_epoch("0" * 64, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_open_txid_wrong_type(self):
        wrong = {**_close_payload(), "type": "EPOCH_CLOSE"}
        fetcher = _MockFetcher({OPEN_TXID: wrong})
        v = Verifier(tx_fetcher=fetcher)
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_close_txid_not_found(self):
        v = _make_verifier(_open_payload())
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_tampered_prev_txid_detected(self):
        wrong_prev = _close_payload(prev_txid="c" * 64)  # wrong prev_txid
        v = _make_verifier(_open_payload(), wrong_prev)
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert result.tampered is True

    @pytest.mark.asyncio
    async def test_mismatched_epoch_id_detected(self):
        wrong_close = _close_payload(epoch_id="ep_wrong_0000")
        v = _make_verifier(_open_payload(), wrong_close)
        result = await v.verify_epoch(OPEN_TXID, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert result.tampered is True

    @pytest.mark.asyncio
    async def test_no_close_txid_returns_error(self):
        """If no close_txid is given and storage has none, return error."""
        v = _make_verifier(_open_payload())
        result = await v.verify_epoch(OPEN_TXID)  # no close_txid, no storage
        assert result.valid is False
        assert "not found" in (result.error or "")


# ---------------------------------------------------------------------------
# Verifier — auto-resolve close_txid from storage
# ---------------------------------------------------------------------------


class TestVerifyEpochStorageResolve:
    @pytest.mark.asyncio
    async def test_resolves_close_txid_from_storage(self):
        store = SQLiteStorage("sqlite://")
        store.save_epoch_open(EPOCH_ID, SYSTEM_ID, OPEN_TXID, MODEL_HASHES, FAKE_HASH, 1_000_000)
        store.save_epoch_close(EPOCH_ID, CLOSE_TXID, FAKE_HASH, 3, 1_000_001)

        fetcher = _MockFetcher({OPEN_TXID: _open_payload(), CLOSE_TXID: _close_payload()})
        v = Verifier(tx_fetcher=fetcher, storage=store)
        result = await v.verify_epoch(OPEN_TXID)  # no explicit close_txid
        assert result.valid is True


# ---------------------------------------------------------------------------
# Verifier — verify_record success path
# ---------------------------------------------------------------------------


class TestVerifyRecordSuccess:
    def _setup(self, n_records: int = 3) -> tuple[Verifier, list[AuditRecord], str]:
        """Build a Verifier with storage containing n_records, return (verifier, records, merkle_root)."""
        records = [_make_record(i) for i in range(n_records)]
        tree = ARIAMerkleTree()
        for rec in records:
            tree.add(rec.hash())
        root = tree.root()

        store = SQLiteStorage("sqlite://")
        store.save_epoch_open(EPOCH_ID, SYSTEM_ID, OPEN_TXID, MODEL_HASHES, FAKE_HASH, 1_000_000)
        store.save_epoch_close(EPOCH_ID, CLOSE_TXID, root, n_records, 1_000_001)
        for rec in records:
            store.save_record(rec)

        fetcher = _MockFetcher({
            OPEN_TXID: _open_payload(),
            CLOSE_TXID: _close_payload(merkle_root=root, records_count=n_records),
        })
        v = Verifier(tx_fetcher=fetcher, storage=store)
        return v, records, root

    @pytest.mark.asyncio
    async def test_valid_record_returns_valid(self):
        v, records, _ = self._setup(3)
        data = {
            "epoch_id": records[1].epoch_id,
            "model_id": records[1].model_id,
            "input_hash": records[1].input_hash,
            "output_hash": records[1].output_hash,
            "sequence": records[1].sequence,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.valid is True
        assert result.tampered is False

    @pytest.mark.asyncio
    async def test_model_id_in_result(self):
        v, records, _ = self._setup(3)
        data = {
            "epoch_id": records[0].epoch_id,
            "model_id": records[0].model_id,
            "input_hash": records[0].input_hash,
            "output_hash": records[0].output_hash,
            "sequence": records[0].sequence,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.model_id == "triage"

    @pytest.mark.asyncio
    async def test_model_version_in_result(self):
        v, records, _ = self._setup(3)
        data = {
            "epoch_id": records[0].epoch_id,
            "model_id": "triage",
            "input_hash": records[0].input_hash,
            "output_hash": records[0].output_hash,
            "sequence": records[0].sequence,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.model_version == MODEL_HASHES["triage"]


# ---------------------------------------------------------------------------
# Verifier — verify_record tamper detection
# ---------------------------------------------------------------------------


class TestVerifyRecordTampered:
    @pytest.mark.asyncio
    async def test_wrong_epoch_id_detected(self):
        v, records, _ = TestVerifyRecordSuccess()._setup(3)
        data = {
            "epoch_id": "ep_wrong_0000",  # tampered
            "model_id": "triage",
            "input_hash": records[0].input_hash,
            "output_hash": records[0].output_hash,
            "sequence": 0,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert result.tampered is True

    @pytest.mark.asyncio
    async def test_unknown_model_id_detected(self):
        v, records, _ = TestVerifyRecordSuccess()._setup(3)
        data = {
            "epoch_id": EPOCH_ID,
            "model_id": "unknown-model",  # not in model_hashes
            "input_hash": records[0].input_hash,
            "output_hash": records[0].output_hash,
            "sequence": 0,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert result.tampered is True

    @pytest.mark.asyncio
    async def test_altered_output_hash_detected(self):
        v, records, _ = TestVerifyRecordSuccess()._setup(3)
        data = {
            "epoch_id": EPOCH_ID,
            "model_id": "triage",
            "input_hash": records[0].input_hash,
            "output_hash": "sha256:" + "0" * 64,  # tampered
            "sequence": 0,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.valid is False
        assert result.tampered is True

    @pytest.mark.asyncio
    async def test_missing_record_data_field_returns_error(self):
        v, records, _ = TestVerifyRecordSuccess()._setup(3)
        data = {
            "epoch_id": EPOCH_ID,
            "model_id": "triage",
            # missing input_hash, output_hash, sequence
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        assert result.valid is False


# ---------------------------------------------------------------------------
# Verifier — verify_record without storage (no Merkle check)
# ---------------------------------------------------------------------------


class TestVerifyRecordNoStorage:
    @pytest.mark.asyncio
    async def test_without_storage_skips_merkle_check(self):
        """Without storage, Merkle verification is skipped — basic checks still run."""
        fetcher = _MockFetcher({
            OPEN_TXID: _open_payload(),
            CLOSE_TXID: _close_payload(),
        })
        v = Verifier(tx_fetcher=fetcher)  # no storage
        data = {
            "epoch_id": EPOCH_ID,
            "model_id": "triage",
            "input_hash": hash_object({"x": 1}),
            "output_hash": hash_object({"y": 2}),
            "sequence": 0,
        }
        result = await v.verify_record(OPEN_TXID, data, close_txid=CLOSE_TXID)
        # Basic checks pass; Merkle check is skipped (not tampered, valid).
        assert result.valid is True
