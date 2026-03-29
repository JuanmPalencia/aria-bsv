"""Tests for aria.dataset — DatasetAnchorer and verify_dataset_anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.core.errors import ARIAError
from aria.core.hasher import canonical_json
from aria.dataset import (
    DatasetAnchor,
    DatasetAnchorer,
    hash_bytes,
    hash_columns,
    verify_dataset_anchor,
    _guess_media_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_TXID = "ab" * 32


def _fake_wallet(txid: str = _FAKE_TXID) -> MagicMock:
    wallet = MagicMock()
    wallet.sign_and_broadcast = AsyncMock(return_value=txid)
    return wallet


def _anchorer(wallet=None) -> DatasetAnchorer:
    return DatasetAnchorer("test-system", wallet=wallet or _fake_wallet())


# ---------------------------------------------------------------------------
# hash_bytes
# ---------------------------------------------------------------------------


class TestHashBytes:
    def test_returns_sha256_prefixed(self):
        result = hash_bytes(b"hello")
        assert result.startswith("sha256:")

    def test_hex_part_is_64_chars(self):
        result = hash_bytes(b"hello")
        assert len(result.split("sha256:")[1]) == 64

    def test_deterministic(self):
        assert hash_bytes(b"data") == hash_bytes(b"data")

    def test_different_data_different_hash(self):
        assert hash_bytes(b"a") != hash_bytes(b"b")

    def test_empty_bytes(self):
        # SHA-256 of b"" is well-defined
        expected = "sha256:" + hashlib.sha256(b"").hexdigest()
        assert hash_bytes(b"") == expected

    def test_matches_hashlib(self):
        data = b"some dataset content"
        expected = "sha256:" + hashlib.sha256(data).hexdigest()
        assert hash_bytes(data) == expected


# ---------------------------------------------------------------------------
# hash_columns
# ---------------------------------------------------------------------------


class TestHashColumns:
    def test_returns_sha256_prefixed(self):
        assert hash_columns(["age", "name"]).startswith("sha256:")

    def test_order_independent(self):
        assert hash_columns(["b", "a"]) == hash_columns(["a", "b"])

    def test_different_columns_different_hash(self):
        assert hash_columns(["a", "b"]) != hash_columns(["a", "c"])

    def test_single_column(self):
        result = hash_columns(["id"])
        assert result.startswith("sha256:")

    def test_empty_columns(self):
        result = hash_columns([])
        assert result.startswith("sha256:")


# ---------------------------------------------------------------------------
# verify_dataset_anchor
# ---------------------------------------------------------------------------


class TestVerifyDatasetAnchor:
    def _make_anchor(self, data: bytes) -> DatasetAnchor:
        return DatasetAnchor(
            dataset_id="test-id",
            system_id="sys",
            content_hash=hash_bytes(data),
            schema_hash=None,
            row_count=None,
            column_names=None,
            media_type="application/octet-stream",
            anchored_at="2025-01-01T00:00:00+00:00",
            txid=_FAKE_TXID,
            payload={},
        )

    def test_valid_data_returns_true(self):
        data = b"original dataset"
        anchor = self._make_anchor(data)
        assert verify_dataset_anchor(data, anchor) is True

    def test_tampered_data_returns_false(self):
        data = b"original dataset"
        anchor = self._make_anchor(data)
        assert verify_dataset_anchor(b"tampered dataset", anchor) is False

    def test_empty_bytes_match(self):
        data = b""
        anchor = self._make_anchor(data)
        assert verify_dataset_anchor(data, anchor) is True


# ---------------------------------------------------------------------------
# DatasetAnchorer.anchor_bytes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnchorBytes:
    async def test_returns_dataset_anchor(self):
        result = await _anchorer().anchor_bytes(b"data")
        assert isinstance(result, DatasetAnchor)

    async def test_content_hash_matches(self):
        data = b"my dataset"
        result = await _anchorer().anchor_bytes(data)
        assert result.content_hash == hash_bytes(data)

    async def test_system_id_set(self):
        result = await _anchorer().anchor_bytes(b"x")
        assert result.system_id == "test-system"

    async def test_txid_set_when_wallet_provided(self):
        result = await _anchorer().anchor_bytes(b"x")
        assert result.txid == _FAKE_TXID

    async def test_txid_empty_without_wallet(self):
        anchorer = DatasetAnchorer("test-system", wallet=None)
        result = await anchorer.anchor_bytes(b"x")
        assert result.txid == ""

    async def test_media_type_default(self):
        result = await _anchorer().anchor_bytes(b"x")
        assert result.media_type == "application/octet-stream"

    async def test_media_type_custom(self):
        result = await _anchorer().anchor_bytes(b"x", media_type="text/csv")
        assert result.media_type == "text/csv"

    async def test_row_count_stored(self):
        result = await _anchorer().anchor_bytes(b"x", row_count=1000)
        assert result.row_count == 1000

    async def test_column_names_stored_sorted(self):
        result = await _anchorer().anchor_bytes(
            b"x", column_names=["z", "a", "m"]
        )
        assert result.column_names == ["z", "a", "m"]  # stored as-is

    async def test_schema_hash_computed_when_columns_provided(self):
        result = await _anchorer().anchor_bytes(
            b"x", column_names=["a", "b"]
        )
        assert result.schema_hash == hash_columns(["a", "b"])

    async def test_schema_hash_none_without_columns(self):
        result = await _anchorer().anchor_bytes(b"x")
        assert result.schema_hash is None

    async def test_dataset_id_is_unique(self):
        a1 = await _anchorer().anchor_bytes(b"x")
        a2 = await _anchorer().anchor_bytes(b"x")
        assert a1.dataset_id != a2.dataset_id

    async def test_payload_contains_required_keys(self):
        result = await _anchorer().anchor_bytes(b"x")
        assert result.payload["type"] == "DATASET_ANCHOR"
        assert "brc121_version" in result.payload
        assert "content_hash" in result.payload
        assert "nonce" in result.payload
        assert "anchored_at" in result.payload

    async def test_wallet_error_raises_aria_error(self):
        wallet = MagicMock()
        wallet.sign_and_broadcast = AsyncMock(side_effect=RuntimeError("no funds"))
        anchorer = DatasetAnchorer("sys", wallet=wallet)
        with pytest.raises(ARIAError):
            await anchorer.anchor_bytes(b"data")


# ---------------------------------------------------------------------------
# DatasetAnchorer.anchor_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnchorJson:
    async def test_returns_anchor(self):
        result = await _anchorer().anchor_json({"key": "value"})
        assert isinstance(result, DatasetAnchor)

    async def test_media_type_is_json(self):
        result = await _anchorer().anchor_json({})
        assert result.media_type == "application/json"

    async def test_order_independent_hash(self):
        a1 = await _anchorer().anchor_json({"b": 2, "a": 1})
        a2 = await _anchorer().anchor_json({"a": 1, "b": 2})
        assert a1.content_hash == a2.content_hash

    async def test_different_objects_different_hash(self):
        a1 = await _anchorer().anchor_json({"x": 1})
        a2 = await _anchorer().anchor_json({"x": 2})
        assert a1.content_hash != a2.content_hash

    async def test_list_preserves_order(self):
        a1 = await _anchorer().anchor_json([1, 2, 3])
        a2 = await _anchorer().anchor_json([3, 2, 1])
        assert a1.content_hash != a2.content_hash


# ---------------------------------------------------------------------------
# DatasetAnchorer.anchor_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnchorText:
    async def test_returns_anchor(self):
        result = await _anchorer().anchor_text("col1,col2\n1,2\n")
        assert isinstance(result, DatasetAnchor)

    async def test_default_media_type(self):
        result = await _anchorer().anchor_text("hello")
        assert result.media_type == "text/plain"

    async def test_custom_media_type(self):
        result = await _anchorer().anchor_text("a,b", media_type="text/csv")
        assert result.media_type == "text/csv"

    async def test_content_hash_matches_utf8(self):
        text = "hello world"
        result = await _anchorer().anchor_text(text)
        assert result.content_hash == hash_bytes(text.encode("utf-8"))

    async def test_verify_roundtrip(self):
        text = "col1,col2\n1,a\n2,b\n"
        result = await _anchorer().anchor_text(text)
        assert verify_dataset_anchor(text.encode("utf-8"), result) is True


# ---------------------------------------------------------------------------
# DatasetAnchorer.anchor_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnchorFile:
    async def test_anchors_existing_file(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"col1,col2\n1,2\n")
        result = await _anchorer().anchor_file(f)
        assert result.content_hash == hash_bytes(b"col1,col2\n1,2\n")

    async def test_guesses_csv_media_type(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n")
        result = await _anchorer().anchor_file(f)
        assert result.media_type == "text/csv"

    async def test_guesses_json_media_type(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_bytes(b'{"x": 1}')
        result = await _anchorer().anchor_file(f)
        assert result.media_type == "application/json"

    async def test_missing_file_raises(self):
        with pytest.raises(ARIAError):
            await _anchorer().anchor_file("/nonexistent/path/data.csv")

    async def test_explicit_media_type_overrides_guess(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n")
        result = await _anchorer().anchor_file(f, media_type="text/plain")
        assert result.media_type == "text/plain"

    async def test_verify_roundtrip(self, tmp_path: Path):
        data = b"col1,col2\n10,20\n"
        f = tmp_path / "data.csv"
        f.write_bytes(data)
        result = await _anchorer().anchor_file(f)
        assert verify_dataset_anchor(data, result) is True


# ---------------------------------------------------------------------------
# _guess_media_type
# ---------------------------------------------------------------------------


class TestGuessMediaType:
    def test_csv(self):
        assert _guess_media_type(".csv") == "text/csv"

    def test_json(self):
        assert _guess_media_type(".json") == "application/json"

    def test_parquet(self):
        assert _guess_media_type(".parquet") == "application/vnd.apache.parquet"

    def test_unknown_extension(self):
        assert _guess_media_type(".xyz") == "application/octet-stream"

    def test_case_insensitive(self):
        assert _guess_media_type(".CSV") == "text/csv"
        assert _guess_media_type(".JSON") == "application/json"
