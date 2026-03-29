"""Tests for aria.export_bundle — Portable verification package."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest

from aria.core.record import AuditRecord
from aria.export_bundle import create_bundle, create_bundle_bytes
from aria.storage.sqlite import SQLiteStorage


def _populated_storage():
    """Create storage with sample data."""
    storage = SQLiteStorage(dsn="sqlite://")
    import time
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-001",
        system_id="test-sys",
        open_txid="tx_" + "a" * 60,
        model_hashes={"model-a": "sha256:" + "a" * 64},
        state_hash="sha256:" + "b" * 64,
        opened_at=now,
    )
    storage.save_epoch_close(
        epoch_id="ep-001",
        close_txid="tx_" + "c" * 60,
        merkle_root="sha256:" + "d" * 64,
        records_count=3,
        closed_at=now + 60,
    )

    for i in range(3):
        storage.save_record(AuditRecord(
            epoch_id="ep-001",
            model_id="model-a",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.9,
            latency_ms=100,
        ))

    return storage


class TestCreateBundle:
    def test_creates_zip_file(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                epoch_ids=["ep-001"],
                output=Path(td) / "test.zip",
            )
            assert path.exists()
            assert zipfile.is_zipfile(path)

    def test_zip_contains_expected_files(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                epoch_ids=["ep-001"],
                output=Path(td) / "test.zip",
            )
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                assert "records.json" in names
                assert "epochs.json" in names
                assert "proofs.json" in names
                assert "metadata.json" in names
                assert "verify.html" in names

    def test_records_json_has_data(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                epoch_ids=["ep-001"],
                output=Path(td) / "test.zip",
            )
            with zipfile.ZipFile(path) as zf:
                records = json.loads(zf.read("records.json"))
                assert len(records) == 3
                assert records[0]["model_id"] == "model-a"

    def test_metadata_json(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                epoch_ids=["ep-001"],
                output=Path(td) / "test.zip",
            )
            with zipfile.ZipFile(path) as zf:
                meta = json.loads(zf.read("metadata.json"))
                assert meta["epochs_count"] == 1
                assert meta["records_count"] == 3
                assert "bundle_hash" in meta

    def test_without_html(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                epoch_ids=["ep-001"],
                output=Path(td) / "test.zip",
                include_html=False,
            )
            with zipfile.ZipFile(path) as zf:
                assert "verify.html" not in zf.namelist()

    def test_all_epochs_when_none_specified(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = create_bundle(
                storage,
                output=Path(td) / "test.zip",
            )
            with zipfile.ZipFile(path) as zf:
                records = json.loads(zf.read("records.json"))
                assert len(records) == 3


class TestCreateBundleBytes:
    def test_returns_valid_zip_bytes(self):
        storage = _populated_storage()
        data = create_bundle_bytes(storage, epoch_ids=["ep-001"])
        assert isinstance(data, bytes)
        assert len(data) > 0

        import io
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "records.json" in zf.namelist()
            assert "verify.html" in zf.namelist()
