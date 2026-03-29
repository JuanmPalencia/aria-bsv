"""Tests for aria.backup — Backup and restore."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from pathlib import Path

import pytest

from aria.backup import backup, restore, list_backups
from aria.core.record import AuditRecord
from aria.storage.sqlite import SQLiteStorage


def _populated_storage():
    storage = SQLiteStorage(dsn="sqlite://")
    now = int(time.time())

    storage.save_epoch_open(
        epoch_id="ep-1",
        system_id="sys-1",
        open_txid="tx_" + "a" * 60,
        model_hashes={"model-a": "sha256:" + "a" * 64},
        state_hash="sha256:" + "b" * 64,
        opened_at=now,
    )
    storage.save_epoch_close(
        epoch_id="ep-1",
        close_txid="tx_" + "c" * 60,
        merkle_root="sha256:" + "d" * 64,
        records_count=5,
        closed_at=now + 100,
    )

    for i in range(5):
        storage.save_record(AuditRecord(
            epoch_id="ep-1",
            model_id="model-a",
            input_hash="sha256:" + f"{i:064x}",
            output_hash="sha256:" + f"{i + 100:064x}",
            sequence=i,
            confidence=0.85,
            latency_ms=120,
        ))

    return storage


class TestBackup:
    def test_creates_compressed_file(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td)
            assert path.exists()
            assert path.name.endswith(".json.gz")

    def test_creates_uncompressed_file(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td, compress=False)
            assert path.exists()
            assert path.name.endswith(".json")

    def test_backup_content(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td)
            with gzip.open(path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            assert data["version"] == 1
            assert data["stats"]["epochs_count"] == 1
            assert data["stats"]["records_count"] == 5
            assert len(data["epochs"]) == 1
            assert len(data["epochs"][0]["records"]) == 5


class TestRestore:
    def test_restore_full(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td)

            target = SQLiteStorage(dsn="sqlite://")
            counts = restore(path, target)
            assert counts["epochs_restored"] == 1
            assert counts["records_restored"] == 5
            assert counts["skipped"] == 0

    def test_restore_skip_existing(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td)

            # Restore once
            target = SQLiteStorage(dsn="sqlite://")
            restore(path, target)

            # Restore again (should skip)
            counts = restore(path, target, skip_existing=True)
            assert counts["skipped"] == 1
            assert counts["epochs_restored"] == 0

    def test_restore_uncompressed(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td, compress=False)
            target = SQLiteStorage(dsn="sqlite://")
            counts = restore(path, target)
            assert counts["records_restored"] == 5

    def test_restore_records_are_valid(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            path = backup(storage, output_dir=td)
            target = SQLiteStorage(dsn="sqlite://")
            restore(path, target)

            records = target.list_records_by_epoch("ep-1")
            assert len(records) == 5
            assert records[0].model_id == "model-a"
            assert records[0].confidence == 0.85


class TestListBackups:
    def test_list_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            assert list_backups(td) == []

    def test_list_with_backups(self):
        storage = _populated_storage()
        with tempfile.TemporaryDirectory() as td:
            backup(storage, output_dir=td)
            backup(storage, output_dir=td, compress=False)
            results = list_backups(td)
            assert len(results) >= 1
            assert "path" in results[0]
            assert "size_bytes" in results[0]

    def test_nonexistent_dir(self):
        assert list_backups("/nonexistent/path") == []
