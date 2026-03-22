"""Tests for aria.cli — Click command-line interface."""

from __future__ import annotations

import json
import pytest
from click.testing import CliRunner

from aria.cli import cli
from aria.storage.sqlite import SQLiteStorage
from aria.core.record import AuditRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _storage_with_data(tmp_path) -> tuple[str, SQLiteStorage]:
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(f"sqlite:///{db_path}")

    storage.save_epoch_open(
        "ep_1000000_0001", "test-system", "a" * 64,
        {"model-a": "sha256:" + "a" * 64}, "sha256:" + "0" * 64, 1742848200000,
    )
    storage.save_epoch_close(
        "ep_1000000_0001", "b" * 64, "sha256:" + "c" * 64, 3, 1742848201500,
    )

    for i in range(3):
        rec = AuditRecord(
            epoch_id="ep_1000000_0001",
            model_id="model-a",
            input_hash="sha256:" + "a" * 64,
            output_hash="sha256:" + "b" * 64,
            sequence=i,
            confidence=0.9 + i * 0.01,
            latency_ms=50,
        )
        storage.save_record(rec)

    return db_path, storage


# ---------------------------------------------------------------------------
# aria verify
# ---------------------------------------------------------------------------

class TestVerifyCommand:
    def test_verify_invalid_txid_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--open", "a" * 64, "--network", "mainnet"
        ])
        # No live network — should fail with non-zero exit
        assert result.exit_code != 0

    def test_verify_json_flag_on_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--open", "a" * 64, "--json"
        ])
        # Should exit non-zero
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# aria epochs list
# ---------------------------------------------------------------------------

class TestEpochsListCommand:
    def test_list_shows_stored_epochs(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "list", "--db", db_path])
        assert result.exit_code == 0
        assert "ep_1000000_0001" in result.output
        assert "test-system" in result.output

    def test_list_json_output(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "list", "--db", db_path, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["epoch_id"] == "ep_1000000_0001"

    def test_list_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        SQLiteStorage(f"sqlite:///{db_path}")
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "list", "--db", db_path])
        assert result.exit_code == 0
        assert "No epochs" in result.output

    def test_list_filter_by_system(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "list", "--db", db_path, "--system", "test-system"])
        assert result.exit_code == 0
        assert "ep_1000000_0001" in result.output

    def test_list_filter_nonexistent_system(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "list", "--db", db_path, "--system", "no-such-system"])
        assert result.exit_code == 0
        assert "No epochs" in result.output


# ---------------------------------------------------------------------------
# aria epochs show
# ---------------------------------------------------------------------------

class TestEpochsShowCommand:
    def test_show_existing_epoch(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "show", "ep_1000000_0001", "--db", db_path])
        assert result.exit_code == 0
        assert "test-system" in result.output
        assert "model-a" in result.output

    def test_show_nonexistent_epoch_exits_1(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        SQLiteStorage(f"sqlite:///{db_path}")
        runner = CliRunner()
        result = runner.invoke(cli, ["epochs", "show", "ep_nope", "--db", db_path])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# aria hash-file
# ---------------------------------------------------------------------------

class TestHashFileCommand:
    def test_hash_file_produces_sha256(self, tmp_path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"fake model content")
        runner = CliRunner()
        result = runner.invoke(cli, ["hash-file", str(model)])
        assert result.exit_code == 0
        assert result.output.strip().startswith("sha256:")
        assert len(result.output.strip()) == 71  # "sha256:" + 64 hex chars

    def test_hash_file_deterministic(self, tmp_path):
        model = tmp_path / "model.bin"
        model.write_bytes(b"consistent bytes")
        runner = CliRunner()
        r1 = runner.invoke(cli, ["hash-file", str(model)])
        r2 = runner.invoke(cli, ["hash-file", str(model)])
        assert r1.output == r2.output


# ---------------------------------------------------------------------------
# aria status
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def test_status_shows_stats(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--db", db_path])
        assert result.exit_code == 0
        assert "test-system" in result.output
        assert "1" in result.output  # 1 epoch

    def test_status_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        SQLiteStorage(f"sqlite:///{db_path}")
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--db", db_path])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# aria report
# ---------------------------------------------------------------------------

class TestReportCommand:
    def test_report_text_output(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "ep_1000000_0001", "--db", db_path])
        assert result.exit_code == 0
        assert "ARIA EPOCH COMPLIANCE REPORT" in result.output
        assert "test-system" in result.output

    def test_report_json_output(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "ep_1000000_0001", "--db", db_path, "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["epoch_id"] == "ep_1000000_0001"
        assert data["system_id"] == "test-system"

    def test_report_to_file(self, tmp_path):
        db_path, _ = _storage_with_data(tmp_path)
        out = str(tmp_path / "report.txt")
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "ep_1000000_0001", "--db", db_path, "--output", out])
        assert result.exit_code == 0
        with open(out) as f:
            content = f.read()
        assert "test-system" in content

    def test_report_nonexistent_epoch_exits_1(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        SQLiteStorage(f"sqlite:///{db_path}")
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "ep_nope", "--db", db_path])
        assert result.exit_code == 1
