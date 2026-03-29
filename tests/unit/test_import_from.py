"""Tests for aria.import_from — Import data from external platforms."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aria.core.record import AuditRecord
from aria.import_from import (
    from_jsonl,
    from_openai_log,
    from_mlflow_export,
    from_wandb_export,
    save_imported,
)
from aria.storage.sqlite import SQLiteStorage


def _write_temp(content: str, suffix: str = ".jsonl") -> Path:
    """Write content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


class TestFromJsonl:
    def test_basic_import(self):
        data = "\n".join([
            json.dumps({"input": "hello", "output": "world"}),
            json.dumps({"input": "foo", "output": "bar"}),
        ])
        path = _write_temp(data)
        records = from_jsonl(path, epoch_id="test-ep", model_id="test-model")
        assert len(records) == 2
        assert all(isinstance(r, AuditRecord) for r in records)
        assert records[0].epoch_id == "test-ep"
        assert records[0].model_id == "test-model"
        assert records[0].sequence == 0
        assert records[1].sequence == 1

    def test_hashes_are_valid(self):
        data = json.dumps({"input": "test", "output": "result"})
        path = _write_temp(data)
        records = from_jsonl(path)
        assert records[0].input_hash.startswith("sha256:")
        assert len(records[0].input_hash) == 71

    def test_with_confidence_and_latency(self):
        data = json.dumps({
            "input": "x",
            "output": "y",
            "confidence": 0.95,
            "latency_ms": 150,
        })
        path = _write_temp(data)
        records = from_jsonl(path)
        assert records[0].confidence == 0.95
        assert records[0].latency_ms == 150

    def test_custom_field_names(self):
        data = json.dumps({"prompt": "hi", "response": "hello"})
        path = _write_temp(data)
        records = from_jsonl(
            path, input_field="prompt", output_field="response"
        )
        assert len(records) == 1
        assert records[0].input_hash.startswith("sha256:")

    def test_model_from_data(self):
        data = json.dumps({"input": "x", "output": "y", "model": "gpt-4"})
        path = _write_temp(data)
        records = from_jsonl(path)
        assert records[0].model_id == "gpt-4"

    def test_empty_lines_skipped(self):
        data = json.dumps({"input": "x", "output": "y"}) + "\n\n\n"
        path = _write_temp(data)
        records = from_jsonl(path)
        assert len(records) == 1


class TestFromOpenaiLog:
    def test_basic_openai_import(self):
        data = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "choices": [{"message": {"content": "hi there"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "latency_ms": 200,
        })
        path = _write_temp(data)
        records = from_openai_log(path)
        assert len(records) == 1
        assert records[0].model_id == "gpt-4"
        assert records[0].latency_ms == 200
        assert records[0].metadata["source"] == "openai"
        assert records[0].metadata["total_tokens"] == 8

    def test_multiple_entries(self):
        entries = [
            json.dumps({"model": "gpt-4", "choices": [{"message": {"content": "a"}}], "usage": {}}),
            json.dumps({"model": "gpt-3.5", "choices": [{"message": {"content": "b"}}], "usage": {}}),
        ]
        path = _write_temp("\n".join(entries))
        records = from_openai_log(path)
        assert len(records) == 2
        assert records[0].model_id == "gpt-4"
        assert records[1].model_id == "gpt-3.5"


class TestFromMlflowExport:
    def test_basic_mlflow_import(self):
        data = json.dumps({
            "runs": [
                {
                    "run_id": "abc123",
                    "params": {"model_name": "bert-base", "lr": "0.001"},
                    "metrics": {"accuracy": 0.92, "latency_ms": 45},
                    "tags": {},
                },
            ]
        })
        path = _write_temp(data, suffix=".json")
        records = from_mlflow_export(path)
        assert len(records) == 1
        assert records[0].model_id == "bert-base"
        assert records[0].confidence == 0.92
        assert records[0].metadata["source"] == "mlflow"

    def test_list_format(self):
        data = json.dumps([
            {"params": {"model_name": "m1"}, "metrics": {"f1": 0.85}, "run_id": "r1"},
        ])
        path = _write_temp(data, suffix=".json")
        records = from_mlflow_export(path)
        assert len(records) == 1
        assert records[0].confidence == 0.85


class TestFromWandbExport:
    def test_basic_wandb_import(self):
        data = json.dumps([
            {
                "name": "run-1",
                "config": {"model": "resnet50", "epochs": 10},
                "summary": {"accuracy": 0.94, "_runtime": 3600},
            },
        ])
        path = _write_temp(data, suffix=".json")
        records = from_wandb_export(path)
        assert len(records) == 1
        assert records[0].model_id == "resnet50"
        assert records[0].confidence == 0.94
        assert records[0].metadata["source"] == "wandb"

    def test_table_format(self):
        data = json.dumps({
            "data": [
                {"name": "r1", "config": {"model_name": "bert"}, "summary": {}},
            ]
        })
        path = _write_temp(data, suffix=".json")
        records = from_wandb_export(path)
        assert len(records) == 1


class TestSaveImported:
    def test_save_to_storage(self):
        data = "\n".join([
            json.dumps({"input": "a", "output": "b"}),
            json.dumps({"input": "c", "output": "d"}),
        ])
        path = _write_temp(data)
        records = from_jsonl(path, epoch_id="imp-ep")
        storage = SQLiteStorage(dsn="sqlite://")

        storage.save_epoch_open(
            epoch_id="imp-ep",
            system_id="sys",
            open_txid="tx_" + "z" * 60,
            model_hashes={},
            state_hash="sha256:" + "0" * 64,
            opened_at=0,
        )

        count = save_imported(records, storage)
        assert count == 2
        stored = storage.list_records_by_epoch("imp-ep")
        assert len(stored) == 2
