"""
aria.import_from — Import audit data from external ML platforms.

Supports importing experiment data from MLflow, Weights & Biases,
and OpenAI API logs into ARIA's audit format.

Usage::

    from aria.import_from import from_mlflow, from_wandb, from_jsonl

    # Import from MLflow
    records = from_mlflow("http://localhost:5000", experiment_id="1")

    # Import from W&B export
    records = from_wandb("wandb_export.json")

    # Import from generic JSONL logs
    records = from_jsonl("api_logs.jsonl")

    # Import from OpenAI API logs
    records = from_openai_log("openai_responses.jsonl")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core.hasher import hash_object
from .core.record import AuditRecord


def from_jsonl(
    path: str | Path,
    epoch_id: str = "imported",
    model_id: str = "unknown",
    input_field: str = "input",
    output_field: str = "output",
    confidence_field: str | None = "confidence",
    latency_field: str | None = "latency_ms",
) -> list[AuditRecord]:
    """Import records from a JSONL file.

    Each line is a JSON object. Fields are mapped to AuditRecord fields.

    Args:
        path: Path to the JSONL file.
        epoch_id: Epoch ID to assign to all records.
        model_id: Model ID to assign (overridden if 'model' field exists).
        input_field: JSON field name for input data.
        output_field: JSON field name for output data.
        confidence_field: JSON field name for confidence (None to skip).
        latency_field: JSON field name for latency in ms (None to skip).
    """
    path = Path(path)
    records: list[AuditRecord] = []

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)

        input_data = entry.get(input_field, "")
        output_data = entry.get(output_field, "")
        mid = entry.get("model", entry.get("model_id", model_id))

        conf = None
        if confidence_field and confidence_field in entry:
            conf = float(entry[confidence_field])

        lat = 0
        if latency_field and latency_field in entry:
            lat = int(entry[latency_field])

        records.append(AuditRecord(
            epoch_id=epoch_id,
            model_id=mid,
            input_hash=hash_object(input_data) if input_data else hash_object(""),
            output_hash=hash_object(output_data) if output_data else hash_object(""),
            sequence=i,
            confidence=conf,
            latency_ms=lat,
            metadata=entry.get("metadata", {}),
        ))

    return records


def from_openai_log(
    path: str | Path,
    epoch_id: str = "imported-openai",
) -> list[AuditRecord]:
    """Import from OpenAI API response logs (JSONL).

    Expected format: one JSON object per line with OpenAI API response
    structure (model, choices, usage, etc.).
    """
    path = Path(path)
    records: list[AuditRecord] = []

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)

        model = entry.get("model", "openai-unknown")

        # Extract input from messages or prompt
        messages = entry.get("messages", entry.get("input", []))
        input_hash = hash_object(messages) if messages else hash_object("")

        # Extract output from choices
        choices = entry.get("choices", [])
        output_data = choices[0].get("message", {}).get("content", "") if choices else ""
        output_hash = hash_object(output_data) if output_data else hash_object("")

        # Usage as latency proxy
        usage = entry.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        records.append(AuditRecord(
            epoch_id=epoch_id,
            model_id=model,
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=i,
            confidence=None,
            latency_ms=int(entry.get("latency_ms", 0)),
            metadata={
                "source": "openai",
                "total_tokens": total_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        ))

    return records


def from_mlflow_export(
    path: str | Path,
    epoch_id: str = "imported-mlflow",
) -> list[AuditRecord]:
    """Import from an MLflow experiment export (JSON).

    Expects the JSON structure from ``mlflow experiments export``.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    records: list[AuditRecord] = []
    runs = data if isinstance(data, list) else data.get("runs", [data])

    for i, run in enumerate(runs):
        params = run.get("params", run.get("data", {}).get("params", {}))
        metrics = run.get("metrics", run.get("data", {}).get("metrics", {}))
        tags = run.get("tags", run.get("data", {}).get("tags", {}))

        model = (
            params.get("model_name", "")
            or tags.get("mlflow.runName", "")
            or f"mlflow-run-{i}"
        )

        # Use params as input, metrics as output
        input_hash = hash_object(params) if params else hash_object("")
        output_hash = hash_object(metrics) if metrics else hash_object("")

        conf = None
        for key in ("accuracy", "f1", "score", "confidence"):
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)) and 0 <= val <= 1:
                    conf = float(val)
                    break

        records.append(AuditRecord(
            epoch_id=epoch_id,
            model_id=model,
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=i,
            confidence=conf,
            latency_ms=int(metrics.get("latency_ms", metrics.get("duration", 0))),
            metadata={"source": "mlflow", "run_id": run.get("run_id", "")},
        ))

    return records


def from_wandb_export(
    path: str | Path,
    epoch_id: str = "imported-wandb",
) -> list[AuditRecord]:
    """Import from a Weights & Biases export (JSON).

    Expects a JSON array of run objects or a W&B table export.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "data" in data:
        rows = data["data"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = [data]

    records: list[AuditRecord] = []

    for i, row in enumerate(rows):
        config = row.get("config", {})
        summary = row.get("summary", row.get("summary_metrics", {}))
        name = row.get("name", row.get("displayName", f"wandb-run-{i}"))

        model = config.get("model", config.get("model_name", name))

        input_hash = hash_object(config) if config else hash_object("")
        output_hash = hash_object(summary) if summary else hash_object("")

        conf = None
        for key in ("accuracy", "eval_accuracy", "f1", "best_score"):
            if key in summary:
                val = summary[key]
                if isinstance(val, (int, float)) and 0 <= val <= 1:
                    conf = float(val)
                    break

        records.append(AuditRecord(
            epoch_id=epoch_id,
            model_id=str(model),
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=i,
            confidence=conf,
            latency_ms=int(summary.get("_runtime", 0)),
            metadata={"source": "wandb", "run_name": name},
        ))

    return records


def save_imported(
    records: list[AuditRecord],
    storage: Any,
) -> int:
    """Convenience: save a list of imported records to storage.

    Args:
        records: AuditRecord list from any from_* function.
        storage: ARIA StorageInterface.

    Returns:
        Number of records saved.
    """
    for r in records:
        storage.save_record(r)
    return len(records)
