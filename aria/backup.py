"""
aria.backup — Backup and restore ARIA audit databases.

Creates versioned backups of the local audit store and supports
full restore for disaster recovery or migration.

Usage::

    from aria.backup import backup, restore, list_backups

    # Create a backup
    path = backup(storage, output_dir="./backups")

    # List available backups
    for b in list_backups("./backups"):
        print(b)

    # Restore from backup
    restore("./backups/aria_backup_20240101_120000.json.gz", storage)
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface


def backup(
    storage: "StorageInterface",
    output_dir: str | Path = ".",
    system_id: str | None = None,
    compress: bool = True,
) -> Path:
    """Create a full backup of all epochs and records.

    Args:
        storage: ARIA StorageInterface to backup from.
        output_dir: Directory to write the backup file.
        system_id: Filter to a specific system (None = all).
        compress: Whether to gzip the output.

    Returns:
        Path to the backup file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = storage.list_epochs(system_id=system_id, limit=100_000)

    data: dict[str, Any] = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aria_version": _get_version(),
        "epochs": [],
    }

    for epoch in epochs:
        records = storage.list_records_by_epoch(epoch.epoch_id)
        epoch_data = _to_dict(epoch)
        epoch_data["records"] = [_to_dict(r) for r in records]
        data["epochs"].append(epoch_data)

    data["stats"] = {
        "epochs_count": len(data["epochs"]),
        "records_count": sum(
            len(e["records"]) for e in data["epochs"]
        ),
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    ext = ".json.gz" if compress else ".json"
    filename = f"aria_backup_{timestamp}{ext}"
    filepath = output_dir / filename

    payload = json.dumps(data, indent=2, default=str).encode("utf-8")

    if compress:
        with gzip.open(filepath, "wb") as f:
            f.write(payload)
    else:
        filepath.write_bytes(payload)

    return filepath


def restore(
    backup_path: str | Path,
    storage: "StorageInterface",
    *,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Restore epochs and records from a backup file.

    Args:
        backup_path: Path to the backup file (.json or .json.gz).
        storage: StorageInterface to restore into.
        skip_existing: If True, skip epochs that already exist.

    Returns:
        Dict with counts: {"epochs_restored", "records_restored", "skipped"}.
    """
    from .core.record import AuditRecord

    backup_path = Path(backup_path)

    if backup_path.suffix == ".gz" or backup_path.name.endswith(".json.gz"):
        with gzip.open(backup_path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
    else:
        data = json.loads(backup_path.read_text(encoding="utf-8"))

    counts = {"epochs_restored": 0, "records_restored": 0, "skipped": 0}

    for epoch_data in data.get("epochs", []):
        epoch_id = epoch_data["epoch_id"]

        if skip_existing:
            existing = storage.get_epoch(epoch_id)
            if existing is not None:
                counts["skipped"] += 1
                continue

        # Restore epoch open
        storage.save_epoch_open(
            epoch_id=epoch_id,
            system_id=epoch_data.get("system_id", ""),
            open_txid=epoch_data.get("open_txid", ""),
            model_hashes=epoch_data.get("model_hashes", {}),
            state_hash=epoch_data.get("state_hash", ""),
            opened_at=epoch_data.get("opened_at", 0),
        )

        # Restore epoch close if it was closed
        if epoch_data.get("close_txid"):
            storage.save_epoch_close(
                epoch_id=epoch_id,
                close_txid=epoch_data["close_txid"],
                merkle_root=epoch_data.get("merkle_root", ""),
                records_count=epoch_data.get("records_count", 0),
                closed_at=epoch_data.get("closed_at", 0),
            )

        counts["epochs_restored"] += 1

        # Restore records
        for rec_data in epoch_data.get("records", []):
            record = AuditRecord(
                epoch_id=rec_data["epoch_id"],
                model_id=rec_data["model_id"],
                input_hash=rec_data["input_hash"],
                output_hash=rec_data["output_hash"],
                sequence=rec_data["sequence"],
                confidence=rec_data.get("confidence"),
                latency_ms=rec_data.get("latency_ms", 0),
                metadata=rec_data.get("metadata", {}),
            )
            storage.save_record(record)
            counts["records_restored"] += 1

    return counts


def list_backups(directory: str | Path = ".") -> list[dict[str, Any]]:
    """List available backup files in a directory.

    Returns:
        List of dicts with 'path', 'size_bytes', 'created_at' for each backup.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    backups = []
    for f in sorted(directory.glob("aria_backup_*")):
        if f.suffix in (".json", ".gz"):
            backups.append({
                "path": str(f),
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "modified": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(f.stat().st_mtime),
                ),
            })

    return backups


def _to_dict(obj: Any) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
