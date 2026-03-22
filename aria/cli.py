"""
aria.cli — Command-line interface for the ARIA SDK.

Usage:
    aria verify --open <txid> [--close <txid>] [--network mainnet|testnet]
    aria epochs list --system <id> [--limit 20]
    aria epochs show <epoch_id>
    aria record --model <id> --input '{"q":"..."}' --output '{"a":"..."}'
    aria status
    aria report --epoch <epoch_id> [--format text|json]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

try:
    import click
except ImportError:
    raise ImportError("CLI requires click: pip install aria-bsv[cli]")

from aria.core.hasher import hash_file
from aria.storage.sqlite import SQLiteStorage
from aria.verify import Verifier, WhatsOnChainFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_storage(db_path: str) -> SQLiteStorage:
    dsn = f"sqlite:///{db_path}" if not db_path.startswith("sqlite") else db_path
    return SQLiteStorage(dsn)


def _run(coro):
    """Run a coroutine from sync context."""
    return asyncio.run(coro)


def _ok(msg: str) -> None:
    click.echo(click.style(f"✓ {msg}", fg="green"))


def _err(msg: str) -> None:
    click.echo(click.style(f"✗ {msg}", fg="red"), err=True)


def _info(msg: str) -> None:
    click.echo(click.style(msg, fg="cyan"))


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="aria-bsv")
def cli() -> None:
    """ARIA — Auditable Real-time Inference Architecture.

    Cryptographic accountability for production AI systems on BSV.
    """


# ---------------------------------------------------------------------------
# aria verify
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--open", "open_txid", required=True, help="EPOCH_OPEN transaction ID")
@click.option("--close", "close_txid", default=None, help="EPOCH_CLOSE transaction ID (optional)")
@click.option("--network", default="mainnet", type=click.Choice(["mainnet", "testnet"]), show_default=True)
@click.option("--db", default="aria.db", show_default=True, help="Path to local ARIA SQLite database")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def verify(open_txid: str, close_txid: Optional[str], network: str, db: str, as_json: bool) -> None:
    """Verify an ARIA epoch from BSV transactions."""
    storage = _get_storage(db)
    fetcher = WhatsOnChainFetcher(network=network)
    verifier = Verifier(fetcher=fetcher, storage=storage)

    async def _verify():
        return await verifier.verify_epoch(open_txid=open_txid, close_txid=close_txid)

    try:
        result = _run(_verify())
    except Exception as exc:
        _err(f"Verification failed: {exc}")
        sys.exit(1)

    if as_json:
        data = {
            "valid": result.valid,
            "tampered": result.tampered,
            "epoch_id": result.epoch_id,
            "system_id": result.system_id,
            "records_count": result.records_count,
            "merkle_root": result.merkle_root,
            "error": result.error,
        }
        click.echo(json.dumps(data, indent=2))
        return

    if result.valid:
        _ok(f"Epoch verified — {result.system_id}")
        click.echo(f"  Epoch ID    : {result.epoch_id}")
        click.echo(f"  Records     : {result.records_count}")
        click.echo(f"  Merkle root : {result.merkle_root}")
        if result.decided_at:
            click.echo(f"  Opened at   : {result.decided_at.isoformat()}")
    else:
        if result.tampered:
            _err(f"TAMPERED — {result.error}")
        else:
            _err(f"Invalid — {result.error}")
        sys.exit(2)


# ---------------------------------------------------------------------------
# aria epochs
# ---------------------------------------------------------------------------

@cli.group()
def epochs() -> None:
    """Inspect and query stored epochs."""


@epochs.command("list")
@click.option("--system", "system_id", default=None, help="Filter by system_id")
@click.option("--limit", default=20, show_default=True)
@click.option("--db", default="aria.db", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def epochs_list(system_id: Optional[str], limit: int, db: str, as_json: bool) -> None:
    """List epochs stored in the local database."""
    storage = _get_storage(db)
    rows = storage.list_epochs(system_id=system_id, limit=limit)

    if not rows:
        _info("No epochs found.")
        return

    if as_json:
        click.echo(json.dumps([
            {
                "epoch_id": r.epoch_id,
                "system_id": r.system_id,
                "open_txid": r.open_txid,
                "close_txid": r.close_txid,
                "records_count": r.records_count,
                "opened_at": r.opened_at,
            }
            for r in rows
        ], indent=2))
        return

    click.echo(f"{'EPOCH ID':<35} {'SYSTEM':<20} {'RECORDS':>7}  {'OPEN TXID':<16}")
    click.echo("─" * 90)
    for r in rows:
        txid_short = (r.open_txid or "")[:16] + "…" if r.open_txid else "pending"
        click.echo(f"{r.epoch_id:<35} {r.system_id:<20} {(r.records_count or 0):>7}  {txid_short}")


@epochs.command("show")
@click.argument("epoch_id")
@click.option("--db", default="aria.db", show_default=True)
def epochs_show(epoch_id: str, db: str) -> None:
    """Show details for a specific epoch."""
    storage = _get_storage(db)
    row = storage.get_epoch(epoch_id)
    if row is None:
        _err(f"Epoch '{epoch_id}' not found in {db}")
        sys.exit(1)

    click.echo(f"Epoch       : {row.epoch_id}")
    click.echo(f"System      : {row.system_id}")
    click.echo(f"Records     : {row.records_count or 0}")
    click.echo(f"Open txid   : {row.open_txid or 'pending'}")
    click.echo(f"Close txid  : {row.close_txid or 'pending'}")
    click.echo(f"Merkle root : {row.merkle_root or 'not yet computed'}")
    if row.opened_at:
        click.echo(f"Opened at   : {row.opened_at}")

    records = storage.list_records_by_epoch(epoch_id)
    if records:
        click.echo(f"\nRecords ({len(records)}):")
        for rec in records[:10]:
            click.echo(f"  [{rec.sequence:>5}] {rec.record_id}  model={rec.model_id}  conf={rec.confidence}")
        if len(records) > 10:
            click.echo(f"  ... and {len(records) - 10} more")


# ---------------------------------------------------------------------------
# aria record
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "model_id", required=True, help="Model ID")
@click.option("--input", "input_data", required=True, help="Input JSON string or @file path")
@click.option("--output", "output_data", required=True, help="Output JSON string or @file path")
@click.option("--confidence", default=None, type=float)
@click.option("--latency-ms", default=0, type=int)
@click.option("--system", "system_id", envvar="ARIA_SYSTEM_ID", required=True)
@click.option("--key", "bsv_key", envvar="ARIA_BSV_KEY", required=True, help="BSV WIF key")
@click.option("--model-hash", "model_hash", envvar="ARIA_MODEL_HASH", default=None)
@click.option("--db", default="aria.db", show_default=True)
def record(
    model_id: str,
    input_data: str,
    output_data: str,
    confidence: Optional[float],
    latency_ms: int,
    system_id: str,
    bsv_key: str,
    model_hash: Optional[str],
    db: str,
) -> None:
    """Record a single inference to the audit log."""
    from aria.auditor import AuditConfig, InferenceAuditor

    def _load(s: str):
        if s.startswith("@"):
            with open(s[1:]) as f:
                return json.load(f)
        return json.loads(s)

    try:
        inp = _load(input_data)
        out = _load(output_data)
    except (json.JSONDecodeError, OSError) as exc:
        _err(f"JSON parse error: {exc}")
        sys.exit(1)

    mhash = model_hash or f"sha256:{'0' * 64}"
    storage = _get_storage(db)
    config = AuditConfig(system_id=system_id, bsv_key=bsv_key, batch_ms=30000, batch_size=1000)

    auditor = InferenceAuditor(config, model_hashes={model_id: mhash}, _storage=storage)
    try:
        rec_id = auditor.record(model_id, inp, out, confidence=confidence, latency_ms=latency_ms)
        auditor.flush()
        auditor.close()
        _ok(f"Recorded: {rec_id}")
    except Exception as exc:
        _err(f"Record failed: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# aria hash-file
# ---------------------------------------------------------------------------

@cli.command("hash-file")
@click.argument("path", type=click.Path(exists=True))
def hash_file_cmd(path: str) -> None:
    """Compute the ARIA SHA-256 hash of a model file."""
    h = hash_file(path)
    click.echo(h)


# ---------------------------------------------------------------------------
# aria status
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
def status(db: str) -> None:
    """Show ARIA database statistics."""
    storage = _get_storage(db)
    epochs_all = storage.list_epochs(limit=10000)
    total_epochs = len(epochs_all)

    open_epochs = sum(1 for e in epochs_all if e.close_txid is None)
    closed_epochs = total_epochs - open_epochs

    systems = {e.system_id for e in epochs_all}

    click.echo(f"Database    : {db}")
    click.echo(f"Systems     : {len(systems)}")
    click.echo(f"Total epochs: {total_epochs}  (open: {open_epochs}, closed: {closed_epochs})")

    if systems:
        click.echo("\nSystems:")
        for sid in sorted(systems):
            sys_epochs = [e for e in epochs_all if e.system_id == sid]
            total_records = sum(e.records_count or 0 for e in sys_epochs)
            click.echo(f"  {sid:<30} {len(sys_epochs):>5} epochs  {total_records:>8} records")


# ---------------------------------------------------------------------------
# aria report
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("epoch_id")
@click.option("--db", default="aria.db", show_default=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]))
@click.option("--output", "out_path", default=None, help="Write to file instead of stdout")
def report(epoch_id: str, db: str, fmt: str, out_path: Optional[str]) -> None:
    """Generate a compliance report for an epoch."""
    storage = _get_storage(db)
    row = storage.get_epoch(epoch_id)
    if row is None:
        _err(f"Epoch '{epoch_id}' not found")
        sys.exit(1)

    records = storage.list_records_by_epoch(epoch_id)

    if fmt == "json":
        data = {
            "epoch_id": row.epoch_id,
            "system_id": row.system_id,
            "records_count": row.records_count or len(records),
            "open_txid": row.open_txid,
            "close_txid": row.close_txid,
            "merkle_root": row.merkle_root,
            "opened_at": row.opened_at,
            "models": list({r.model_id for r in records}),
        }
        content = json.dumps(data, indent=2)
    else:
        lines = [
            "ARIA EPOCH COMPLIANCE REPORT",
            "=" * 50,
            f"Epoch ID  : {row.epoch_id}",
            f"System    : {row.system_id}",
            f"Records   : {row.records_count or len(records)}",
            f"Open txid : {row.open_txid or 'pending'}",
            f"Close txid: {row.close_txid or 'pending'}",
            f"Merkle root: {row.merkle_root or 'N/A'}",
        ]
        if row.opened_at:
            lines.append(f"Opened at : {row.opened_at}")
        if records:
            models = list({r.model_id for r in records})
            lines.append(f"Models    : {', '.join(models)}")
            confs = [r.confidence for r in records if r.confidence is not None]
            if confs:
                lines.append(f"Confidence: min={min(confs):.3f}  avg={sum(confs)/len(confs):.3f}  max={max(confs):.3f}")
        content = "\n".join(lines)

    if out_path:
        with open(out_path, "w") as f:
            f.write(content)
        _ok(f"Report written to {out_path}")
    else:
        click.echo(content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
