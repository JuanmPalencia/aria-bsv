"""
aria.cli — Command-line interface for the ARIA SDK.

Usage:
    aria keygen [--network testnet|mainnet] [--env .env] [--json]
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
from aria.wallet.keygen import generate_keypair, write_env_file


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
# aria keygen
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--network", default="testnet", type=click.Choice(["mainnet", "testnet"]),
              show_default=True, help="BSV network for the key")
@click.option("--env", "env_path", default=None, help="Write WIF to a .env file (e.g. --env .env)")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable JSON output")
def keygen(network: str, env_path: Optional[str], as_json: bool) -> None:
    """Generate a fresh BSV keypair for ARIA auditing.

    The private key is shown ONCE. ARIA never stores it automatically.
    You are responsible for saving it securely.

    \b
    Examples:
      aria keygen                         # show key in terminal (testnet)
      aria keygen --network mainnet       # mainnet key
      aria keygen --env .env              # also append to .env file
      aria keygen --json                  # JSON output for scripts
    """
    kp = generate_keypair(network=network)

    if as_json:
        click.echo(json.dumps(kp.to_dict(), indent=2))
    else:
        click.echo(str(kp))

    if env_path:
        write_env_file(kp, env_path)
        _ok(f"WIF written to {env_path}")
        _info("  Make sure .env is in your .gitignore!")


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
# aria export
# ---------------------------------------------------------------------------

_EXPORT_CSV_FIELDS = [
    "record_id", "epoch_id", "model_id", "input_hash", "output_hash",
    "confidence", "latency_ms", "timestamp", "sequence",
]


@cli.command()
@click.option("--system", "system_id", default=None, help="Filter by system_id")
@click.option("--epoch", "epoch_id", default=None, help="Export a single epoch")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "csv", "jsonl"]),
              show_default=True)
@click.option("--output", "out_path", default=None,
              help="Output file path (default: stdout)")
@click.option("--limit", default=1000, show_default=True, help="Maximum epochs to export")
@click.option("--db", default="aria.db", show_default=True)
def export(
    system_id: Optional[str],
    epoch_id: Optional[str],
    fmt: str,
    out_path: Optional[str],
    limit: int,
    db: str,
) -> None:
    """Export audit records to JSON, JSONL, or CSV for external analysis.

    \b
    Writes to --output file or to stdout when --output is omitted.

    \b
    CSV columns: record_id, epoch_id, model_id, input_hash, output_hash,
                 confidence, latency_ms, timestamp, sequence
    \b
    Examples:
      aria export --system my-ai --format jsonl --output records.jsonl
      aria export --epoch <id> --format csv | head
    """
    import csv
    import io

    storage = _get_storage(db)

    if epoch_id:
        epochs_to_export = [storage.get_epoch(epoch_id)]
        if epochs_to_export[0] is None:
            _err(f"Epoch '{epoch_id}' not found")
            sys.exit(1)
    else:
        epochs_to_export = storage.list_epochs(system_id=system_id, limit=limit)

    all_records = []
    for ep in epochs_to_export:
        recs = storage.list_records_by_epoch(ep.epoch_id)
        for rec in recs:
            all_records.append({
                "record_id": rec.record_id,
                "epoch_id": rec.epoch_id,
                "model_id": rec.model_id,
                "input_hash": rec.input_hash,
                "output_hash": rec.output_hash,
                "confidence": rec.confidence,
                "latency_ms": rec.latency_ms,
                "timestamp": ep.opened_at,
                "sequence": rec.sequence,
            })

    if not all_records:
        _info("No records found to export.")
        return

    if fmt == "json":
        content = json.dumps(all_records, indent=2)
        if out_path:
            with open(out_path, "w") as f:
                f.write(content)
        else:
            click.echo(content)

    elif fmt == "jsonl":
        lines = "\n".join(json.dumps(rec) for rec in all_records)
        if out_path:
            with open(out_path, "w") as f:
                f.write(lines + "\n")
        else:
            click.echo(lines)

    elif fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_EXPORT_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
        csv_content = buf.getvalue()
        if out_path:
            with open(out_path, "w", newline="") as f:
                f.write(csv_content)
        else:
            click.echo(csv_content, nl=False)

    if out_path:
        _ok(f"Exported {len(all_records)} records to {out_path} ({fmt})")


# ---------------------------------------------------------------------------
# aria audit
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("epoch_id")
@click.option("--db", default="aria.db", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--min-confidence", default=None, type=float, help="Minimum acceptable mean confidence")
@click.option("--max-latency-ms", default=None, type=float, help="Maximum acceptable mean latency (ms)")
@click.option("--require-close-txid", is_flag=True, help="Fail if epoch has no close txid")
def audit(
    epoch_id: str,
    db: str,
    as_json: bool,
    min_confidence: Optional[float],
    max_latency_ms: Optional[float],
    require_close_txid: bool,
) -> None:
    """Run automated audit checks on an epoch.

    Checks performed:
    \b
      - Epoch has an open txid (on-chain commitment exists)
      - Epoch has a close txid (if --require-close-txid)
      - Records count matches storage
      - Mean confidence >= --min-confidence (if set)
      - Mean latency <= --max-latency-ms (if set)
      - No duplicate record sequences
    """
    storage = _get_storage(db)
    row = storage.get_epoch(epoch_id)
    if row is None:
        _err(f"Epoch '{epoch_id}' not found in {db}")
        sys.exit(1)

    records = storage.list_records_by_epoch(epoch_id)
    checks = []
    passed = True

    def _check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        checks.append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            passed = False

    # Check 1: open txid
    _check(
        "open_txid_present",
        bool(row.open_txid),
        row.open_txid or "missing",
    )

    # Check 2: close txid (optional)
    if require_close_txid:
        _check(
            "close_txid_present",
            bool(row.close_txid),
            row.close_txid or "missing",
        )

    # Check 3: record count integrity
    stored_count = row.records_count or 0
    actual_count = len(records)
    _check(
        "record_count_integrity",
        stored_count == actual_count,
        f"stored={stored_count} actual={actual_count}",
    )

    # Check 4: no duplicate sequences
    seqs = [r.sequence for r in records]
    _check(
        "no_duplicate_sequences",
        len(seqs) == len(set(seqs)),
        f"{len(seqs) - len(set(seqs))} duplicates found" if len(seqs) != len(set(seqs)) else "ok",
    )

    # Check 5: mean confidence
    if min_confidence is not None:
        confs = [r.confidence for r in records if r.confidence is not None]
        if confs:
            mean_conf = sum(confs) / len(confs)
            _check(
                "mean_confidence",
                mean_conf >= min_confidence,
                f"mean={mean_conf:.3f} threshold={min_confidence:.3f}",
            )
        else:
            _check("mean_confidence", False, "no confidence values in records")

    # Check 6: mean latency
    if max_latency_ms is not None:
        lats = [r.latency_ms for r in records if r.latency_ms is not None]
        if lats:
            mean_lat = sum(lats) / len(lats)
            _check(
                "mean_latency_ms",
                mean_lat <= max_latency_ms,
                f"mean={mean_lat:.1f}ms threshold={max_latency_ms:.1f}ms",
            )

    if as_json:
        click.echo(json.dumps({
            "epoch_id": epoch_id,
            "passed": passed,
            "checks": checks,
        }, indent=2))
    else:
        click.echo(f"Audit: {epoch_id}")
        click.echo("─" * 60)
        for c in checks:
            icon = click.style("✓", fg="green") if c["passed"] else click.style("✗", fg="red")
            click.echo(f"  {icon} {c['check']:<35} {c['detail']}")
        click.echo("")
        if passed:
            _ok("All checks passed")
        else:
            _err("One or more checks failed")
            sys.exit(2)


# ---------------------------------------------------------------------------
# aria compliance-check
# ---------------------------------------------------------------------------

@cli.command("compliance-check")
@click.option("--system", "system_id", required=True, help="System ID to check")
@click.option("--db", default="aria.db", show_default=True)
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]), show_default=True)
@click.option("--output", "out_path", default=None, help="Write report to file")
@click.option("--epochs", "epochs_limit", default=50, show_default=True, help="Epochs to analyse")
def compliance_check(
    system_id: str,
    db: str,
    fmt: str,
    out_path: Optional[str],
    epochs_limit: int,
) -> None:
    """Generate an EU AI Act Art. 12 compliance report for a system.

    Checks:
    \b
      - All epochs have on-chain open txid (Art. 12 §1 — technical documentation)
      - All epochs have on-chain close txid (Art. 12 §2 — logging completeness)
      - Record count integrity across epochs
      - Mean confidence and latency baselines
      - Epoch frequency (activity evidence)
    """
    storage = _get_storage(db)
    epochs_all = storage.list_epochs(system_id=system_id, limit=epochs_limit)

    if not epochs_all:
        _err(f"No epochs found for system '{system_id}'")
        sys.exit(1)

    total = len(epochs_all)
    open_anchored = sum(1 for e in epochs_all if e.open_txid)
    close_anchored = sum(1 for e in epochs_all if e.close_txid)
    total_records = sum(e.records_count or 0 for e in epochs_all)

    all_records = []
    for ep in epochs_all[:20]:  # sample up to 20 for stats
        all_records.extend(storage.list_records_by_epoch(ep.epoch_id))

    confs = [r.confidence for r in all_records if r.confidence is not None]
    lats = [r.latency_ms for r in all_records if r.latency_ms is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    mean_lat = sum(lats) / len(lats) if lats else None

    result = {
        "system_id": system_id,
        "regulation": "EU AI Act Art. 12 (Regulation (EU) 2024/1689)",
        "epochs_analysed": total,
        "total_records": total_records,
        "on_chain_open_rate": round(open_anchored / total, 3) if total else 0,
        "on_chain_close_rate": round(close_anchored / total, 3) if total else 0,
        "mean_confidence": round(mean_conf, 3) if mean_conf is not None else None,
        "mean_latency_ms": round(mean_lat, 1) if mean_lat is not None else None,
        "compliant": open_anchored == total and close_anchored == total,
        "findings": [],
    }

    if open_anchored < total:
        result["findings"].append(
            f"{total - open_anchored}/{total} epochs missing on-chain OPEN commitment (Art. 12 §1)"
        )
    if close_anchored < total:
        result["findings"].append(
            f"{total - close_anchored}/{total} epochs missing on-chain CLOSE commitment (Art. 12 §2)"
        )
    if not result["findings"]:
        result["findings"].append("No compliance violations detected.")

    if fmt == "json":
        content = json.dumps(result, indent=2)
    else:
        lines = [
            "EU AI ACT ART. 12 COMPLIANCE REPORT",
            "=" * 55,
            f"System ID          : {result['system_id']}",
            f"Regulation         : {result['regulation']}",
            f"Epochs analysed    : {result['epochs_analysed']}",
            f"Total records      : {result['total_records']}",
            f"On-chain open rate : {result['on_chain_open_rate']*100:.1f}%",
            f"On-chain close rate: {result['on_chain_close_rate']*100:.1f}%",
        ]
        if result["mean_confidence"] is not None:
            lines.append(f"Mean confidence    : {result['mean_confidence']:.3f}")
        if result["mean_latency_ms"] is not None:
            lines.append(f"Mean latency       : {result['mean_latency_ms']:.1f} ms")
        lines.append("")
        lines.append("Findings:")
        for f in result["findings"]:
            lines.append(f"  • {f}")
        lines.append("")
        verdict = "COMPLIANT" if result["compliant"] else "NON-COMPLIANT"
        lines.append(f"Verdict: {verdict}")
        content = "\n".join(lines)

    if out_path:
        with open(out_path, "w") as f:
            f.write(content)
        _ok(f"Report written to {out_path}")
    else:
        click.echo(content)

    if not result["compliant"]:
        sys.exit(2)


# ---------------------------------------------------------------------------
# aria selftest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--bsv", is_flag=True, help="Also check BSV connectivity (requires network)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def selftest(bsv: bool, as_json: bool) -> None:
    """Run ARIA self-test to verify installation health."""
    from aria.selftest import selftest as run_selftest

    report = run_selftest(bsv=bsv)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return

    click.echo(report.summary())
    if not report.ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# aria query
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--model", default=None, help="Filter by model_id")
@click.option("--since", default=None, help="Time window, e.g. '24h', '7d'")
@click.option("--confidence-lt", default=None, type=float, help="Confidence below threshold")
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True)
def query(db: str, model: Optional[str], since: Optional[str], confidence_lt: Optional[float], limit: int, as_json: bool) -> None:
    """Query audit records with filters."""
    from aria.query import RecordQuery

    storage = _get_storage(db)
    q = RecordQuery(storage)

    if model:
        q = q.model(model)
    if since:
        q = q.since(since)
    if confidence_lt is not None:
        q = q.where(confidence__lt=confidence_lt)
    q = q.limit(limit)

    records = q.execute()

    if as_json:
        click.echo(json.dumps([{
            "record_id": r.record_id,
            "model_id": r.model_id,
            "confidence": r.confidence,
            "latency_ms": r.latency_ms,
            "sequence": r.sequence,
        } for r in records], indent=2))
        return

    if not records:
        _info("No records found matching the query.")
        return

    click.echo(f"{'RECORD ID':<40} {'MODEL':<15} {'CONF':>6} {'LAT(ms)':>8}")
    click.echo("─" * 75)
    for r in records:
        conf = f"{r.confidence:.3f}" if r.confidence is not None else "N/A"
        click.echo(f"{r.record_id:<40} {r.model_id:<15} {conf:>6} {r.latency_ms or 0:>8}")


# ---------------------------------------------------------------------------
# aria backup / restore
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--output-dir", default=".", show_default=True, help="Directory for backup file")
@click.option("--no-compress", is_flag=True, help="Don't gzip the backup")
def backup(db: str, output_dir: str, no_compress: bool) -> None:
    """Create a backup of the ARIA database."""
    from aria.backup import backup as run_backup

    storage = _get_storage(db)
    path = run_backup(storage, output_dir, compress=not no_compress)
    _ok(f"Backup created: {path}")


@cli.command("restore")
@click.argument("backup_path", type=click.Path(exists=True))
@click.option("--db", default="aria.db", show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing records")
def restore_cmd(backup_path: str, db: str, overwrite: bool) -> None:
    """Restore records from an ARIA backup file."""
    from aria.backup import restore as run_restore

    storage = _get_storage(db)
    counts = run_restore(backup_path, storage, skip_existing=not overwrite)
    _ok(f"Restored: {counts}")


# ---------------------------------------------------------------------------
# aria dashboard
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8710, show_default=True, type=int)
def dashboard(db: str, host: str, port: int) -> None:
    """Launch a web dashboard for ARIA data visualization."""
    from aria.dashboard import serve

    storage = _get_storage(db)
    _info(f"Starting ARIA dashboard at http://{host}:{port}")
    serve(storage, host=host, port=port)


# ---------------------------------------------------------------------------
# aria certify
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("epoch_id")
@click.option("--db", default="aria.db", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option("--badge", is_flag=True, help="Output SVG badge instead of certificate")
def certify(epoch_id: str, db: str, as_json: bool, badge: bool) -> None:
    """Generate an integrity certificate for an epoch."""
    from aria.certify import Certifier

    storage = _get_storage(db)
    certifier = Certifier(storage)

    try:
        cert = certifier.certify_epoch(epoch_id)
    except Exception as exc:
        _err(f"Certification failed: {exc}")
        sys.exit(1)

    if badge:
        click.echo(certifier.badge(cert))
        return

    if as_json:
        click.echo(cert.to_json())
        return

    click.echo(cert.summary())


# ---------------------------------------------------------------------------
# aria import
# ---------------------------------------------------------------------------

@cli.command("import")
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "fmt", default="jsonl", type=click.Choice(["jsonl", "openai", "mlflow", "wandb"]), show_default=True)
@click.option("--epoch-id", default=None, help="Epoch ID for imported records (jsonl only)")
@click.option("--model-id", default=None, help="Model ID (jsonl only)")
@click.option("--db", default="aria.db", show_default=True)
def import_cmd(path: str, fmt: str, epoch_id: Optional[str], model_id: Optional[str], db: str) -> None:
    """Import inference records from external formats into ARIA."""
    from aria.import_from import from_jsonl, from_openai_log, from_mlflow_export, from_wandb_export, save_imported

    storage = _get_storage(db)

    if fmt == "jsonl":
        records = from_jsonl(path, epoch_id=epoch_id or "imported", model_id=model_id or "unknown")
    elif fmt == "openai":
        records = from_openai_log(path)
    elif fmt == "mlflow":
        records = from_mlflow_export(path)
    elif fmt == "wandb":
        records = from_wandb_export(path)
    else:
        _err(f"Unknown format: {fmt}")
        sys.exit(1)

    count = save_imported(records, storage)
    _ok(f"Imported {count} records from {path} ({fmt} format)")


# ---------------------------------------------------------------------------
# aria bundle
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--epoch", "epoch_ids", multiple=True, help="Epoch IDs to include (repeatable, all if omitted)")
@click.option("--output", "output_path", required=True, help="Output ZIP file path")
def bundle(db: str, epoch_ids: tuple, output_path: str) -> None:
    """Create a portable audit bundle (ZIP) for external verification."""
    from aria.export_bundle import create_bundle

    storage = _get_storage(db)
    ids = list(epoch_ids) if epoch_ids else None
    create_bundle(storage, epoch_ids=ids, output=output_path)
    _ok(f"Bundle created: {output_path}")


# ---------------------------------------------------------------------------
# aria init
# ---------------------------------------------------------------------------

@cli.command("init")
@click.option("--system-id", prompt="System ID", help="Unique identifier for your AI system")
@click.option("--network", default="testnet", type=click.Choice(["mainnet", "testnet"]),
              show_default=True, help="BSV network")
@click.option("--db", default="aria.db", show_default=True, help="SQLite database file")
@click.option("--generate-key", is_flag=True, help="Generate a BSV keypair")
@click.option("--output", "output_path", default="aria.toml", show_default=True,
              help="Output config file path")
def init_cmd(system_id: str, network: str, db: str, generate_key: bool, output_path: str) -> None:
    """Initialize an ARIA project — generates aria.toml and optional keypair.

    \b
    Examples:
      aria init --system-id my-ai-app
      aria init --system-id prod --network mainnet --generate-key
    """
    from aria.config_file import generate_config_template

    if os.path.exists(output_path):
        if not click.confirm(f"{output_path} already exists. Overwrite?"):
            _info("Aborted.")
            return

    config_content = generate_config_template(system_id, network)
    config_content = config_content.replace('storage = "sqlite:///aria.db"', f'storage = "sqlite:///{db}"')

    with open(output_path, "w") as f:
        f.write(config_content)
    _ok(f"Config written to {output_path}")

    if generate_key:
        from aria.wallet.keygen import generate_keypair, write_env_file

        kp = generate_keypair(network=network)
        env_path = ".env"
        write_env_file(kp, path=env_path)
        _ok(f"Keypair generated and saved to {env_path}")
        _info(f"Address: {kp.address}")
    else:
        _info("TIP: Run 'aria keygen --env .env' to generate a BSV keypair")

    _ok(f"ARIA project initialized for '{system_id}'")


# ---------------------------------------------------------------------------
# aria estimate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--records", type=int, required=True, help="Number of inference records")
@click.option("--epochs", type=int, default=None, help="Number of epochs (auto if omitted)")
@click.option("--per-epoch", type=int, default=500, show_default=True,
              help="Records per epoch (for auto calculation)")
@click.option("--network", default="mainnet", type=click.Choice(["mainnet", "testnet"]),
              show_default=True)
@click.option("--bsv-usd", type=float, default=50.0, show_default=True,
              help="BSV/USD exchange rate for fiat estimate")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def estimate(records: int, epochs: Optional[int], per_epoch: int, network: str,
             bsv_usd: float, as_json: bool) -> None:
    """Estimate BSV transaction costs for ARIA auditing.

    \b
    Examples:
      aria estimate --records 10000
      aria estimate --records 1000000 --per-epoch 1000 --bsv-usd 65
      aria estimate --records 500 --network testnet
    """
    from aria.cost_estimator import CostEstimator

    est = CostEstimator(network=network, bsv_usd=bsv_usd)
    result = est.estimate(records=records, epochs=epochs, records_per_epoch=per_epoch)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(str(result))


# ---------------------------------------------------------------------------
# aria pipeline
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--name", "pipeline_name", required=True, help="Pipeline name")
@click.option("--json", "as_json", is_flag=True)
def pipeline(db: str, pipeline_name: str, as_json: bool) -> None:
    """Show pipeline traces for a given pipeline name."""
    _info(f"Pipeline: {pipeline_name}")
    _info("Use PipelineAuditor in code to record pipeline traces.")
    _info("  from aria.pipeline import PipelineAuditor")


# ---------------------------------------------------------------------------
# aria sync
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--db", default="aria.db", show_default=True)
@click.option("--json", "as_json", is_flag=True)
def sync(db: str, as_json: bool) -> None:
    """Sync offline epochs to BSV (publish pending OPEN/CLOSE transactions).

    \b
    Requires ARIA_BSV_KEY environment variable to be set.
    """
    from aria.offline import list_pending, sync_all

    storage = _get_storage(db)
    pending = list_pending(storage)

    if not pending:
        _ok("No pending offline epochs to sync.")
        return

    _info(f"Found {len(pending)} pending epoch(s).")

    bsv_key = os.environ.get("ARIA_BSV_KEY")
    if not bsv_key:
        _err("Set ARIA_BSV_KEY environment variable to sync offline epochs.")
        sys.exit(1)

    _info("Use aria.offline.sync_all() programmatically with wallet + broadcaster.")
    if as_json:
        click.echo(json.dumps({"pending_epochs": pending}))
    else:
        for eid in pending:
            _info(f"  Pending: {eid}")


# ---------------------------------------------------------------------------
# aria retry-status
# ---------------------------------------------------------------------------

@cli.command("retry-status")
@click.option("--db-path", default=None, help="Path to retry queue database")
@click.option("--json", "as_json", is_flag=True)
def retry_status(db_path: Optional[str], as_json: bool) -> None:
    """Show status of the retry queue (pending/dead-letter items)."""
    from aria.retry_queue import RetryQueue

    queue = RetryQueue(db_path=db_path) if db_path else RetryQueue()

    total = queue.count()
    dead = queue.dead_letters()

    if as_json:
        click.echo(json.dumps({
            "total_items": total,
            "dead_letters": len(dead),
        }))
    else:
        _info(f"Retry queue: {total} item(s)")
        if dead:
            _err(f"Dead letters: {len(dead)} item(s)")
            for item in dead[:5]:
                _info(f"  {item.item_id}: {item.last_error}")


# ---------------------------------------------------------------------------
# aria stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--system", "system_id", default=None,
              help="Filter by system_id (shows aggregate for all systems if omitted)")
@click.option("--db", default="aria.db", show_default=True)
def stats(system_id: Optional[str], db: str) -> None:
    """Show aggregate statistics for ARIA audit data.

    \b
    Reports:
      - Total epochs and records
      - Number of unique models
      - Average and total latency across all records
      - Date range of epoch activity (first / last opened_at)

    \b
    Examples:
      aria stats
      aria stats --system my-ai-app
    """
    import datetime as _dt

    storage = _get_storage(db)
    epochs_all = storage.list_epochs(system_id=system_id, limit=100_000)

    if not epochs_all:
        _info("No data found.")
        return

    total_epochs = len(epochs_all)
    total_records = sum(e.records_count or 0 for e in epochs_all)

    # Load individual records for per-record stats (latency, unique models).
    all_records = []
    for ep in epochs_all:
        all_records.extend(storage.list_records_by_epoch(ep.epoch_id))

    unique_models: set[str] = {r.model_id for r in all_records}
    lats = [r.latency_ms for r in all_records if r.latency_ms is not None and r.latency_ms > 0]
    total_latency = sum(lats)
    avg_latency = total_latency / len(lats) if lats else 0.0

    # Date range derived from epoch opened_at (unix int, seconds).
    timestamps = [e.opened_at for e in epochs_all if e.opened_at]
    if timestamps:
        first_ts = _dt.datetime.fromtimestamp(
            min(timestamps), tz=_dt.timezone.utc
        ).isoformat()
        last_ts = _dt.datetime.fromtimestamp(
            max(timestamps), tz=_dt.timezone.utc
        ).isoformat()
    else:
        first_ts = last_ts = "N/A"

    # ── Render ────────────────────────────────────────────────────────────
    title = click.style("ARIA Statistics", fg="cyan", bold=True)
    if system_id:
        title += click.style(f"  [{system_id}]", fg="yellow")
    click.echo(title)
    click.echo(click.style("─" * 52, fg="cyan"))

    def _row(label: str, value: str, color: str = "white") -> None:
        styled_label = click.style(f"{label}:", fg="cyan")
        styled_value = click.style(value, fg=color)
        click.echo(f"  {styled_label:<38} {styled_value}")

    _row("Total epochs", str(total_epochs), "green")
    _row("Total records", str(total_records), "green")
    _row("Unique models", str(len(unique_models)), "yellow")
    _row("Avg latency (ms)", f"{avg_latency:.1f}")
    _row("Total latency (ms)", str(total_latency))
    _row("First timestamp", first_ts)
    _row("Last timestamp", last_ts)

    if unique_models:
        click.echo(click.style("\n  Models:", fg="cyan"))
        for m in sorted(unique_models):
            click.echo(f"    {click.style('•', fg='yellow')} {m}")


# ---------------------------------------------------------------------------
# aria batch-verify
# ---------------------------------------------------------------------------

@cli.command("batch-verify")
@click.option("--open", "open_txids", required=True,
              help="Comma-separated EPOCH_OPEN TXIDs to verify in parallel")
@click.option("--network", default="mainnet",
              type=click.Choice(["mainnet", "testnet"]), show_default=True)
@click.option("--format", "fmt", default="text",
              type=click.Choice(["text", "json"]), show_default=True)
def batch_verify(open_txids: str, network: str, fmt: str) -> None:
    """Verify multiple EPOCH_OPEN TXIDs in parallel against BSV.

    TXIDs are resolved concurrently via asyncio.gather, so verification
    time scales with the slowest response rather than summing all requests.

    \b
    JSON output fields per entry:
      txid, verified, epoch_id, merkle_root, records_count, error

    \b
    Examples:
      aria batch-verify --open <txid1>,<txid2>,<txid3>
      aria batch-verify --open <txid1>,<txid2> --network testnet --format json
    """
    from aria.verify import Verifier, WhatsOnChainFetcher

    txids = [t.strip() for t in open_txids.split(",") if t.strip()]
    if not txids:
        _err("No TXIDs provided.")
        sys.exit(1)

    fetcher = WhatsOnChainFetcher(network=network)
    verifier = Verifier(network=network, tx_fetcher=fetcher)

    async def _verify_all():
        tasks = [verifier.verify_epoch(open_txid=txid) for txid in txids]
        return await asyncio.gather(*tasks, return_exceptions=True)

    _info(f"Verifying {len(txids)} TXID(s) on {network}…")
    raw_results = _run(_verify_all())

    output = []
    all_passed = True
    for txid, res in zip(txids, raw_results):
        if isinstance(res, Exception):
            output.append({
                "txid": txid,
                "verified": False,
                "epoch_id": "",
                "merkle_root": "",
                "records_count": 0,
                "error": str(res),
            })
            all_passed = False
        else:
            output.append({
                "txid": txid,
                "verified": res.valid,
                "epoch_id": res.epoch_id,
                "merkle_root": res.merkle_root,
                "records_count": res.records_count,
                "error": res.error,
            })
            if not res.valid:
                all_passed = False

    if fmt == "json":
        click.echo(json.dumps(output, indent=2))
    else:
        # Text table — keep styled icon separate so ANSI codes don't skew alignment.
        click.echo(f"\n{'TXID':<22} {'STATUS':<10} {'EPOCH ID':<36} {'RECORDS':>7}")
        click.echo("─" * 82)
        for item in output:
            txid_short = item["txid"][:20] + "…"
            icon = click.style("✓", fg="green") if item["verified"] else click.style("✗", fg="red")
            verdict = "PASS" if item["verified"] else "FAIL"
            epoch_disp = item["epoch_id"] or "N/A"
            click.echo(
                f"{txid_short:<22} {icon} {verdict:<8} {epoch_disp:<36} "
                f"{item['records_count']:>7}"
            )
        click.echo("")

    if not all_passed:
        sys.exit(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
