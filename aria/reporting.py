"""
aria.reporting — Epoch audit report generation.

Generates human-readable audit reports in multiple formats:
  - Text  (.txt)  — plain-text for logging / email
  - JSON  (.json) — machine-readable structured report
  - HTML  (.html) — self-contained report with styling
  - PDF   (.pdf)  — via WeasyPrint or ReportLab (optional dependencies)

The report includes:
  - Epoch metadata (ID, system, timestamps, tx hashes)
  - Merkle root and record count
  - Per-model statistics (record count, mean/p95 latency, mean confidence)
  - Compliance summary (BRC-120 §6 checks)
  - First N records table

Usage::

    from aria.reporting import ReportGenerator
    from aria.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage("aria.db")
    gen = ReportGenerator(storage)
    text = gen.render_text("epoch-abc")
    gen.save("epoch-abc", path="reports/epoch-abc.html", fmt="html")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core.record import AuditRecord
    from .storage.base import EpochRow, StorageInterface

from .analytics import CrossEpochAnalytics


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

@dataclass
class ModelReport:
    model_id: str
    record_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    mean_confidence: float | None
    model_hash: str


@dataclass
class ComplianceCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class EpochReport:
    epoch_id: str
    system_id: str
    open_txid: str
    close_txid: str
    opened_at_iso: str
    closed_at_iso: str | None
    records_count: int
    merkle_root: str
    model_reports: list[ModelReport]
    compliance_checks: list[ComplianceCheck]
    generated_at_iso: str
    aria_version: str = "0.1.0"

    @property
    def is_closed(self) -> bool:
        return bool(self.close_txid)

    @property
    def compliance_pass(self) -> bool:
        return all(c.passed for c in self.compliance_checks)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate multi-format audit reports for ARIA epochs.

    Args:
        storage: StorageInterface providing epochs and records.
        max_records_in_table: Maximum number of records to include in the
                              detail table (default 50).
    """

    def __init__(
        self,
        storage: "StorageInterface",
        max_records_in_table: int = 50,
    ) -> None:
        self._storage = storage
        self._analytics = CrossEpochAnalytics(storage)
        self._max_records = max_records_in_table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_report(self, epoch_id: str) -> EpochReport:
        """Build a structured EpochReport from storage data."""
        row = self._storage.get_epoch(epoch_id)
        if row is None:
            raise ValueError(f"Epoch {epoch_id!r} not found in storage")

        records = self._storage.list_records_by_epoch(epoch_id)

        model_reports = self._build_model_reports(row, records)
        compliance = self._build_compliance_checks(row, records)

        return EpochReport(
            epoch_id=epoch_id,
            system_id=row.system_id,
            open_txid=row.open_txid,
            close_txid=row.close_txid or "",
            opened_at_iso=_ms_to_iso(row.opened_at),
            closed_at_iso=_ms_to_iso(row.closed_at) if row.closed_at else None,
            records_count=len(records),
            merkle_root=row.merkle_root or "",
            model_reports=model_reports,
            compliance_checks=compliance,
            generated_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    def render_text(self, epoch_id: str) -> str:
        """Render a plain-text report."""
        report = self.build_report(epoch_id)
        return _render_text(report, self._storage.list_records_by_epoch(epoch_id))

    def render_json(self, epoch_id: str) -> str:
        """Render a JSON-serialised report."""
        report = self.build_report(epoch_id)
        return _render_json(report)

    def render_html(self, epoch_id: str) -> str:
        """Render a self-contained HTML report."""
        report = self.build_report(epoch_id)
        return _render_html(report, self._storage.list_records_by_epoch(epoch_id)[: self._max_records])

    def save(
        self,
        epoch_id: str,
        path: str | Path,
        fmt: Literal["text", "json", "html", "pdf"] = "html",
    ) -> Path:
        """Render the report and write it to *path*.

        Returns the absolute Path of the saved file.

        Raises:
            ImportError: if ``fmt="pdf"`` and neither WeasyPrint nor ReportLab
                         is installed.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "text":
            dest.write_text(self.render_text(epoch_id), encoding="utf-8")
        elif fmt == "json":
            dest.write_text(self.render_json(epoch_id), encoding="utf-8")
        elif fmt == "html":
            dest.write_text(self.render_html(epoch_id), encoding="utf-8")
        elif fmt == "pdf":
            html_content = self.render_html(epoch_id)
            _write_pdf(html_content, dest)
        else:
            raise ValueError(f"Unknown format {fmt!r}. Use text, json, html, or pdf.")

        return dest.resolve()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model_reports(self, row: "EpochRow", records: list["AuditRecord"]) -> list[ModelReport]:
        from collections import defaultdict
        buckets: dict[str, list["AuditRecord"]] = defaultdict(list)
        for r in records:
            buckets[r.model_id].append(r)

        reports = []
        for model_id, recs in sorted(buckets.items()):
            latencies = sorted(r.latency_ms for r in recs)
            confs = [r.confidence for r in recs if r.confidence is not None]
            from .analytics import _percentile
            p95 = _percentile(latencies, 95) if latencies else 0.0
            mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
            mean_conf = sum(confs) / len(confs) if confs else None

            model_hash = row.model_hashes.get(model_id, "")
            reports.append(ModelReport(
                model_id=model_id,
                record_count=len(recs),
                mean_latency_ms=round(mean_lat, 1),
                p95_latency_ms=round(p95, 1),
                mean_confidence=round(mean_conf, 4) if mean_conf is not None else None,
                model_hash=model_hash,
            ))
        return reports

    def _build_compliance_checks(
        self, row: "EpochRow", records: list["AuditRecord"]
    ) -> list[ComplianceCheck]:
        checks = []

        # BRC-120 §6.1 — EPOCH_OPEN tx must exist
        checks.append(ComplianceCheck(
            name="BRC-120 §6.1 EPOCH_OPEN broadcast",
            passed=bool(row.open_txid),
            detail=f"open_txid={row.open_txid or 'missing'}",
        ))

        # BRC-120 §6.2 — EPOCH_CLOSE tx must exist
        checks.append(ComplianceCheck(
            name="BRC-120 §6.2 EPOCH_CLOSE broadcast",
            passed=bool(row.close_txid),
            detail=f"close_txid={row.close_txid or 'missing'}",
        ))

        # BRC-120 §6.3 — Merkle root must be present on close
        merkle_ok = bool(row.merkle_root) if row.close_txid else True  # N/A if open
        checks.append(ComplianceCheck(
            name="BRC-120 §6.3 Merkle root integrity",
            passed=merkle_ok,
            detail=f"merkle_root={row.merkle_root or 'missing'}",
        ))

        # BRC-120 §6.4 — Model hashes must be committed
        model_hashes_ok = len(row.model_hashes) > 0
        checks.append(ComplianceCheck(
            name="BRC-120 §6.4 Model hash commitment",
            passed=model_hashes_ok,
            detail=f"{len(row.model_hashes)} model(s) committed",
        ))

        # BRC-120 §6.5 — Record count consistency
        count_ok = len(records) == row.records_count or not row.close_txid
        checks.append(ComplianceCheck(
            name="BRC-120 §6.5 Record count consistency",
            passed=count_ok,
            detail=(
                f"epoch.records_count={row.records_count}, "
                f"actual={len(records)}"
            ),
        ))

        # EU AI Act Art. 13 — Model identification
        model_id_ok = all(bool(mh) for mh in row.model_hashes.values())
        checks.append(ComplianceCheck(
            name="EU AI Act Art. 13 Model identification",
            passed=model_id_ok,
            detail="All models have committed hash values" if model_id_ok else "Missing model hashes",
        ))

        return checks


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------

def _render_text(report: EpochReport, records: list["AuditRecord"]) -> str:
    lines = [
        "=" * 70,
        "ARIA AUDIT REPORT",
        "=" * 70,
        f"Generated : {report.generated_at_iso}",
        f"Epoch ID  : {report.epoch_id}",
        f"System ID : {report.system_id}",
        f"Opened at : {report.opened_at_iso}",
        f"Closed at : {report.closed_at_iso or 'NOT CLOSED'}",
        f"Open txid : {report.open_txid or 'MISSING'}",
        f"Close txid: {report.close_txid or 'MISSING'}",
        f"Records   : {report.records_count}",
        f"Merkle root: {report.merkle_root or 'N/A'}",
        "",
        "─" * 70,
        "MODEL STATISTICS",
        "─" * 70,
    ]
    for m in report.model_reports:
        lines.append(
            f"  {m.model_id:<30}  count={m.record_count:>5}  "
            f"latency={m.mean_latency_ms:>8.1f}ms (p95={m.p95_latency_ms:.1f}ms)  "
            f"confidence={m.mean_confidence if m.mean_confidence is not None else 'N/A'}"
        )

    lines += [
        "",
        "─" * 70,
        "COMPLIANCE CHECKS (BRC-120 / EU AI Act)",
        "─" * 70,
    ]
    for c in report.compliance_checks:
        mark = "✓" if c.passed else "✗"
        lines.append(f"  [{mark}] {c.name}")
        lines.append(f"       {c.detail}")

    overall = "PASS" if report.compliance_pass else "FAIL"
    lines += [
        "",
        "─" * 70,
        f"OVERALL COMPLIANCE: {overall}",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------

def _render_json(report: EpochReport) -> str:
    data = {
        "aria_version": report.aria_version,
        "generated_at": report.generated_at_iso,
        "epoch": {
            "epoch_id": report.epoch_id,
            "system_id": report.system_id,
            "open_txid": report.open_txid,
            "close_txid": report.close_txid,
            "opened_at": report.opened_at_iso,
            "closed_at": report.closed_at_iso,
            "records_count": report.records_count,
            "merkle_root": report.merkle_root,
        },
        "model_stats": [
            {
                "model_id": m.model_id,
                "record_count": m.record_count,
                "mean_latency_ms": m.mean_latency_ms,
                "p95_latency_ms": m.p95_latency_ms,
                "mean_confidence": m.mean_confidence,
                "model_hash": m.model_hash,
            }
            for m in report.model_reports
        ],
        "compliance": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in report.compliance_checks
        ],
        "compliance_pass": report.compliance_pass,
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARIA Audit Report — {epoch_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f4f6fa;
          color: #1a1a2e; padding: 2rem; }}
  h1 {{ color: #0f3460; font-size: 1.6rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #16213e; font-size: 1.1rem; margin: 1.5rem 0 0.6rem;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem; }}
  .header {{ background: linear-gradient(135deg, #0f3460, #1a6b9e);
             color: #fff; padding: 1.5rem 2rem; border-radius: 8px;
             margin-bottom: 1.5rem; }}
  .header p {{ opacity: 0.85; font-size: 0.85rem; margin-top: 0.3rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.2rem 1.5rem;
           margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
           gap: 0.8rem; }}
  .kv {{ display: flex; flex-direction: column; }}
  .kv .label {{ font-size: 0.7rem; text-transform: uppercase; color: #718096;
                letter-spacing: 0.05em; }}
  .kv .value {{ font-size: 0.9rem; font-weight: 600; word-break: break-all; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #f7f8fc; color: #4a5568; text-align: left;
        padding: 0.5rem 0.8rem; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.04em; }}
  td {{ padding: 0.45rem 0.8rem; border-top: 1px solid #edf2f7; }}
  tr:hover td {{ background: #f7faff; }}
  .badge {{ display: inline-block; padding: 0.2em 0.6em; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; }}
  .pass {{ background: #c6f6d5; color: #276749; }}
  .fail {{ background: #fed7d7; color: #9b2c2c; }}
  .hash {{ font-family: monospace; font-size: 0.78rem; color: #4a5568; }}
  footer {{ text-align: center; color: #a0aec0; font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>ARIA Audit Report</h1>
  <p>Generated {generated_at} &nbsp;|&nbsp; ARIA v{aria_version}</p>
</div>

<div class="card">
  <h2>Epoch Metadata</h2>
  <div class="grid">
    <div class="kv"><span class="label">Epoch ID</span>
      <span class="value hash">{epoch_id}</span></div>
    <div class="kv"><span class="label">System ID</span>
      <span class="value">{system_id}</span></div>
    <div class="kv"><span class="label">Status</span>
      <span class="badge {status_class}">{status}</span></div>
    <div class="kv"><span class="label">Records</span>
      <span class="value">{records_count}</span></div>
    <div class="kv"><span class="label">Opened at</span>
      <span class="value">{opened_at}</span></div>
    <div class="kv"><span class="label">Closed at</span>
      <span class="value">{closed_at}</span></div>
    <div class="kv"><span class="label">Open txid</span>
      <span class="value hash">{open_txid}</span></div>
    <div class="kv"><span class="label">Close txid</span>
      <span class="value hash">{close_txid}</span></div>
    <div class="kv"><span class="label">Merkle root</span>
      <span class="value hash">{merkle_root}</span></div>
  </div>
</div>

<div class="card">
  <h2>Model Statistics</h2>
  {model_table}
</div>

<div class="card">
  <h2>Compliance Checks
    <span class="badge {compliance_class}" style="margin-left:0.5rem">{compliance_label}</span>
  </h2>
  {compliance_table}
</div>

{records_section}

<footer>ARIA &mdash; Auditable Real-time Inference Architecture &mdash; BRC-120 reference implementation</footer>
</body>
</html>
"""


def _render_html(report: EpochReport, records: list["AuditRecord"]) -> str:
    # Model table
    model_rows = "\n".join(
        f"<tr>"
        f"<td>{m.model_id}</td>"
        f"<td>{m.record_count}</td>"
        f"<td>{m.mean_latency_ms:.1f}</td>"
        f"<td>{m.p95_latency_ms:.1f}</td>"
        f"<td>{m.mean_confidence if m.mean_confidence is not None else '—'}</td>"
        f"<td class='hash'>{m.model_hash[:16] + '…' if m.model_hash else '—'}</td>"
        f"</tr>"
        for m in report.model_reports
    ) if report.model_reports else "<tr><td colspan='6'>No records in this epoch.</td></tr>"

    model_table = (
        "<table>"
        "<thead><tr>"
        "<th>Model ID</th><th>Records</th><th>Mean Latency (ms)</th>"
        "<th>p95 Latency (ms)</th><th>Mean Confidence</th><th>Model Hash</th>"
        "</tr></thead>"
        f"<tbody>{model_rows}</tbody></table>"
    )

    # Compliance table
    compliance_rows = "\n".join(
        f"<tr>"
        f"<td><span class='badge {'pass' if c.passed else 'fail'}'>{'✓' if c.passed else '✗'}</span></td>"
        f"<td>{c.name}</td>"
        f"<td>{c.detail}</td>"
        f"</tr>"
        for c in report.compliance_checks
    )
    compliance_table = (
        "<table>"
        "<thead><tr><th>Status</th><th>Check</th><th>Detail</th></tr></thead>"
        f"<tbody>{compliance_rows}</tbody></table>"
    )

    # Records section (first N)
    if records:
        record_rows = "\n".join(
            f"<tr>"
            f"<td>{r.sequence}</td>"
            f"<td class='hash'>{r.record_id[:16]}…</td>"
            f"<td>{r.model_id}</td>"
            f"<td class='hash'>{r.input_hash[:16]}…</td>"
            f"<td class='hash'>{r.output_hash[:16]}…</td>"
            f"<td>{r.latency_ms}</td>"
            f"<td>{r.confidence if r.confidence is not None else '—'}</td>"
            f"</tr>"
            for r in records
        )
        records_section = (
            '<div class="card">'
            f"<h2>Records (first {len(records)})</h2>"
            "<table>"
            "<thead><tr>"
            "<th>Seq</th><th>Record ID</th><th>Model</th>"
            "<th>Input Hash</th><th>Output Hash</th><th>Latency (ms)</th><th>Confidence</th>"
            "</tr></thead>"
            f"<tbody>{record_rows}</tbody></table></div>"
        )
    else:
        records_section = ""

    status = "CLOSED" if report.is_closed else "OPEN"
    status_class = "pass" if report.is_closed else "fail"
    compliance_class = "pass" if report.compliance_pass else "fail"
    compliance_label = "PASS" if report.compliance_pass else "FAIL"

    return _HTML_TEMPLATE.format(
        epoch_id=report.epoch_id,
        system_id=report.system_id,
        status=status,
        status_class=status_class,
        records_count=report.records_count,
        opened_at=report.opened_at_iso,
        closed_at=report.closed_at_iso or "—",
        open_txid=report.open_txid or "—",
        close_txid=report.close_txid or "—",
        merkle_root=report.merkle_root or "—",
        model_table=model_table,
        compliance_table=compliance_table,
        compliance_class=compliance_class,
        compliance_label=compliance_label,
        records_section=records_section,
        generated_at=report.generated_at_iso,
        aria_version=report.aria_version,
    )


# ---------------------------------------------------------------------------
# PDF writer (optional dependency)
# ---------------------------------------------------------------------------

def _write_pdf(html_content: str, dest: Path) -> None:
    """Write *html_content* to *dest* as PDF.

    Tries WeasyPrint first, then ReportLab (basic fallback), then raises ImportError.
    """
    try:
        from weasyprint import HTML as WeasyHTML  # type: ignore[import]
        WeasyHTML(string=html_content).write_pdf(str(dest))
        return
    except ImportError:
        pass

    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph  # type: ignore[import]
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        doc = SimpleDocTemplate(str(dest), pagesize=A4)
        styles = getSampleStyleSheet()
        # Strip HTML tags for ReportLab basic fallback
        import re
        text = re.sub(r"<[^>]+>", " ", html_content)
        doc.build([Paragraph(text[:4000], styles["Normal"])])
        return
    except ImportError:
        pass

    raise ImportError(
        "PDF generation requires WeasyPrint or ReportLab:\n"
        "  pip install aria-bsv[pdf]\n"
        "or:\n"
        "  pip install weasyprint\n"
        "  pip install reportlab"
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ms_to_iso(ms: int) -> str:
    if ms <= 0:
        return "—"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
