"""
aria.reports — High-level multi-epoch reporting.

Extends aria.reporting with multi-epoch summaries, date-range
filtering, and cross-system comparison reports.

Usage::

    from aria.reports import MultiReport

    mr = MultiReport(storage)

    # Generate a report spanning all epochs in the last 7 days
    report = mr.date_range("7d")
    print(report.summary())

    # Generate a full system report
    report = mr.system_report("my-ai-system")
    report.save("system_report.html")

    # Compare across systems
    report = mr.cross_system(["system-a", "system-b"])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface


@dataclass
class EpochSummary:
    """Summary stats for a single epoch."""
    epoch_id: str
    system_id: str
    records_count: int
    avg_confidence: float | None = None
    avg_latency_ms: float = 0.0
    is_closed: bool = False
    merkle_root: str = ""
    open_txid: str = ""


@dataclass
class MultiEpochReport:
    """Report spanning multiple epochs."""
    title: str
    generated_at: str
    epochs: list[EpochSummary] = field(default_factory=list)
    total_records: int = 0
    total_epochs: int = 0
    avg_confidence: float | None = None
    avg_latency_ms: float = 0.0
    models: list[str] = field(default_factory=list)
    duration_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "total_epochs": self.total_epochs,
            "total_records": self.total_records,
            "avg_confidence": self.avg_confidence,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "models": self.models,
            "duration_hours": round(self.duration_hours, 1),
            "epochs": [
                {
                    "epoch_id": e.epoch_id,
                    "system_id": e.system_id,
                    "records_count": e.records_count,
                    "avg_confidence": e.avg_confidence,
                    "avg_latency_ms": round(e.avg_latency_ms, 1),
                    "is_closed": e.is_closed,
                }
                for e in self.epochs
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        lines = [
            self.title,
            "=" * 60,
            f"  Generated: {self.generated_at}",
            f"  Epochs:    {self.total_epochs}",
            f"  Records:   {self.total_records}",
            f"  Models:    {', '.join(self.models) if self.models else 'N/A'}",
        ]
        if self.avg_confidence is not None:
            lines.append(f"  Avg Conf:  {self.avg_confidence:.4f}")
        lines.append(f"  Avg Lat:   {self.avg_latency_ms:.0f}ms")
        lines.append(f"  Duration:  {self.duration_hours:.1f}h")
        lines.append("-" * 60)
        for e in self.epochs[:20]:
            status = "closed" if e.is_closed else "open"
            lines.append(
                f"  {e.epoch_id[:24]:24s}  {e.records_count:>5d} recs  "
                f"conf={e.avg_confidence or 0:.3f}  [{status}]"
            )
        if len(self.epochs) > 20:
            lines.append(f"  ... and {len(self.epochs) - 20} more epochs")
        return "\n".join(lines)

    def save(self, path: str, fmt: str = "json") -> None:
        """Save report to file.

        Args:
            path: Output file path.
            fmt: 'json', 'text', or 'html'.
        """
        from pathlib import Path as P

        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            p.write_text(self.to_json(), encoding="utf-8")
        elif fmt == "text":
            p.write_text(self.summary(), encoding="utf-8")
        elif fmt == "html":
            p.write_text(self._render_html(), encoding="utf-8")

    def _render_html(self) -> str:
        rows = ""
        for e in self.epochs:
            status = '<span style="color:#22c55e">Closed</span>' if e.is_closed else '<span style="color:#eab308">Open</span>'
            rows += f"<tr><td>{e.epoch_id}</td><td>{e.system_id}</td><td>{e.records_count}</td><td>{e.avg_confidence or 'N/A'}</td><td>{status}</td></tr>\n"
        return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{self.title}</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}
h1{{color:#1a365d}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:.5rem;border-bottom:1px solid #e2e8f0;text-align:left}}
th{{background:#f7fafc}}.stat{{display:inline-block;margin:0 1.5rem 1rem 0}}
.stat h3{{margin:0;color:#64748b;font-size:.8rem}}.stat p{{margin:0;font-size:1.5rem;font-weight:700}}</style>
</head><body>
<h1>{self.title}</h1>
<p>Generated: {self.generated_at}</p>
<div>
<div class="stat"><h3>Epochs</h3><p>{self.total_epochs}</p></div>
<div class="stat"><h3>Records</h3><p>{self.total_records}</p></div>
<div class="stat"><h3>Avg Confidence</h3><p>{self.avg_confidence or 'N/A'}</p></div>
<div class="stat"><h3>Avg Latency</h3><p>{self.avg_latency_ms:.0f}ms</p></div>
</div>
<h2>Epochs</h2>
<table><tr><th>Epoch ID</th><th>System</th><th>Records</th><th>Confidence</th><th>Status</th></tr>
{rows}</table>
</body></html>"""


_DURATION_MAP = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_duration(s: str) -> float:
    s = s.strip().lower()
    for suffix, mult in _DURATION_MAP.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


class MultiReport:
    """High-level multi-epoch report generator.

    Args:
        storage: ARIA StorageInterface for data access.
    """

    def __init__(self, storage: "StorageInterface") -> None:
        self._storage = storage

    def date_range(
        self,
        duration: str,
        system_id: str | None = None,
    ) -> MultiEpochReport:
        """Generate a report for epochs within a time window.

        Args:
            duration: Human duration like '24h', '7d', '30d'.
            system_id: Filter to a specific system.
        """
        seconds = _parse_duration(duration)
        cutoff_s = int(time.time() - seconds)

        epochs = self._storage.list_epochs(system_id=system_id, limit=10_000)
        filtered = [e for e in epochs if e.opened_at >= cutoff_s]

        return self._build(
            title=f"ARIA Report — Last {duration}",
            epochs_data=filtered,
        )

    def system_report(self, system_id: str) -> MultiEpochReport:
        """Generate a comprehensive report for a specific system."""
        epochs = self._storage.list_epochs(system_id=system_id, limit=10_000)
        return self._build(
            title=f"ARIA System Report — {system_id}",
            epochs_data=epochs,
        )

    def cross_system(self, system_ids: list[str]) -> MultiEpochReport:
        """Generate a comparison report across multiple systems."""
        all_epochs = []
        for sid in system_ids:
            all_epochs.extend(self._storage.list_epochs(system_id=sid, limit=10_000))
        return self._build(
            title=f"ARIA Cross-System Report — {', '.join(system_ids)}",
            epochs_data=all_epochs,
        )

    def all_epochs(self, limit: int = 1000) -> MultiEpochReport:
        """Generate a report covering all epochs."""
        epochs = self._storage.list_epochs(limit=limit)
        return self._build(title="ARIA Full Report", epochs_data=epochs)

    def _build(self, title: str, epochs_data: list) -> MultiEpochReport:
        """Build a MultiEpochReport from epoch rows."""
        summaries: list[EpochSummary] = []
        all_confs: list[float] = []
        all_lats: list[int] = []
        all_models: set[str] = set()
        total_records = 0

        for ep in epochs_data:
            records = self._storage.list_records_by_epoch(ep.epoch_id)
            confs = [r.confidence for r in records if r.confidence is not None]
            lats = [r.latency_ms for r in records]
            models = set(r.model_id for r in records)
            all_models.update(models)
            all_confs.extend(confs)
            all_lats.extend(lats)
            total_records += len(records)

            summaries.append(EpochSummary(
                epoch_id=ep.epoch_id,
                system_id=ep.system_id,
                records_count=len(records),
                avg_confidence=round(sum(confs) / len(confs), 6) if confs else None,
                avg_latency_ms=sum(lats) / len(lats) if lats else 0,
                is_closed=bool(ep.close_txid),
                merkle_root=ep.merkle_root,
                open_txid=ep.open_txid,
            ))

        # Duration
        duration_h = 0.0
        if epochs_data:
            times = [e.opened_at for e in epochs_data if e.opened_at]
            if len(times) >= 2:
                duration_h = (max(times) - min(times)) / 3600

        return MultiEpochReport(
            title=title,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            epochs=summaries,
            total_records=total_records,
            total_epochs=len(summaries),
            avg_confidence=(
                round(sum(all_confs) / len(all_confs), 6) if all_confs else None
            ),
            avg_latency_ms=(
                sum(all_lats) / len(all_lats) if all_lats else 0
            ),
            models=sorted(all_models),
            duration_hours=duration_h,
        )
