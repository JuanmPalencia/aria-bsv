"""
aria.jupyter — Jupyter cell magic for ARIA auditing.

Provides ``%%aria`` cell magic to track notebook inference cells
with automatic ARIA audit records.

Usage (in a Jupyter notebook)::

    %load_ext aria.jupyter

    %%aria track --model gpt-4 --confidence 0.95
    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

The cell output is automatically hashed and recorded as an ARIA audit record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

_IPYTHON_AVAILABLE = False
try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic  # type: ignore[import]
    from IPython.core.magic_arguments import (  # type: ignore[import]
        argument,
        magic_arguments,
        parse_argstring,
    )
    from IPython import get_ipython  # type: ignore[import]
    _IPYTHON_AVAILABLE = True
except ImportError:
    pass


@dataclass
class NotebookRecord:
    """A lightweight record of a notebook cell execution."""

    record_id: str
    model_id: str
    cell_hash: str
    output_hash: str
    confidence: float | None
    latency_ms: int
    cell_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "model_id": self.model_id,
            "cell_hash": self.cell_hash,
            "output_hash": self.output_hash,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "cell_number": self.cell_number,
        }


class NotebookTracker:
    """Tracks cell executions in a Jupyter notebook without IPython dependency.

    Can be used standalone for testing or in non-IPython environments.

    Args:
        system_id: ARIA system identifier.
        auditor:   Optional InferenceAuditor instance for full auditing.
    """

    def __init__(
        self,
        system_id: str = "jupyter-notebook",
        auditor: Any | None = None,
    ) -> None:
        self.system_id = system_id
        self._auditor = auditor
        self._records: list[NotebookRecord] = []
        self._cell_counter = 0

    @property
    def records(self) -> list[NotebookRecord]:
        return list(self._records)

    def track_cell(
        self,
        cell_source: str,
        cell_output: Any,
        model_id: str = "unknown",
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NotebookRecord:
        """Record a cell execution.

        Args:
            cell_source: The cell source code.
            cell_output: The cell output (will be hashed).
            model_id:    Model identifier.
            confidence:  Confidence score.
            metadata:    Extra metadata.

        Returns:
            NotebookRecord
        """
        self._cell_counter += 1
        started = time.time()

        cell_hash = hashlib.sha256(cell_source.encode()).hexdigest()
        output_str = json.dumps(cell_output, sort_keys=True, default=str) if not isinstance(cell_output, str) else cell_output
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()

        elapsed = int((time.time() - started) * 1000)

        rec = NotebookRecord(
            record_id=uuid.uuid4().hex[:16],
            model_id=model_id,
            cell_hash=cell_hash,
            output_hash=output_hash,
            confidence=confidence,
            latency_ms=elapsed,
            cell_number=self._cell_counter,
            metadata=metadata or {},
        )

        self._records.append(rec)

        if self._auditor is not None:
            try:
                self._auditor.record(
                    model_id=model_id,
                    input_data=cell_source,
                    output_data=cell_output,
                    confidence=confidence,
                    latency_ms=elapsed,
                    metadata={"cell_number": self._cell_counter, **(metadata or {})},
                )
            except Exception as exc:
                _log.warning("Failed to record to auditor: %s", exc)

        return rec

    def summary(self) -> str:
        lines = [
            f"ARIA Notebook Tracker: {self.system_id}",
            f"Tracked cells: {len(self._records)}",
        ]
        for r in self._records:
            lines.append(
                f"  [{r.cell_number}] model={r.model_id}, conf={r.confidence}, "
                f"hash={r.output_hash[:12]}..."
            )
        return "\n".join(lines)


if _IPYTHON_AVAILABLE:

    @magics_class
    class ARIAMagics(Magics):
        """ARIA cell magic for Jupyter notebooks."""

        _tracker: NotebookTracker | None = None

        @line_magic
        def aria_init(self, line: str) -> None:
            """Initialize ARIA tracking: %aria_init <system_id>"""
            system_id = line.strip() or "jupyter-notebook"
            ARIAMagics._tracker = NotebookTracker(system_id=system_id)
            print(f"ARIA tracking initialized for system: {system_id}")

        @line_magic
        def aria_summary(self, line: str) -> None:
            """Show summary of tracked cells: %aria_summary"""
            if ARIAMagics._tracker is None:
                print("No tracker initialized. Run %aria_init first.")
                return
            print(ARIAMagics._tracker.summary())

        @cell_magic
        @magic_arguments()
        @argument("command", nargs="?", default="track", help="Sub-command (track)")
        @argument("--model", default="unknown", help="Model identifier")
        @argument("--confidence", type=float, default=None, help="Confidence score")
        def aria(self, line: str, cell: str) -> None:
            """%%aria track --model gpt-4 --confidence 0.95

            Execute and audit a notebook cell.
            """
            args = parse_argstring(self.aria, line)

            if ARIAMagics._tracker is None:
                ARIAMagics._tracker = NotebookTracker()

            ip = self.shell
            result = ip.run_cell(cell) if ip else None
            output = result.result if result else None

            rec = ARIAMagics._tracker.track_cell(
                cell_source=cell,
                cell_output=output,
                model_id=args.model,
                confidence=args.confidence,
            )
            print(f"[ARIA] Recorded cell #{rec.cell_number}: model={rec.model_id}, hash={rec.output_hash[:12]}...")


def load_ipython_extension(ipython: Any) -> None:
    """Called by %load_ext aria.jupyter"""
    if not _IPYTHON_AVAILABLE:
        print("IPython is required for ARIA magic commands.")
        return
    ipython.register_magics(ARIAMagics)


def unload_ipython_extension(ipython: Any) -> None:
    """Called when the extension is unloaded."""
    pass
