"""
aria.integrations.wandb — Weights & Biases integration for ARIA.

Logs W&B runs and sweeps to ARIA, creating BSV-anchored audit records
for every experiment tracked in Weights & Biases.

Usage::

    from aria.integrations.wandb import ARIAWandbLogger, log_run_to_aria

    # Log a single run
    import wandb
    run = wandb.init(project="my-project")
    log_run_to_aria(run, auditor=auditor)

    # Or use the logger
    logger = ARIAWandbLogger(auditor=auditor)
    logger.log_run(run_id="abc123def", project="my-project", entity="my-team")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


def _get_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError("wandb not installed. pip install aria-bsv[wandb]")


def _run_to_dict(run: Any) -> dict:
    """Extract relevant fields from a W&B Run object."""
    try:
        return {
            "run_id":    getattr(run, "id", ""),
            "run_name":  getattr(run, "name", ""),
            "project":   getattr(run, "project", ""),
            "entity":    getattr(run, "entity", ""),
            "state":     getattr(run, "state", ""),
            "url":       getattr(run, "url", ""),
            "config":    dict(getattr(run, "config", {}) or {}),
            "summary":   {
                k: v for k, v in (getattr(run, "summary", {}) or {}).items()
                if not k.startswith("_")
            },
            "tags":      list(getattr(run, "tags", []) or []),
        }
    except Exception as exc:
        _log.warning("wandb: could not extract run data: %s", exc)
        return {"raw": str(run)[:500]}


def log_run_to_aria(
    run: Any,
    auditor: "InferenceAuditor | None" = None,
    aria: "ARIAQuick | None" = None,
    model_id: str | None = None,
) -> str | None:
    """Log a W&B run to ARIA.

    Args:
        run:      W&B Run object.
        auditor:  InferenceAuditor instance.
        aria:     ARIAQuick instance.
        model_id: Override model label. Defaults to run_name or run_id.

    Returns:
        ARIA record_id if successful, else None.
    """
    run_dict = _run_to_dict(run)
    mid = model_id or run_dict.get("run_name") or run_dict.get("run_id", "wandb-run")
    input_data  = {"config": run_dict.get("config", {}), "tags": run_dict.get("tags", [])}
    output_data = {"summary": run_dict.get("summary", {})}
    metadata    = {
        "provider":  "wandb",
        "run_id":    run_dict.get("run_id", ""),
        "project":   run_dict.get("project", ""),
        "entity":    run_dict.get("entity", ""),
        "state":     run_dict.get("state", ""),
        "url":       run_dict.get("url", ""),
    }

    try:
        if auditor is not None:
            return auditor.record(mid, input_data, output_data, metadata=metadata)
        elif aria is not None:
            return aria.record(
                model_id=mid,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            )
    except Exception as exc:
        _log.warning("log_run_to_aria (wandb): record error: %s", exc)
    return None


class ARIAWandbLogger:
    """Logs W&B runs and sweeps to ARIA.

    Args:
        auditor:  InferenceAuditor instance.
        aria:     ARIAQuick instance.
        model_id: Default model label override.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor  = auditor
        self._aria     = aria
        self._model_id = model_id

    def log_run(
        self,
        run_id: str,
        project: str,
        entity: str | None = None,
    ) -> str | None:
        """Log a specific run by ID.

        Args:
            run_id:  W&B run ID.
            project: W&B project name.
            entity:  W&B entity (username or team). Defaults to current user.
        """
        wandb = _get_wandb()
        api = wandb.Api()
        path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
        run = api.run(path)
        return log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                               model_id=self._model_id)

    def log_active_run(self) -> str | None:
        """Log the currently active W&B run."""
        wandb = _get_wandb()
        run = wandb.run
        if run is None:
            _log.warning("ARIAWandbLogger: no active run")
            return None
        return log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                               model_id=self._model_id)

    def log_project(
        self,
        project: str,
        entity: str | None = None,
        max_runs: int = 50,
    ) -> list[str]:
        """Log all runs in a W&B project to ARIA.

        Args:
            project:  W&B project name.
            entity:   W&B entity.
            max_runs: Maximum number of runs to log.

        Returns:
            List of ARIA record IDs.
        """
        wandb = _get_wandb()
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, per_page=max_runs)
        record_ids = []
        for run in runs:
            rid = log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                                  model_id=self._model_id)
            if rid:
                record_ids.append(rid)
        return record_ids
