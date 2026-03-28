"""
aria.integrations.mlflow — MLflow experiment tracking integration for ARIA.

Auto-logs every MLflow run's parameters, metrics, and artifacts to ARIA,
creating BSV-anchored audit trails for all your MLflow experiments.

Usage::

    from aria.integrations.mlflow import ARIAMLflowCallback, log_run_to_aria

    # Option A: callback (MLflow ≥ 2.0)
    import mlflow
    mlflow.set_experiment("my-experiment")
    with mlflow.start_run():
        mlflow.log_param("lr", 0.01)
        mlflow.log_metric("accuracy", 0.95)
        # ... training ...
        log_run_to_aria(mlflow.active_run(), auditor=auditor)

    # Option B: wrapper
    from aria.integrations.mlflow import ARIAMLflowLogger

    logger = ARIAMLflowLogger(auditor=auditor)
    logger.log_run(run_id="abc123")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        raise ImportError("mlflow not installed. pip install aria-bsv[mlflow]")


def _run_to_dict(run: Any) -> dict:
    """Extract relevant fields from an MLflow Run object."""
    try:
        info = run.info
        data = run.data
        return {
            "run_id":      info.run_id,
            "experiment_id": info.experiment_id,
            "run_name":    getattr(info, "run_name", ""),
            "status":      info.status,
            "start_time":  str(info.start_time),
            "end_time":    str(getattr(info, "end_time", "")),
            "params":      dict(data.params),
            "metrics":     {k: v for k, v in data.metrics.items()},
            "tags":        {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
        }
    except Exception as exc:
        _log.warning("mlflow: could not extract run data: %s", exc)
        return {"raw": str(run)[:500]}


def log_run_to_aria(
    run: Any,
    auditor: "InferenceAuditor | None" = None,
    aria: "ARIAQuick | None" = None,
    model_id: str | None = None,
) -> str | None:
    """Log a single MLflow run to ARIA.

    Args:
        run:      MLflow Run object (from mlflow.active_run() or mlflow.get_run()).
        auditor:  InferenceAuditor instance.
        aria:     ARIAQuick instance.
        model_id: Override model label. Defaults to run_name or run_id.

    Returns:
        ARIA record_id if successful, else None.
    """
    run_dict = _run_to_dict(run)
    mid = model_id or run_dict.get("run_name") or run_dict.get("run_id", "mlflow-run")
    input_data  = {"params": run_dict.get("params", {}), "tags": run_dict.get("tags", {})}
    output_data = {"metrics": run_dict.get("metrics", {})}
    metadata    = {
        "provider":      "mlflow",
        "run_id":        run_dict.get("run_id", ""),
        "experiment_id": run_dict.get("experiment_id", ""),
        "status":        run_dict.get("status", ""),
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
        _log.warning("log_run_to_aria: record error: %s", exc)
    return None


class ARIAMLflowLogger:
    """Logs MLflow runs to ARIA by run ID or from the active run.

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

    def log_run(self, run_id: str) -> str | None:
        """Log a specific run by ID."""
        mlflow = _get_mlflow()
        run = mlflow.get_run(run_id)
        return log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                               model_id=self._model_id)

    def log_active_run(self) -> str | None:
        """Log the currently active MLflow run."""
        mlflow = _get_mlflow()
        run = mlflow.active_run()
        if run is None:
            _log.warning("ARIAMLflowLogger: no active run")
            return None
        return log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                               model_id=self._model_id)

    def log_experiment(
        self,
        experiment_name: str,
        max_runs: int = 50,
    ) -> list[str]:
        """Log all runs in an experiment to ARIA.

        Args:
            experiment_name: MLflow experiment name.
            max_runs:        Maximum number of runs to log.

        Returns:
            List of ARIA record IDs.
        """
        mlflow = _get_mlflow()
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            max_results=max_runs,
            output_format="list",
        )
        record_ids = []
        for run in runs:
            rid = log_run_to_aria(run, auditor=self._auditor, aria=self._aria,
                                  model_id=self._model_id)
            if rid:
                record_ids.append(rid)
        return record_ids
