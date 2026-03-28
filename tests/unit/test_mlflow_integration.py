"""Tests for aria.integrations.mlflow — ARIAMLflowLogger."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from aria.integrations.mlflow import _run_to_dict, log_run_to_aria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(
    run_id="abc123",
    run_name="my-run",
    experiment_id="exp-1",
    status="FINISHED",
    params=None,
    metrics=None,
    tags=None,
):
    run = MagicMock()
    run.info.run_id       = run_id
    run.info.run_name     = run_name
    run.info.experiment_id = experiment_id
    run.info.status       = status
    run.info.start_time   = 1700000000000
    run.info.end_time     = 1700003600000
    run.data.params       = params or {"lr": "0.01", "batch_size": "32"}
    run.data.metrics      = metrics or {"accuracy": 0.95, "loss": 0.05}
    run.data.tags         = tags or {"mlflow.user": "test", "custom_tag": "v2"}
    return run


def _fake_mlflow_module(run=None, runs=None):
    mod = ModuleType("mlflow")
    if run:
        mod.active_run = MagicMock(return_value=run)
        mod.get_run = MagicMock(return_value=run)
    mod.search_runs = MagicMock(return_value=runs or [])
    return mod


# ---------------------------------------------------------------------------
# _run_to_dict
# ---------------------------------------------------------------------------

class TestRunToDict:
    def test_basic(self):
        run = _mock_run()
        d = _run_to_dict(run)
        assert d["run_id"] == "abc123"
        assert d["run_name"] == "my-run"
        assert d["params"]["lr"] == "0.01"
        assert d["metrics"]["accuracy"] == pytest.approx(0.95)

    def test_tags_filtered(self):
        run = _mock_run(tags={"mlflow.user": "hidden", "custom": "kept"})
        d = _run_to_dict(run)
        assert "mlflow.user" not in d["tags"]
        assert "custom" in d["tags"]

    def test_exception_returns_raw(self):
        d = _run_to_dict(None)
        assert "raw" in d


# ---------------------------------------------------------------------------
# log_run_to_aria
# ---------------------------------------------------------------------------

class TestLogRunToAria:
    def test_records_to_auditor(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"
        run = _mock_run()
        record_id = log_run_to_aria(run, auditor=auditor)
        assert record_id == "rec-1"
        auditor.record.assert_called_once()
        args = auditor.record.call_args[0]
        assert args[0] == "my-run"  # model_id = run_name

    def test_records_to_aria(self):
        aria = MagicMock()
        aria.record.return_value = "rec-2"
        run = _mock_run()
        record_id = log_run_to_aria(run, aria=aria)
        assert record_id == "rec-2"
        aria.record.assert_called_once()

    def test_model_id_override(self):
        auditor = MagicMock()
        run = _mock_run()
        log_run_to_aria(run, auditor=auditor, model_id="custom-id")
        args = auditor.record.call_args[0]
        assert args[0] == "custom-id"

    def test_no_auditor_returns_none(self):
        run = _mock_run()
        result = log_run_to_aria(run)
        assert result is None

    def test_metadata_has_provider(self):
        auditor = MagicMock()
        run = _mock_run()
        log_run_to_aria(run, auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "mlflow"

    def test_metadata_has_run_id(self):
        auditor = MagicMock()
        run = _mock_run(run_id="xyz789")
        log_run_to_aria(run, auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["run_id"] == "xyz789"

    def test_exception_returns_none(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        run = _mock_run()
        result = log_run_to_aria(run, auditor=auditor)
        assert result is None

    def test_params_in_input(self):
        auditor = MagicMock()
        run = _mock_run(params={"lr": "1e-5"})
        log_run_to_aria(run, auditor=auditor)
        args = auditor.record.call_args[0]
        assert args[1]["params"]["lr"] == "1e-5"

    def test_metrics_in_output(self):
        auditor = MagicMock()
        run = _mock_run(metrics={"f1": 0.88})
        log_run_to_aria(run, auditor=auditor)
        args = auditor.record.call_args[0]
        assert args[2]["metrics"]["f1"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# ARIAMLflowLogger
# ---------------------------------------------------------------------------

class TestARIAMLflowLogger:
    def test_log_run_fetches_and_records(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"
        run = _mock_run()
        fake_mlflow = _fake_mlflow_module(run=run)

        with patch_dict_sys_modules(fake_mlflow):
            from aria.integrations.mlflow import ARIAMLflowLogger
            logger = ARIAMLflowLogger(auditor=auditor)
            result = logger.log_run("abc123")

        assert result == "rec-1"

    def test_log_active_run(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-2"
        run = _mock_run()
        fake_mlflow = _fake_mlflow_module(run=run)

        with patch_dict_sys_modules(fake_mlflow):
            from aria.integrations.mlflow import ARIAMLflowLogger
            logger = ARIAMLflowLogger(auditor=auditor)
            result = logger.log_active_run()

        assert result == "rec-2"

    def test_log_active_run_no_active(self):
        auditor = MagicMock()
        fake_mlflow = _fake_mlflow_module()
        fake_mlflow.active_run = MagicMock(return_value=None)

        with patch_dict_sys_modules(fake_mlflow):
            from aria.integrations.mlflow import ARIAMLflowLogger
            logger = ARIAMLflowLogger(auditor=auditor)
            result = logger.log_active_run()

        assert result is None

    def test_import_error(self):
        with patch_dict_sys_modules(None):
            from aria.integrations import mlflow as m
            import importlib
            importlib.reload(m)
            with pytest.raises(ImportError, match="mlflow"):
                m._get_mlflow()


# ---------------------------------------------------------------------------
# Helper for sys.modules patching
# ---------------------------------------------------------------------------

from contextlib import contextmanager

@contextmanager
def patch_dict_sys_modules(fake_mod):
    import importlib
    old = sys.modules.get("mlflow")
    sys.modules["mlflow"] = fake_mod
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("mlflow", None)
        else:
            sys.modules["mlflow"] = old
        # Reload to reset cached imports
        import aria.integrations.mlflow as _m
        importlib.reload(_m)
