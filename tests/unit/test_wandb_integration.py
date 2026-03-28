"""Tests for aria.integrations.wandb — ARIAWandbLogger."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from aria.integrations.wandb import _run_to_dict, log_run_to_aria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(
    run_id="run1abc",
    run_name="experiment-1",
    project="my-project",
    entity="my-team",
    state="finished",
    config=None,
    summary=None,
    tags=None,
):
    run = MagicMock()
    run.id      = run_id
    run.name    = run_name
    run.project = project
    run.entity  = entity
    run.state   = state
    run.url     = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    run.config  = config or {"lr": 1e-3, "epochs": 10}
    run.summary = summary or {"accuracy": 0.92, "loss": 0.08, "_timestamp": 12345}
    run.tags    = tags or ["prod", "v2"]
    return run


# ---------------------------------------------------------------------------
# _run_to_dict
# ---------------------------------------------------------------------------

class TestRunToDict:
    def test_basic(self):
        run = _mock_run()
        d = _run_to_dict(run)
        assert d["run_id"] == "run1abc"
        assert d["run_name"] == "experiment-1"
        assert d["project"] == "my-project"

    def test_summary_filters_private_keys(self):
        run = _mock_run(summary={"accuracy": 0.9, "_step": 100, "_timestamp": 999})
        d = _run_to_dict(run)
        assert "accuracy" in d["summary"]
        assert "_step" not in d["summary"]
        assert "_timestamp" not in d["summary"]

    def test_config_included(self):
        run = _mock_run(config={"lr": 0.01})
        d = _run_to_dict(run)
        assert d["config"]["lr"] == pytest.approx(0.01)

    def test_exception_returns_raw(self):
        class Broken:
            @property
            def id(self):
                raise RuntimeError("broken")
        d = _run_to_dict(Broken())
        assert "raw" in d


# ---------------------------------------------------------------------------
# log_run_to_aria
# ---------------------------------------------------------------------------

class TestLogRunToAria:
    def test_records_to_auditor(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"
        run = _mock_run()
        result = log_run_to_aria(run, auditor=auditor)
        assert result == "rec-1"
        auditor.record.assert_called_once()

    def test_model_id_from_run_name(self):
        auditor = MagicMock()
        run = _mock_run(run_name="my-exp")
        log_run_to_aria(run, auditor=auditor)
        args = auditor.record.call_args[0]
        assert args[0] == "my-exp"

    def test_model_id_override(self):
        auditor = MagicMock()
        run = _mock_run()
        log_run_to_aria(run, auditor=auditor, model_id="override-id")
        args = auditor.record.call_args[0]
        assert args[0] == "override-id"

    def test_records_to_aria(self):
        aria = MagicMock()
        aria.record.return_value = "rec-2"
        run = _mock_run()
        result = log_run_to_aria(run, aria=aria)
        assert result == "rec-2"

    def test_no_auditor_returns_none(self):
        assert log_run_to_aria(_mock_run()) is None

    def test_provider_metadata(self):
        auditor = MagicMock()
        run = _mock_run()
        log_run_to_aria(run, auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "wandb"

    def test_run_url_in_metadata(self):
        auditor = MagicMock()
        run = _mock_run()
        log_run_to_aria(run, auditor=auditor)
        kwargs = auditor.record.call_args[1]
        assert "wandb.ai" in kwargs["metadata"]["url"]

    def test_config_in_input(self):
        auditor = MagicMock()
        run = _mock_run(config={"batch": 64})
        log_run_to_aria(run, auditor=auditor)
        args = auditor.record.call_args[0]
        assert args[1]["config"]["batch"] == 64

    def test_summary_in_output(self):
        auditor = MagicMock()
        run = _mock_run(summary={"f1": 0.88})
        log_run_to_aria(run, auditor=auditor)
        args = auditor.record.call_args[0]
        assert args[2]["summary"]["f1"] == pytest.approx(0.88)

    def test_exception_returns_none(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        result = log_run_to_aria(_mock_run(), auditor=auditor)
        assert result is None


# ---------------------------------------------------------------------------
# ARIAWandbLogger
# ---------------------------------------------------------------------------

def _fake_wandb_module(active_run=None):
    mod = ModuleType("wandb")
    mod.run = active_run
    mock_api = MagicMock()
    if active_run:
        mock_api.run = MagicMock(return_value=active_run)
        mock_api.runs = MagicMock(return_value=[active_run])
    mod.Api = MagicMock(return_value=mock_api)
    return mod


from contextlib import contextmanager

@contextmanager
def patch_wandb(fake_mod):
    import importlib
    old = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_mod
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = old
        import aria.integrations.wandb as _m
        importlib.reload(_m)


class TestARIAWandbLogger:
    def test_log_run(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"
        run = _mock_run()
        fake_wb = _fake_wandb_module(active_run=run)

        with patch_wandb(fake_wb):
            from aria.integrations.wandb import ARIAWandbLogger
            logger = ARIAWandbLogger(auditor=auditor)
            result = logger.log_run("run1abc", "my-project", "my-team")

        assert result == "rec-1"

    def test_log_active_run(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-2"
        run = _mock_run()
        fake_wb = _fake_wandb_module(active_run=run)

        with patch_wandb(fake_wb):
            from aria.integrations.wandb import ARIAWandbLogger
            logger = ARIAWandbLogger(auditor=auditor)
            result = logger.log_active_run()

        assert result == "rec-2"

    def test_log_active_run_none(self):
        auditor = MagicMock()
        fake_wb = _fake_wandb_module(active_run=None)

        with patch_wandb(fake_wb):
            from aria.integrations.wandb import ARIAWandbLogger
            logger = ARIAWandbLogger(auditor=auditor)
            result = logger.log_active_run()

        assert result is None

    def test_import_error(self):
        with patch_wandb(None):
            from aria.integrations import wandb as m
            import importlib
            importlib.reload(m)
            with pytest.raises(ImportError, match="wandb"):
                m._get_wandb()
