"""Tests for aria.integrations.dspy — ARIADSPyModule and ARIADSPyOptimizer."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake dspy module factory
# ---------------------------------------------------------------------------

def _make_dspy_module() -> ModuleType:
    """Build a minimal fake ``dspy`` module."""
    mod = ModuleType("dspy")
    mod.Module = object  # base class placeholder
    return mod


# ---------------------------------------------------------------------------
# Mock DSPy module / prediction factories
# ---------------------------------------------------------------------------

class MockPrediction:
    """Mimics dspy.Prediction with a toDict() method."""

    def __init__(self, answer="42"):
        self.answer = answer

    def toDict(self):
        return {"answer": self.answer}


class MockDSPyModule:
    """Mimics a compiled or uncompiled dspy.Module."""

    def __init__(self, return_value=None, compiled=False):
        self._return_value = return_value or MockPrediction()
        self._compiled = compiled

    def __call__(self, *args, **kwargs):
        return self._return_value


class MockOptimizer:
    """Mimics a DSPy optimizer (e.g. BootstrapFewShot)."""

    def compile(self, student, **kwargs):
        compiled = MockDSPyModule(compiled=True)
        compiled._compiled = True
        return compiled


# ---------------------------------------------------------------------------
# Helper: import module under fake dspy
# ---------------------------------------------------------------------------

def _reload_dspy_integration(dspy_mod=None):
    import importlib
    import aria.integrations.dspy as m
    mod = dspy_mod or _make_dspy_module()
    with patch.dict(sys.modules, {"dspy": mod}):
        importlib.reload(m)
        return m, mod


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestPredictionToOutput:
    def test_todict_used_when_available(self):
        m, _ = _reload_dspy_integration()
        with patch.dict(sys.modules, {"dspy": _make_dspy_module()}):
            import importlib
            importlib.reload(m)
            pred = MockPrediction("Paris")
            out = m._prediction_to_output(pred)
            assert out["answer"] == "Paris"

    def test_fallback_to_str(self):
        m, _ = _reload_dspy_integration()
        out = m._prediction_to_output("plain string")
        assert "plain string" in out.get("result", "")

    def test_exception_returns_result_key(self):
        m, _ = _reload_dspy_integration()
        out = m._prediction_to_output(None)
        assert "result" in out


class TestKwargsToInput:
    def test_basic(self):
        m, _ = _reload_dspy_integration()
        result = m._kwargs_to_input({"question": "What is BSV?"})
        assert result["question"] == "What is BSV?"

    def test_truncates_long_values(self):
        m, _ = _reload_dspy_integration()
        result = m._kwargs_to_input({"q": "x" * 1000})
        assert len(result["q"]) == 500


# ---------------------------------------------------------------------------
# Tests for ARIADSPyModule
# ---------------------------------------------------------------------------

class TestARIADSPyModuleImportError:
    def test_import_error_when_dspy_missing(self):
        with patch.dict(sys.modules, {"dspy": None}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="dspy-ai"):
                m.ARIADSPyModule(module=MockDSPyModule(), auditor=MagicMock())


class TestARIADSPyModuleForward:
    def _build(self, auditor=None, aria=None, model_id=None, compiled=False):
        dspy_mod = _make_dspy_module()
        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            inner = MockDSPyModule(compiled=compiled)
            wrapper = m.ARIADSPyModule(
                module=inner,
                auditor=auditor,
                aria=aria,
                model_id=model_id,
            )
            return wrapper

    def test_forward_returns_prediction(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor)
        result = wrapper(question="What is BSV?")
        assert isinstance(result, MockPrediction)

    def test_forward_records_to_auditor(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor)
        wrapper(question="Hello")
        auditor.record.assert_called_once()

    def test_forward_records_to_aria(self):
        aria = MagicMock()
        wrapper = self._build(aria=aria)
        wrapper(question="Test")
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert "model_id" in kwargs

    def test_model_id_defaults_to_class_name(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor)
        wrapper(question="Q")
        args = auditor.record.call_args[0]
        assert args[0] == "MockDSPyModule"

    def test_model_id_override(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor, model_id="my-dspy-prog")
        wrapper(question="Q")
        args = auditor.record.call_args[0]
        assert args[0] == "my-dspy-prog"

    def test_call_proxies_to_forward(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor)
        result = wrapper(question="via __call__")
        assert isinstance(result, MockPrediction)
        auditor.record.assert_called_once()

    def test_record_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        wrapper = self._build(auditor=auditor)
        # Must not raise
        result = wrapper(question="Q")
        assert result is not None

    def test_compiled_flag_in_metadata(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor, compiled=True)
        wrapper(question="Q")
        kwargs = auditor.record.call_args[1]
        meta = kwargs.get("metadata", {})
        assert "compiled" in meta

    def test_getattr_proxies_to_inner_module(self):
        dspy_mod = _make_dspy_module()
        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            inner = MockDSPyModule()
            inner.custom_attr = "special"
            wrapper = m.ARIADSPyModule(module=inner, auditor=MagicMock())
            assert wrapper.custom_attr == "special"

    def test_input_data_contains_kwargs(self):
        auditor = MagicMock()
        wrapper = self._build(auditor=auditor)
        wrapper(question="BSV rocks")
        args = auditor.record.call_args[0]
        input_data = args[1]
        assert "question" in input_data
        assert "BSV rocks" in input_data["question"]


# ---------------------------------------------------------------------------
# Tests for ARIADSPyOptimizer
# ---------------------------------------------------------------------------

class TestARIADSPyOptimizerImportError:
    def test_import_error_when_dspy_missing(self):
        with patch.dict(sys.modules, {"dspy": None}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="dspy-ai"):
                m.ARIADSPyOptimizer(optimizer=MockOptimizer(), auditor=MagicMock())


class TestARIADSPyOptimizerCompile:
    def _build(self, auditor=None, model_id=None):
        dspy_mod = _make_dspy_module()
        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            wrapper = m.ARIADSPyOptimizer(
                optimizer=MockOptimizer(),
                auditor=auditor,
                model_id=model_id,
            )
            return wrapper

    def test_compile_returns_compiled_module(self):
        auditor = MagicMock()
        opt = self._build(auditor=auditor)
        result = opt.compile(MockDSPyModule())
        assert isinstance(result, MockDSPyModule)
        assert getattr(result, "_compiled", False)

    def test_compile_records_to_auditor(self):
        auditor = MagicMock()
        opt = self._build(auditor=auditor)
        opt.compile(MockDSPyModule())
        auditor.record.assert_called_once()

    def test_model_id_defaults_to_optimizer_class_name(self):
        auditor = MagicMock()
        opt = self._build(auditor=auditor)
        opt.compile(MockDSPyModule())
        args = auditor.record.call_args[0]
        assert args[0] == "MockOptimizer"

    def test_model_id_override(self):
        auditor = MagicMock()
        opt = self._build(auditor=auditor, model_id="opt-run")
        opt.compile(MockDSPyModule())
        args = auditor.record.call_args[0]
        assert args[0] == "opt-run"

    def test_getattr_proxies_to_optimizer(self):
        dspy_mod = _make_dspy_module()
        with patch.dict(sys.modules, {"dspy": dspy_mod}):
            import importlib
            import aria.integrations.dspy as m
            importlib.reload(m)
            inner = MockOptimizer()
            inner.custom_setting = 99
            wrapper = m.ARIADSPyOptimizer(optimizer=inner, auditor=MagicMock())
            assert wrapper.custom_setting == 99
