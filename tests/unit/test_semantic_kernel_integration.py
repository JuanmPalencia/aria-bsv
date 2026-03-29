"""Tests for aria.integrations.semantic_kernel — ARIAKernelMiddleware and ARIASemanticKernel."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake semantic_kernel module factory
# ---------------------------------------------------------------------------

def _make_sk_module() -> ModuleType:
    """Build a minimal fake ``semantic_kernel`` module."""
    mod = ModuleType("semantic_kernel")
    # Kernel constructor returns a mock kernel instance
    mod.Kernel = MagicMock(side_effect=lambda **kw: _make_mock_kernel())
    return mod


def _make_mock_kernel():
    """Build a minimal mock Kernel that supports add_filter and invoke."""
    kernel = MagicMock()
    kernel.add_filter = MagicMock()
    kernel.invoke = AsyncMock(return_value=_make_kernel_result("hello"))
    return kernel


def _make_kernel_result(value="result"):
    result = MagicMock()
    result.value = value
    return result


def _make_function_context(plugin_name="MyPlugin", fn_name="my_fn", arg_val="hello"):
    """Build a minimal mock FunctionInvocationContext."""
    fn = MagicMock()
    fn.plugin_name = plugin_name
    fn.name = fn_name

    ctx = MagicMock()
    ctx.function = fn
    ctx.arguments = {"input": arg_val}
    ctx.result = _make_kernel_result("done")
    return ctx


# ---------------------------------------------------------------------------
# Helper: reload integration module with fake semantic_kernel
# ---------------------------------------------------------------------------

def _reload_sk_integration(sk_mod=None):
    import importlib
    import aria.integrations.semantic_kernel as m
    mod = sk_mod or _make_sk_module()
    with patch.dict(sys.modules, {"semantic_kernel": mod}):
        importlib.reload(m)
        return m, mod


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestKernelArgsToInput:
    def test_mapping_converted(self):
        m, _ = _reload_sk_integration()
        result = m._kernel_args_to_input({"input": "hello", "context": "world"})
        assert result["input"] == "hello"
        assert result["context"] == "world"

    def test_none_returns_empty(self):
        m, _ = _reload_sk_integration()
        assert m._kernel_args_to_input(None) == {}

    def test_non_mapping_converted_to_str(self):
        m, _ = _reload_sk_integration()
        result = m._kernel_args_to_input("raw string")
        assert "input" in result

    def test_truncation(self):
        m, _ = _reload_sk_integration()
        result = m._kernel_args_to_input({"q": "x" * 1000})
        assert len(result["q"]) == 500


class TestKernelResultToOutput:
    def test_value_attribute_used(self):
        m, _ = _reload_sk_integration()
        result_obj = _make_kernel_result("Paris")
        out = m._kernel_result_to_output(result_obj)
        assert out["result"] == "Paris"

    def test_none_result(self):
        m, _ = _reload_sk_integration()
        out = m._kernel_result_to_output(None)
        assert out["result"] == ""

    def test_fallback_to_str(self):
        m, _ = _reload_sk_integration()
        out = m._kernel_result_to_output("plain string")
        assert "plain string" in out["result"]


# ---------------------------------------------------------------------------
# Tests for ARIAKernelMiddleware
# ---------------------------------------------------------------------------

class TestARIAKernelMiddlewareImportError:
    def test_import_error_when_sk_missing(self):
        with patch.dict(sys.modules, {"semantic_kernel": None}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="semantic-kernel"):
                m.ARIAKernelMiddleware(auditor=MagicMock())


class TestARIAKernelMiddlewareFilter:
    def _build_middleware(self, auditor=None, aria=None, model_id=None):
        sk_mod = _make_sk_module()
        with patch.dict(sys.modules, {"semantic_kernel": sk_mod}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            middleware = m.ARIAKernelMiddleware(
                auditor=auditor, aria=aria, model_id=model_id
            )
            return middleware

    def test_filter_records_to_auditor(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor)
        ctx = _make_function_context()
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        next_fn.assert_called_once_with(ctx)
        auditor.record.assert_called_once()

    def test_filter_records_to_aria(self):
        aria = MagicMock()
        middleware = self._build_middleware(aria=aria)
        ctx = _make_function_context()
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert "model_id" in kwargs

    def test_model_id_defaults_to_qualified_fn_name(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor)
        ctx = _make_function_context(plugin_name="SearchPlugin", fn_name="search")
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        args = auditor.record.call_args[0]
        assert args[0] == "SearchPlugin:search"

    def test_model_id_override(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor, model_id="my-kernel")
        ctx = _make_function_context()
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        args = auditor.record.call_args[0]
        assert args[0] == "my-kernel"

    def test_record_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("db down")
        middleware = self._build_middleware(auditor=auditor)
        ctx = _make_function_context()
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        # Must not raise
        asyncio.get_event_loop().run_until_complete(_run())

    def test_function_none_defaults_to_unknown(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor)
        ctx = MagicMock()
        ctx.function = None
        ctx.arguments = {}
        ctx.result = None
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        args = auditor.record.call_args[0]
        assert args[0] == "unknown"

    def test_arguments_captured_in_input(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor)
        ctx = _make_function_context(arg_val="world")
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        args = auditor.record.call_args[0]
        input_data = args[1]
        assert input_data.get("input") == "world"

    def test_metadata_includes_function_name(self):
        auditor = MagicMock()
        middleware = self._build_middleware(auditor=auditor)
        ctx = _make_function_context(plugin_name="P", fn_name="f")
        next_fn = AsyncMock()

        async def _run():
            await middleware.on_function_invocation(ctx, next_fn)

        asyncio.get_event_loop().run_until_complete(_run())
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["function"] == "P:f"


# ---------------------------------------------------------------------------
# Tests for ARIASemanticKernel
# ---------------------------------------------------------------------------

class TestARIASemanticKernelImportError:
    def test_import_error_when_sk_missing(self):
        with patch.dict(sys.modules, {"semantic_kernel": None}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="semantic-kernel"):
                m.ARIASemanticKernel(auditor=MagicMock())


class TestARIASemanticKernelInit:
    def test_registers_filter_on_kernel(self):
        sk_mod = _make_sk_module()
        mock_kernel = _make_mock_kernel()
        sk_mod.Kernel.side_effect = None
        sk_mod.Kernel.return_value = mock_kernel

        with patch.dict(sys.modules, {"semantic_kernel": sk_mod}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            wrapper = m.ARIASemanticKernel(auditor=MagicMock())
            mock_kernel.add_filter.assert_called_once()
            filter_type = mock_kernel.add_filter.call_args[0][0]
            assert filter_type == "function_invocation"

    def test_getattr_proxies_to_kernel(self):
        sk_mod = _make_sk_module()
        mock_kernel = _make_mock_kernel()
        mock_kernel.some_setting = "sk_value"
        sk_mod.Kernel.return_value = mock_kernel

        with patch.dict(sys.modules, {"semantic_kernel": sk_mod}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            wrapper = m.ARIASemanticKernel(auditor=MagicMock())
            assert wrapper.some_setting == "sk_value"

    def test_invoke_delegates_to_kernel(self):
        sk_mod = _make_sk_module()
        mock_kernel = _make_mock_kernel()
        mock_kernel.invoke = AsyncMock(return_value=_make_kernel_result("sk-result"))
        sk_mod.Kernel.return_value = mock_kernel

        with patch.dict(sys.modules, {"semantic_kernel": sk_mod}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            wrapper = m.ARIASemanticKernel(auditor=MagicMock())

            async def _run():
                return await wrapper.invoke("MyPlugin", "my_fn", input="test")

            result = asyncio.get_event_loop().run_until_complete(_run())
            assert result.value == "sk-result"
            mock_kernel.invoke.assert_called_once_with("MyPlugin", "my_fn", input="test")

    def test_add_filter_exception_is_swallowed(self):
        sk_mod = _make_sk_module()
        mock_kernel = _make_mock_kernel()
        mock_kernel.add_filter.side_effect = RuntimeError("filter not supported")
        sk_mod.Kernel.return_value = mock_kernel

        with patch.dict(sys.modules, {"semantic_kernel": sk_mod}):
            import importlib
            import aria.integrations.semantic_kernel as m
            importlib.reload(m)
            # Must not raise — older SK versions may not support filters
            wrapper = m.ARIASemanticKernel(auditor=MagicMock())
            assert wrapper is not None
