"""Tests for aria.integrations.smolagents — ARIASmolAgent and ARIAToolWrapper."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake smolagents module factory
# ---------------------------------------------------------------------------

def _make_smolagents_module() -> ModuleType:
    """Build a minimal fake ``smolagents`` module."""
    mod = ModuleType("smolagents")
    mod.CodeAgent = object
    mod.ToolCallingAgent = object
    return mod


# ---------------------------------------------------------------------------
# Mock agent and tool factories
# ---------------------------------------------------------------------------

class MockAgent:
    """Minimal mock of a smolagents CodeAgent."""

    def __init__(self, answer="42"):
        self._answer = answer
        self.name = "mock_agent"

    def run(self, task, **kwargs):
        return self._answer


class MockAsyncAgent(MockAgent):
    async def arun(self, task, **kwargs):
        return self._answer


class MockTool:
    """Minimal mock of a smolagents tool."""

    def __init__(self, name="calculator", result="100"):
        self.name = name
        self._result = result

    def __call__(self, *args, **kwargs):
        return self._result


class MockAsyncTool(MockTool):
    async def acall(self, *args, **kwargs):
        return self._result


# ---------------------------------------------------------------------------
# Helper: reload module with fake smolagents
# ---------------------------------------------------------------------------

def _reload_smolagents(smolagents_mod=None):
    import importlib
    import aria.integrations.smolagents as m
    mod = smolagents_mod or _make_smolagents_module()
    with patch.dict(sys.modules, {"smolagents": mod}):
        importlib.reload(m)
        return m, mod


def _build_agent_wrapper(auditor=None, aria=None, model_id=None, agent=None):
    mod = _make_smolagents_module()
    with patch.dict(sys.modules, {"smolagents": mod}):
        import importlib
        import aria.integrations.smolagents as m
        importlib.reload(m)
        inner = agent or MockAgent()
        wrapper = m.ARIASmolAgent(
            agent=inner, auditor=auditor, aria=aria, model_id=model_id
        )
        return wrapper


def _build_tool_wrapper(auditor=None, aria=None, model_id=None, tool=None):
    mod = _make_smolagents_module()
    with patch.dict(sys.modules, {"smolagents": mod}):
        import importlib
        import aria.integrations.smolagents as m
        importlib.reload(m)
        inner = tool or MockTool()
        wrapper = m.ARIAToolWrapper(
            tool=inner, auditor=auditor, aria=aria, model_id=model_id
        )
        return wrapper


# ---------------------------------------------------------------------------
# Tests: import error handling
# ---------------------------------------------------------------------------

class TestImportErrors:
    def test_agent_import_error(self):
        with patch.dict(sys.modules, {"smolagents": None}):
            import importlib
            import aria.integrations.smolagents as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="smolagents"):
                m.ARIASmolAgent(agent=MockAgent(), auditor=MagicMock())

    def test_tool_import_error(self):
        with patch.dict(sys.modules, {"smolagents": None}):
            import importlib
            import aria.integrations.smolagents as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="smolagents"):
                m.ARIAToolWrapper(tool=MockTool(), auditor=MagicMock())


# ---------------------------------------------------------------------------
# Tests: ARIASmolAgent
# ---------------------------------------------------------------------------

class TestARIASmolAgentRun:
    def test_run_returns_answer(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor, agent=MockAgent("Paris"))
        result = wrapper.run("What is the capital of France?")
        assert result == "Paris"

    def test_run_records_to_auditor(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor)
        wrapper.run("Tell me a fact")
        auditor.record.assert_called_once()

    def test_run_records_to_aria(self):
        aria = MagicMock()
        wrapper = _build_agent_wrapper(aria=aria)
        wrapper.run("Summarise this")
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert "model_id" in kwargs

    def test_model_id_defaults_to_agent_class_name(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor)
        wrapper.run("task")
        args = auditor.record.call_args[0]
        assert args[0] == "MockAgent"

    def test_model_id_override(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor, model_id="prod-agent")
        wrapper.run("task")
        args = auditor.record.call_args[0]
        assert args[0] == "prod-agent"

    def test_record_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("db down")
        wrapper = _build_agent_wrapper(auditor=auditor)
        result = wrapper.run("task")
        assert result is not None

    def test_task_truncated_in_input(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor)
        wrapper.run("a" * 1000)
        args = auditor.record.call_args[0]
        assert len(args[1]["task"]) == 500

    def test_getattr_proxies_to_inner_agent(self):
        mod = _make_smolagents_module()
        with patch.dict(sys.modules, {"smolagents": mod}):
            import importlib
            import aria.integrations.smolagents as m
            importlib.reload(m)
            inner = MockAgent()
            inner.special_setting = "xyz"
            wrapper = m.ARIASmolAgent(agent=inner, auditor=MagicMock())
            assert wrapper.special_setting == "xyz"

    def test_metadata_includes_agent_type(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor)
        wrapper.run("task")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["agent_type"] == "MockAgent"


class TestARIASmolAgentArun:
    def test_arun_uses_native_arun_when_available(self):
        auditor = MagicMock()
        wrapper = _build_agent_wrapper(auditor=auditor, agent=MockAsyncAgent("async answer"))

        async def _run():
            return await wrapper.arun("async task")

        result = asyncio.run(_run())
        assert result == "async answer"
        auditor.record.assert_called_once()

    def test_arun_falls_back_to_sync_executor(self):
        auditor = MagicMock()
        sync_agent = MockAgent("sync fallback")
        wrapper = _build_agent_wrapper(auditor=auditor, agent=sync_agent)

        async def _run():
            return await wrapper.arun("task")

        result = asyncio.run(_run())
        assert result == "sync fallback"
        auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: ARIAToolWrapper
# ---------------------------------------------------------------------------

class TestARIAToolWrapperCall:
    def test_call_returns_result(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor, tool=MockTool(result="200"))
        result = wrapper("2 * 100")
        assert result == "200"

    def test_call_records_to_auditor(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor)
        wrapper("input")
        auditor.record.assert_called_once()

    def test_model_id_from_tool_name(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor, tool=MockTool(name="web_search"))
        wrapper("query")
        args = auditor.record.call_args[0]
        assert args[0] == "web_search"

    def test_model_id_override(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor, model_id="my-tool")
        wrapper("input")
        args = auditor.record.call_args[0]
        assert args[0] == "my-tool"

    def test_kwargs_recorded_as_input(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor)
        wrapper(query="hello")
        args = auditor.record.call_args[0]
        assert "query" in args[1]

    def test_record_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        wrapper = _build_tool_wrapper(auditor=auditor)
        result = wrapper("input")
        assert result is not None

    def test_getattr_proxies_to_tool(self):
        mod = _make_smolagents_module()
        with patch.dict(sys.modules, {"smolagents": mod}):
            import importlib
            import aria.integrations.smolagents as m
            importlib.reload(m)
            inner = MockTool()
            inner.description = "A calculator tool"
            wrapper = m.ARIAToolWrapper(tool=inner, auditor=MagicMock())
            assert wrapper.description == "A calculator tool"


class TestARIAToolWrapperAcall:
    def test_acall_uses_native_acall(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor, tool=MockAsyncTool(result="async_result"))

        async def _run():
            return await wrapper.acall("input")

        result = asyncio.run(_run())
        assert result == "async_result"
        auditor.record.assert_called_once()

    def test_acall_falls_back_to_sync_executor(self):
        auditor = MagicMock()
        wrapper = _build_tool_wrapper(auditor=auditor, tool=MockTool(result="sync_result"))

        async def _run():
            return await wrapper.acall("input")

        result = asyncio.run(_run())
        assert result == "sync_result"
        auditor.record.assert_called_once()
