"""Tests for aria.integrations.litellm — ARIALiteLLM and make_litellm_callback."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake litellm module factory
# ---------------------------------------------------------------------------

def _make_litellm_module(response: MagicMock | None = None) -> ModuleType:
    """Build a minimal fake ``litellm`` module."""
    mod = ModuleType("litellm")
    default_response = response or _mock_response()
    mod.completion = MagicMock(return_value=default_response)
    mod.acompletion = AsyncMock(return_value=default_response)
    return mod


def _mock_response(content="hello from litellm", model="gpt-4o", finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


# ---------------------------------------------------------------------------
# Helper: build ARIALiteLLM with injected fake litellm
# ---------------------------------------------------------------------------

def _build_client(auditor=None, aria=None, model_id=None, litellm_mod=None):
    """Import and instantiate ARIALiteLLM with a fake litellm module."""
    mod = litellm_mod or _make_litellm_module()
    with patch.dict(sys.modules, {"litellm": mod}):
        import importlib
        import aria.integrations.litellm as m
        importlib.reload(m)
        client = m.ARIALiteLLM(auditor=auditor, aria=aria, model_id=model_id)
        client._litellm = mod
        return client, m


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestMessagesToInput:
    def test_basic_conversion(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            msgs = [{"role": "user", "content": "hello"}]
            result = m._messages_to_input(msgs)
            assert result["messages"][0]["role"] == "user"
            assert result["messages"][0]["content"] == "hello"

    def test_truncates_long_content(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            msgs = [{"role": "user", "content": "x" * 1000}]
            result = m._messages_to_input(msgs)
            assert len(result["messages"][0]["content"]) == 500

    def test_empty_list(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            assert m._messages_to_input([]) == {"messages": []}

    def test_none_returns_empty(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            assert m._messages_to_input(None) == {"messages": []}


class TestResponseToOutput:
    def test_basic_fields(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            resp = _mock_response("hi", "gpt-4o", "stop")
            out = m._response_to_output(resp)
            assert out["content"] == "hi"
            assert out["finish_reason"] == "stop"
            assert out["model"] == "gpt-4o"
            assert "prompt_tokens" in out["usage"]

    def test_exception_returns_raw(self):
        with patch.dict(sys.modules, {"litellm": _make_litellm_module()}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            out = m._response_to_output(None)
            assert "raw" in out


# ---------------------------------------------------------------------------
# Tests for ARIALiteLLM
# ---------------------------------------------------------------------------

class TestARIALiteLLMInit:
    def test_import_error_when_litellm_missing(self):
        with patch.dict(sys.modules, {"litellm": None}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="litellm"):
                m.ARIALiteLLM(auditor=MagicMock())

    def test_init_stores_default_kwargs(self):
        mod = _make_litellm_module()
        with patch.dict(sys.modules, {"litellm": mod}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            client = m.ARIALiteLLM(auditor=MagicMock(), temperature=0.7)
            client._litellm = mod
            assert client._default_kwargs.get("temperature") == 0.7


class TestARIALiteLLMCompletion:
    def test_completion_calls_litellm(self):
        auditor = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, litellm_mod=mod)
        messages = [{"role": "user", "content": "hello"}]
        resp = client.completion(model="gpt-4o", messages=messages)
        assert resp is not None
        mod.completion.assert_called_once()

    def test_completion_records_to_auditor(self):
        auditor = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, litellm_mod=mod)
        client.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
        )
        auditor.record.assert_called_once()
        args = auditor.record.call_args[0]
        assert args[0] == "gpt-4o"

    def test_completion_records_to_aria(self):
        aria = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(aria=aria, litellm_mod=mod)
        client.completion(model="claude-3-opus", messages=[])
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert kwargs["model_id"] == "claude-3-opus"

    def test_model_id_override(self):
        auditor = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, model_id="my-model", litellm_mod=mod)
        client.completion(model="gpt-4o", messages=[])
        args = auditor.record.call_args[0]
        assert args[0] == "my-model"

    def test_record_error_is_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage error")
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, litellm_mod=mod)
        # Must not raise
        result = client.completion(model="gpt-4o", messages=[])
        assert result is not None


class TestARIALiteLLMACompletion:
    def test_acompletion_calls_litellm(self):
        auditor = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, litellm_mod=mod)

        async def _run():
            return await client.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "async hello"}],
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result is not None
        mod.acompletion.assert_called_once()
        auditor.record.assert_called_once()

    def test_acompletion_model_id_override(self):
        auditor = MagicMock()
        mod = _make_litellm_module()
        client, _ = _build_client(auditor=auditor, model_id="async-model", litellm_mod=mod)

        async def _run():
            await client.acompletion(model="gpt-4o", messages=[])

        asyncio.get_event_loop().run_until_complete(_run())
        args = auditor.record.call_args[0]
        assert args[0] == "async-model"


# ---------------------------------------------------------------------------
# Tests for make_litellm_callback
# ---------------------------------------------------------------------------

class TestMakeLiteLLMCallback:
    def _make_datetime_delta(self, seconds=1.5):
        from datetime import timedelta
        return timedelta(seconds=seconds)

    def test_callback_records_to_auditor(self):
        from datetime import datetime, timedelta

        auditor = MagicMock()
        mod = _make_litellm_module()
        with patch.dict(sys.modules, {"litellm": mod}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            cb = m.make_litellm_callback(auditor=auditor)
            kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
            start = datetime.now()
            end = start + timedelta(seconds=2)
            cb(kwargs, _mock_response(), start, end)
            auditor.record.assert_called_once()

    def test_callback_uses_model_id_override(self):
        from datetime import datetime, timedelta

        auditor = MagicMock()
        mod = _make_litellm_module()
        with patch.dict(sys.modules, {"litellm": mod}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            cb = m.make_litellm_callback(auditor=auditor, model_id="override-id")
            start = datetime.now()
            end = start + timedelta(seconds=1)
            cb({"model": "gpt-4o", "messages": []}, _mock_response(), start, end)
            args = auditor.record.call_args[0]
            assert args[0] == "override-id"

    def test_callback_swallows_record_error(self):
        from datetime import datetime, timedelta

        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        mod = _make_litellm_module()
        with patch.dict(sys.modules, {"litellm": mod}):
            import importlib
            import aria.integrations.litellm as m
            importlib.reload(m)
            cb = m.make_litellm_callback(auditor=auditor)
            start = datetime.now()
            end = start + timedelta(seconds=1)
            # Must not raise
            cb({"model": "gpt-4o", "messages": []}, _mock_response(), start, end)
