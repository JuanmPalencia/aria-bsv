"""Tests for aria.integrations.anthropic_sdk — ARIAAnthropic and ARIAAsyncAnthropic."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — inject a fake `anthropic` module so imports work
# ---------------------------------------------------------------------------

def _make_anthropic_module():
    """Build a minimal fake `anthropic` module."""
    mod = ModuleType("anthropic")
    mod.Anthropic = MagicMock
    mod.AsyncAnthropic = MagicMock
    return mod


def _mock_response(text="Hello!", stop_reason="end_turn", model="claude-opus-4-6"):
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    resp.stop_reason = stop_reason
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.input_tokens = 12
    resp.usage.output_tokens = 8
    return resp


# ---------------------------------------------------------------------------
# Pure helpers — these don't need anthropic installed
# ---------------------------------------------------------------------------

def _import_helpers():
    """Import helper functions, injecting fake anthropic if needed."""
    fake_mod = _make_anthropic_module()
    with patch.dict(sys.modules, {"anthropic": fake_mod}):
        from aria.integrations.anthropic_sdk import (
            _messages_to_input,
            _response_to_output,
            _ARIARecorder,
        )
        return _messages_to_input, _response_to_output, _ARIARecorder


class TestMessagesToInput:
    def setup_method(self):
        self._messages_to_input, _, _ = _import_helpers()

    def test_basic(self):
        msgs = [{"role": "user", "content": "hi there"}]
        result = self._messages_to_input(msgs, None)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert "system" not in result

    def test_with_system(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = self._messages_to_input(msgs, "You are helpful.")
        assert result["system"] == "You are helpful."

    def test_truncates_content(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        result = self._messages_to_input(msgs, None)
        assert len(result["messages"][0]["content"]) == 500

    def test_truncates_system(self):
        result = self._messages_to_input([], "y" * 1000)
        assert len(result["system"]) == 500

    def test_empty_messages(self):
        result = self._messages_to_input([], None)
        assert result["messages"] == []


class TestResponseToOutput:
    def setup_method(self):
        _, self._response_to_output, _ = _import_helpers()

    def test_basic(self):
        resp = _mock_response("world")
        out = self._response_to_output(resp)
        assert out["content"] == "world"
        assert out["stop_reason"] == "end_turn"
        assert out["model"] == "claude-opus-4-6"
        assert out["usage"]["input_tokens"] == 12

    def test_exception_returns_raw(self):
        out = self._response_to_output(None)
        assert "raw" in out


# ---------------------------------------------------------------------------
# _ARIARecorder
# ---------------------------------------------------------------------------

class TestARIARecorder:
    def setup_method(self):
        _, _, self._ARIARecorder = _import_helpers()

    def test_calls_auditor(self):
        auditor = MagicMock()
        rec = self._ARIARecorder(auditor=auditor)
        rec.record(
            model_id="claude-opus-4-6",
            input_data={"messages": []},
            output_data={"content": "hi"},
            confidence=None,
            latency_ms=200,
            metadata={"provider": "anthropic"},
        )
        auditor.record.assert_called_once()

    def test_calls_aria(self):
        aria = MagicMock()
        rec = self._ARIARecorder(aria=aria)
        rec.record(
            model_id="claude",
            input_data={},
            output_data={},
            latency_ms=100,
            metadata={},
        )
        aria.record.assert_called_once()

    def test_exception_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = Exception("oops")
        rec = self._ARIARecorder(auditor=auditor)
        rec.record(model_id="m", input_data={}, output_data={}, latency_ms=1, metadata={})

    def test_model_id_attribute(self):
        rec = self._ARIARecorder(model_id="my-model")
        assert rec.model_id == "my-model"


# ---------------------------------------------------------------------------
# Helper: build ARIAAnthropic with mocked module
# ---------------------------------------------------------------------------

def _build_aria_anthropic(auditor=None, aria=None, model_id=None, **client_returns):
    """Import and instantiate ARIAAnthropic with a mocked anthropic module."""
    fake_mod = _make_anthropic_module()
    mock_client_instance = MagicMock()
    for attr, val in client_returns.items():
        setattr(mock_client_instance, attr, val)
    fake_mod.Anthropic = MagicMock(return_value=mock_client_instance)

    with patch.dict(sys.modules, {"anthropic": fake_mod}):
        from aria.integrations import anthropic_sdk
        # Reload to pick up fresh module state
        import importlib
        importlib.reload(anthropic_sdk)
        client = anthropic_sdk.ARIAAnthropic(auditor=auditor, aria=aria, model_id=model_id)

    client._client = mock_client_instance
    return client, mock_client_instance


# ---------------------------------------------------------------------------
# ARIAAnthropic
# ---------------------------------------------------------------------------

class TestARIAAnthropic:
    def test_init_import_error(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            import importlib
            import aria.integrations.anthropic_sdk as _mod
            importlib.reload(_mod)
            with pytest.raises(ImportError, match="anthropic"):
                _mod.ARIAAnthropic(auditor=MagicMock())

    def test_messages_create_records(self):
        auditor = MagicMock()
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response()
        fake_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAnthropic(auditor=auditor)
            client.messages._orig = mock_client.messages
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert resp is not None
        auditor.record.assert_called_once()

    def test_model_id_from_request(self):
        auditor = MagicMock()
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response()
        fake_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAnthropic(auditor=auditor)
            client.messages._orig = mock_client.messages
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[],
            )
        args = auditor.record.call_args[0]
        assert args[0] == "claude-haiku-4-5-20251001"

    def test_model_id_override(self):
        auditor = MagicMock()
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response()
        fake_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAnthropic(auditor=auditor, model_id="my-prod-claude")
            client.messages._orig = mock_client.messages
            client.messages.create(model="claude-opus-4-6", max_tokens=100, messages=[])
        args = auditor.record.call_args[0]
        assert args[0] == "my-prod-claude"

    def test_system_prompt_captured(self):
        auditor = MagicMock()
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response()
        fake_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAnthropic(auditor=auditor)
            client.messages._orig = mock_client.messages
            client.messages.create(
                model="claude-opus-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                system="You are a test assistant",
            )
        args = auditor.record.call_args[0]
        assert "system" in args[1]

    def test_getattr_proxies(self):
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.beta = "beta-obj"
        fake_mod.Anthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAnthropic(auditor=MagicMock())
        assert client.beta == "beta-obj"


# ---------------------------------------------------------------------------
# ARIAAsyncAnthropic
# ---------------------------------------------------------------------------

class TestARIAAsyncAnthropic:
    def test_init(self):
        fake_mod = _make_anthropic_module()
        fake_mod.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAsyncAnthropic(auditor=MagicMock())
        assert hasattr(client, "messages")

    def test_async_create_records(self):
        auditor = MagicMock()
        fake_mod = _make_anthropic_module()
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_mock_response())
        fake_mod.AsyncAnthropic = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"anthropic": fake_mod}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            client = m.ARIAAsyncAnthropic(auditor=auditor)
            client.messages._orig = mock_client.messages

            async def _run():
                return await client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}],
                )

            resp = asyncio.get_event_loop().run_until_complete(_run())
        assert resp is not None
        auditor.record.assert_called_once()

    def test_import_error(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            import importlib
            import aria.integrations.anthropic_sdk as m
            importlib.reload(m)
            with pytest.raises(ImportError):
                m.ARIAAsyncAnthropic(auditor=MagicMock())
