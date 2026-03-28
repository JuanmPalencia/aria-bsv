"""Tests for aria.integrations.openai — ARIAOpenAI and ARIAAsyncOpenAI wrappers."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.integrations.openai import (
    ARIAAsyncOpenAI,
    ARIAOpenAI,
    _ARIAChatCompletions,
    _ARIARecorder,
    _extract_confidence,
    _messages_to_input,
    _response_to_output,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _mock_response(content="hello", model="gpt-4o", finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.model = model
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    resp.choices[0].logprobs = None
    return resp


def _mock_response_logprobs(avg_lp: float = -0.5):
    resp = _mock_response()
    token = MagicMock()
    token.logprob = avg_lp
    resp.choices[0].logprobs = MagicMock()
    resp.choices[0].logprobs.content = [token, token]
    return resp


def _mock_recorder():
    rec = MagicMock(spec=_ARIARecorder)
    rec.model_id = None
    return rec


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestExtractConfidence:
    def test_no_logprobs(self):
        resp = _mock_response()
        assert _extract_confidence(resp) is None

    def test_with_logprobs(self):
        resp = _mock_response_logprobs(-0.2)
        c = _extract_confidence(resp)
        assert c is not None
        expected = round(math.exp(-0.2), 4)
        assert c == pytest.approx(expected)

    def test_exception_returns_none(self):
        assert _extract_confidence(None) is None
        assert _extract_confidence("bad") is None


class TestMessagesToInput:
    def test_basic(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = _messages_to_input(msgs)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "hi"

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        result = _messages_to_input(msgs)
        assert len(result["messages"][0]["content"]) == 500

    def test_empty_messages(self):
        assert _messages_to_input([]) == {"messages": []}

    def test_none_messages(self):
        assert _messages_to_input(None) == {"messages": []}


class TestResponseToOutput:
    def test_basic(self):
        resp = _mock_response("hello world")
        out = _response_to_output(resp)
        assert out["content"] == "hello world"
        assert out["finish_reason"] == "stop"
        assert out["model"] == "gpt-4o"
        assert "prompt_tokens" in out["usage"]

    def test_exception_returns_raw(self):
        out = _response_to_output(None)
        assert "raw" in out


# ---------------------------------------------------------------------------
# _ARIARecorder
# ---------------------------------------------------------------------------

class TestARIARecorder:
    def test_calls_auditor(self):
        auditor = MagicMock()
        rec = _ARIARecorder(auditor=auditor)
        rec.record(
            model_id="gpt-4",
            input_data={"messages": []},
            output_data={"content": "hi"},
            confidence=0.9,
            latency_ms=123.4,
            metadata={"provider": "openai"},
        )
        auditor.record.assert_called_once()
        args = auditor.record.call_args
        assert args[0][0] == "gpt-4"

    def test_calls_aria(self):
        aria = MagicMock()
        rec = _ARIARecorder(aria=aria)
        rec.record(
            model_id="gpt-4",
            input_data={},
            output_data={},
            latency_ms=50,
            metadata={},
        )
        aria.record.assert_called_once()

    def test_exception_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        rec = _ARIARecorder(auditor=auditor)
        # Should not raise
        rec.record(model_id="m", input_data={}, output_data={}, latency_ms=1, metadata={})


# ---------------------------------------------------------------------------
# ARIAOpenAI
# ---------------------------------------------------------------------------

class TestARIAOpenAI:
    def _build(self, auditor=None, aria=None):
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            client = ARIAOpenAI(auditor=auditor, aria=aria)
            client._client = mock_client
            return client, mock_client

    def test_init_creates_chat_embeddings(self):
        auditor = MagicMock()
        client, _ = self._build(auditor=auditor)
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")

    def test_chat_completions_create_records(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_response()
            mock_cls.return_value = mock_client
            client = ARIAOpenAI(auditor=auditor)
            client._client = mock_client
            client.chat.completions._orig = mock_client.chat.completions

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert resp is not None
            auditor.record.assert_called_once()

    def test_getattr_proxies_to_client(self):
        auditor = MagicMock()
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.api_key = "sk-test"
            mock_cls.return_value = mock_client
            client = ARIAOpenAI(auditor=auditor)
            assert client.api_key == "sk-test"

    def test_import_error(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                ARIAOpenAI(auditor=MagicMock())

    def test_model_id_override(self):
        auditor = MagicMock()
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_response()
            mock_cls.return_value = mock_client
            client = ARIAOpenAI(auditor=auditor, model_id="my-model")
            client.chat.completions._orig = mock_client.chat.completions
            client.chat.completions.create(model="gpt-4o", messages=[])
            args = auditor.record.call_args[0]
            assert args[0] == "my-model"


# ---------------------------------------------------------------------------
# ARIAOpenAI — embeddings
# ---------------------------------------------------------------------------

class TestARIAOpenAIEmbeddings:
    def test_embeddings_create_records(self):
        auditor = MagicMock()
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            embed_resp = MagicMock()
            embed_resp.data = [MagicMock()]
            embed_resp.data[0].embedding = [0.1] * 1536
            mock_client.embeddings.create.return_value = embed_resp
            mock_cls.return_value = mock_client

            client = ARIAOpenAI(auditor=auditor)
            client.embeddings._orig = mock_client.embeddings
            client.embeddings.create(model="text-embedding-3-small", input="hello")
            auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# ARIAAsyncOpenAI
# ---------------------------------------------------------------------------

class TestARIAAsyncOpenAI:
    def test_init(self):
        auditor = MagicMock()
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = ARIAAsyncOpenAI(auditor=auditor)
            assert hasattr(client, "chat")

    def test_async_create_records(self):
        auditor = MagicMock()
        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            client = ARIAAsyncOpenAI(auditor=auditor)
            client.chat.completions._orig = mock_client.chat.completions

            async def _run():
                return await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )

            resp = asyncio.get_event_loop().run_until_complete(_run())
            assert resp is not None
            auditor.record.assert_called_once()
