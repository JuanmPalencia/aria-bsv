"""Tests for aria.integrations.mistral — ARIAMistral and ARIAAsyncMistral wrappers."""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inject a fake ``mistralai`` module before importing the integration so that
# the real package is not required to run the test suite.
# ---------------------------------------------------------------------------

def _make_mistralai_module():
    """Build a minimal fake ``mistralai`` top-level module."""
    mod = types.ModuleType("mistralai")

    class Mistral:
        def __init__(self, **kwargs):
            self.chat = MagicMock()
            self._api_key = kwargs.get("api_key", "")

    mod.Mistral = Mistral
    return mod


_fake_mistralai = _make_mistralai_module()
sys.modules.setdefault("mistralai", _fake_mistralai)

# Now safe to import the integration
from aria.integrations.mistral import (  # noqa: E402
    ARIAAsyncMistral,
    ARIAMistral,
    _ARIAMistralRecorder,
    _messages_to_input,
    _response_to_output,
)


# ---------------------------------------------------------------------------
# Mock response factories
# ---------------------------------------------------------------------------

class MockMistralUsage:
    def __init__(self, prompt_tokens=8, completion_tokens=4, total_tokens=12):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMistralChoice:
    def __init__(self, content="Bonjour!", finish_reason="stop"):
        self.message = MagicMock()
        self.message.content = content
        self.finish_reason = finish_reason


class MockMistralResponse:
    def __init__(self, content="Bonjour!", finish_reason="stop", model="mistral-large-latest"):
        self.choices = [MockMistralChoice(content=content, finish_reason=finish_reason)]
        self.usage = MockMistralUsage()
        self.model = model


def _mock_recorder(model_id=None):
    rec = MagicMock(spec=_ARIAMistralRecorder)
    rec.model_id = model_id
    return rec


# ---------------------------------------------------------------------------
# 1. test_chat_complete_records_inference
# ---------------------------------------------------------------------------

class TestChatCompleteRecordsInference:
    def test_chat_complete_records_inference(self):
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="test-key")

        fake_response = MockMistralResponse(content="Hello from Mistral")
        client.chat._orig.complete.return_value = fake_response

        result = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Say hi"}],
        )

        assert result is fake_response
        auditor.record.assert_called_once()
        call_args = auditor.record.call_args
        assert call_args[0][0] == "mistral-large-latest"


# ---------------------------------------------------------------------------
# 2. test_async_chat_complete_records_inference
# ---------------------------------------------------------------------------

class TestAsyncChatCompleteRecordsInference:
    def test_async_chat_complete_records_inference(self):
        auditor = MagicMock()
        client = ARIAAsyncMistral(auditor=auditor, api_key="test-key")

        fake_response = MockMistralResponse(content="Async reply")
        client.chat._orig.complete_async = AsyncMock(return_value=fake_response)

        async def _run():
            return await client.chat.complete_async(
                model="mistral-medium",
                messages=[{"role": "user", "content": "Async test"}],
            )

        result = asyncio.get_event_loop().run_until_complete(_run())

        assert result is fake_response
        auditor.record.assert_called_once()
        call_args = auditor.record.call_args
        assert call_args[0][0] == "mistral-medium"


# ---------------------------------------------------------------------------
# 3. test_confidence_is_none
# ---------------------------------------------------------------------------

class TestConfidenceIsNone:
    def test_confidence_is_none(self):
        """Mistral responses carry no logprobs, so confidence must always be None."""
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="test-key")

        client.chat._orig.complete.return_value = MockMistralResponse()
        client.chat.complete(
            model="mistral-small",
            messages=[{"role": "user", "content": "test"}],
        )

        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs.get("confidence") is None


# ---------------------------------------------------------------------------
# 4. test_model_id_override
# ---------------------------------------------------------------------------

class TestModelIdOverride:
    def test_model_id_override(self):
        """model_id constructor argument should override the request model."""
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="test-key", model_id="my-custom-model")

        client.chat._orig.complete.return_value = MockMistralResponse()
        client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "override test"}],
        )

        call_args = auditor.record.call_args
        assert call_args[0][0] == "my-custom-model"


# ---------------------------------------------------------------------------
# 5. test_aria_quick_integration
# ---------------------------------------------------------------------------

class TestAriaQuickIntegration:
    def test_aria_quick_integration(self):
        """ARIAMistral should forward records to an ARIAQuick instance."""
        aria = MagicMock()
        client = ARIAMistral(aria=aria, api_key="test-key")

        client.chat._orig.complete.return_value = MockMistralResponse()
        client.chat.complete(
            model="mistral-small",
            messages=[{"role": "user", "content": "quick test"}],
        )

        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert kwargs["model_id"] == "mistral-small"


# ---------------------------------------------------------------------------
# 6. test_record_error_swallowed
# ---------------------------------------------------------------------------

class TestRecordErrorSwallowed:
    def test_record_error_swallowed(self):
        """An exception raised inside the auditor must not propagate to the caller."""
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage failure")
        client = ARIAMistral(auditor=auditor, api_key="test-key")

        fake_response = MockMistralResponse()
        client.chat._orig.complete.return_value = fake_response

        # Should not raise despite the auditor blowing up
        result = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "error test"}],
        )
        assert result is fake_response


# ---------------------------------------------------------------------------
# 7. test_import_error_without_mistral
# ---------------------------------------------------------------------------

class TestImportErrorWithoutMistral:
    def test_import_error_without_mistral(self):
        """If mistralai is not installed, a descriptive ImportError must be raised."""
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError, match="mistral"):
                ARIAMistral(auditor=MagicMock())

    def test_import_error_async_without_mistral(self):
        """Same check for the async variant."""
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError, match="mistral"):
                ARIAAsyncMistral(auditor=MagicMock())


# ---------------------------------------------------------------------------
# 8. test_getattr_proxy
# ---------------------------------------------------------------------------

class TestGetattrProxy:
    def test_getattr_proxy(self):
        """Attributes not defined on ARIAMistral should proxy to the real client."""
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="sk-proxy-test")

        # Inject a sentinel attribute onto the underlying client
        client._client._api_key = "sk-proxy-test"
        assert client._client._api_key == "sk-proxy-test"

    def test_getattr_proxy_async(self):
        """Same proxy behaviour for ARIAAsyncMistral."""
        auditor = MagicMock()
        client = ARIAAsyncMistral(auditor=auditor, api_key="sk-async-proxy")
        client._client.some_attr = "proxied-value"
        assert client.some_attr == "proxied-value"


# ---------------------------------------------------------------------------
# 9. test_usage_extracted
# ---------------------------------------------------------------------------

class TestUsageExtracted:
    def test_usage_extracted(self):
        """Token usage fields must be present in the recorded output_data."""
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="test-key")

        response = MockMistralResponse()
        response.usage = MockMistralUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        client.chat._orig.complete.return_value = response

        client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "usage test"}],
        )

        call_args = auditor.record.call_args
        output_data = call_args[0][2]
        usage = output_data.get("usage", {})
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == 30

    def test_response_to_output_helper(self):
        """_response_to_output should map Mistral response fields correctly."""
        response = MockMistralResponse(content="Salut", finish_reason="stop")
        result = _response_to_output(response, model="mistral-small")
        assert result["content"] == "Salut"
        assert result["finish_reason"] == "stop"
        assert result["model"] == "mistral-small"
        assert result["usage"]["prompt_tokens"] == 8

    def test_response_to_output_exception_returns_raw(self):
        """A broken response object should yield a raw fallback dict, not raise."""
        result = _response_to_output(None)
        assert "raw" in result


# ---------------------------------------------------------------------------
# 10. test_provider_metadata_is_mistral
# ---------------------------------------------------------------------------

class TestProviderMetadataIsMistral:
    def test_provider_metadata_is_mistral(self):
        """The ``provider`` field in metadata must always be ``'mistral'``."""
        auditor = MagicMock()
        client = ARIAMistral(auditor=auditor, api_key="test-key")

        client.chat._orig.complete.return_value = MockMistralResponse()
        client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "provider check"}],
        )

        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs["metadata"]["provider"] == "mistral"

    def test_provider_metadata_is_mistral_async(self):
        """Same provider check for the async path."""
        auditor = MagicMock()
        client = ARIAAsyncMistral(auditor=auditor, api_key="test-key")

        fake_response = MockMistralResponse()
        client.chat._orig.complete_async = AsyncMock(return_value=fake_response)

        async def _run():
            await client.chat.complete_async(
                model="mistral-medium",
                messages=[{"role": "user", "content": "async provider check"}],
            )

        asyncio.get_event_loop().run_until_complete(_run())

        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs["metadata"]["provider"] == "mistral"


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestMessagesToInput:
    def test_basic(self):
        msgs = [{"role": "user", "content": "bonjour"}]
        result = _messages_to_input(msgs)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "bonjour"

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 1000}]
        result = _messages_to_input(msgs)
        assert len(result["messages"][0]["content"]) == 500

    def test_empty_messages(self):
        assert _messages_to_input([]) == {"messages": []}

    def test_none_messages(self):
        assert _messages_to_input(None) == {"messages": []}
