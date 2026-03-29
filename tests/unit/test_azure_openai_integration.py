"""Tests for aria.integrations.azure_openai — ARIAAzureOpenAI and ARIAAsyncAzureOpenAI."""

from __future__ import annotations

import asyncio
import importlib
import math
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — inject a fake `openai` module so imports work without the package
# ---------------------------------------------------------------------------

def _make_openai_module():
    """Build a minimal fake ``openai`` module with AzureOpenAI and AsyncAzureOpenAI."""
    mod = ModuleType("openai")
    mod.AzureOpenAI = MagicMock
    mod.AsyncAzureOpenAI = MagicMock
    # Also expose OpenAI/AsyncOpenAI so reloading the base openai integration
    # (if triggered transitively) does not fail.
    mod.OpenAI = MagicMock
    mod.AsyncOpenAI = MagicMock
    return mod


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


# ---------------------------------------------------------------------------
# Helper: reload the module under test with a fake openai present
# ---------------------------------------------------------------------------

def _reload_module():
    fake_mod = _make_openai_module()
    with patch.dict(sys.modules, {"openai": fake_mod}):
        import aria.integrations.azure_openai as m
        importlib.reload(m)
        return m, fake_mod


def _build_client(auditor=None, aria=None, model_id=None, **extra):
    """Return (ARIAAzureOpenAI instance, underlying mock_client)."""
    fake_mod = _make_openai_module()
    mock_client_instance = MagicMock()
    fake_mod.AzureOpenAI = MagicMock(return_value=mock_client_instance)

    with patch.dict(sys.modules, {"openai": fake_mod}):
        import aria.integrations.azure_openai as m
        importlib.reload(m)
        client = m.ARIAAzureOpenAI(
            auditor=auditor,
            aria=aria,
            model_id=model_id,
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
            **extra,
        )

    # Swap in the raw mock so tests can configure return values directly.
    client._client = mock_client_instance
    return client, mock_client_instance


# ---------------------------------------------------------------------------
# Test 1 — chat completions create records an inference
# ---------------------------------------------------------------------------

class TestChatCompletionsCreateRecordsInference:
    def test_chat_completions_create_records_inference(self):
        auditor = MagicMock()
        auditor.record.return_value = "rec-1"

        fake_mod = _make_openai_module()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response()
        fake_mod.AzureOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"openai": fake_mod}):
            import aria.integrations.azure_openai as m
            importlib.reload(m)
            client = m.ARIAAzureOpenAI(
                auditor=auditor,
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-02-01",
            )
            client._client = mock_client
            client.chat.completions._orig = mock_client.chat.completions

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert resp is not None
        auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — async chat completions create
# ---------------------------------------------------------------------------

class TestAsyncChatCompletionsCreate:
    def test_async_chat_completions_create(self):
        auditor = MagicMock()

        fake_mod = _make_openai_module()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_response())
        fake_mod.AsyncAzureOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"openai": fake_mod}):
            import aria.integrations.azure_openai as m
            importlib.reload(m)
            client = m.ARIAAsyncAzureOpenAI(
                auditor=auditor,
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-02-01",
            )
            client._client = mock_client
            client.chat.completions._orig = mock_client.chat.completions

            async def _run():
                return await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )

            resp = asyncio.get_event_loop().run_until_complete(_run())

        assert resp is not None
        auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3 — provider field is "azure_openai"
# ---------------------------------------------------------------------------

class TestAzureProviderMetadata:
    def test_azure_provider_metadata(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor)
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
        )

        auditor.record.assert_called_once()
        # metadata is the 4th positional arg or keyword arg
        call_kwargs = auditor.record.call_args[1]
        metadata = call_kwargs.get("metadata", {})
        assert metadata.get("provider") == "azure_openai"


# ---------------------------------------------------------------------------
# Test 4 — model_id override
# ---------------------------------------------------------------------------

class TestModelIdOverride:
    def test_model_id_override(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor, model_id="my-prod-deployment")
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
        )

        args = auditor.record.call_args[0]
        assert args[0] == "my-prod-deployment"


# ---------------------------------------------------------------------------
# Test 5 — embeddings create records
# ---------------------------------------------------------------------------

class TestEmbeddingsCreateRecords:
    def test_embeddings_create_records(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor)

        embed_resp = MagicMock()
        embed_resp.data = [MagicMock()]
        embed_resp.data[0].embedding = [0.1] * 1536
        mock_client.embeddings.create.return_value = embed_resp
        client.embeddings._orig = mock_client.embeddings

        client.embeddings.create(model="text-embedding-ada-002", input="hello azure")

        auditor.record.assert_called_once()
        call_kwargs = auditor.record.call_args[1]
        metadata = call_kwargs.get("metadata", {})
        assert metadata.get("provider") == "azure_openai"


# ---------------------------------------------------------------------------
# Test 6 — ImportError without openai
# ---------------------------------------------------------------------------

class TestImportErrorWithoutOpenai:
    def test_import_error_without_openai(self):
        with patch.dict(sys.modules, {"openai": None}):
            import aria.integrations.azure_openai as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="openai"):
                m.ARIAAzureOpenAI(
                    auditor=MagicMock(),
                    azure_endpoint="https://test.openai.azure.com/",
                    api_version="2024-02-01",
                )

    def test_import_error_async_without_openai(self):
        with patch.dict(sys.modules, {"openai": None}):
            import aria.integrations.azure_openai as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="openai"):
                m.ARIAAsyncAzureOpenAI(
                    auditor=MagicMock(),
                    azure_endpoint="https://test.openai.azure.com/",
                    api_version="2024-02-01",
                )


# ---------------------------------------------------------------------------
# Test 7 — record error swallowed
# ---------------------------------------------------------------------------

class TestRecordErrorSwallowed:
    def test_record_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage unavailable")

        client, mock_client = _build_client(auditor=auditor)
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        # Should not raise even though auditor.record raises
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert resp is not None


# ---------------------------------------------------------------------------
# Test 8 — __getattr__ proxy
# ---------------------------------------------------------------------------

class TestGetAttrProxy:
    def test_getattr_proxy(self):
        fake_mod = _make_openai_module()
        mock_client = MagicMock()
        mock_client.api_key = "az-key-123"
        fake_mod.AzureOpenAI = MagicMock(return_value=mock_client)

        with patch.dict(sys.modules, {"openai": fake_mod}):
            import aria.integrations.azure_openai as m
            importlib.reload(m)
            client = m.ARIAAzureOpenAI(
                auditor=MagicMock(),
                azure_endpoint="https://test.openai.azure.com/",
                api_version="2024-02-01",
            )

        assert client.api_key == "az-key-123"


# ---------------------------------------------------------------------------
# Test 9 — ARIAQuick integration
# ---------------------------------------------------------------------------

class TestAriaQuickIntegration:
    def test_aria_quick_integration(self):
        aria = MagicMock()
        client, mock_client = _build_client(aria=aria)
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "quick test"}],
        )

        aria.record.assert_called_once()
        call_kwargs = aria.record.call_args[1]
        assert call_kwargs.get("model_id") == "gpt-4o"


# ---------------------------------------------------------------------------
# Test 10 — confidence extraction from logprobs
# ---------------------------------------------------------------------------

class TestConfidenceExtractionFromLogprobs:
    def test_confidence_extraction_from_logprobs(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor)

        avg_lp = -0.3
        resp = _mock_response_logprobs(avg_lp)
        mock_client.chat.completions.create.return_value = resp
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "logprobs test"}],
        )

        auditor.record.assert_called_once()
        call_kwargs = auditor.record.call_args[1]
        confidence = call_kwargs.get("confidence")
        assert confidence is not None
        expected = round(math.exp(avg_lp), 4)
        assert confidence == pytest.approx(expected)

    def test_confidence_none_when_no_logprobs(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor)
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="gpt-4o",
            messages=[],
        )

        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs.get("confidence") is None


# ---------------------------------------------------------------------------
# Additional: deployment field in metadata mirrors model kwarg
# ---------------------------------------------------------------------------

class TestDeploymentMetadata:
    def test_deployment_field_matches_model(self):
        auditor = MagicMock()
        client, mock_client = _build_client(auditor=auditor)
        mock_client.chat.completions.create.return_value = _mock_response()
        client.chat.completions._orig = mock_client.chat.completions

        client.chat.completions.create(
            model="my-gpt4o-deployment",
            messages=[],
        )

        call_kwargs = auditor.record.call_args[1]
        metadata = call_kwargs.get("metadata", {})
        assert metadata.get("deployment") == "my-gpt4o-deployment"
        assert metadata.get("model") == "my-gpt4o-deployment"
