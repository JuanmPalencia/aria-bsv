"""Tests for aria.integrations.cohere — ARIACohere and ARIAAsyncCohere wrappers."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake cohere module factory
# ---------------------------------------------------------------------------

def _make_cohere_module() -> ModuleType:
    """Build a minimal fake ``cohere`` module for injection into sys.modules."""
    mod = ModuleType("cohere")
    mod.ClientV2 = MagicMock
    mod.AsyncClientV2 = MagicMock
    return mod


# ---------------------------------------------------------------------------
# Mock response factories
# ---------------------------------------------------------------------------

def _make_chat_response(
    text: str = "Hello from Cohere!",
    finish_reason: str = "COMPLETE",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock Cohere V2 chat response."""
    resp = MagicMock()

    content_block = MagicMock()
    content_block.text = text
    resp.message = MagicMock()
    resp.message.content = [content_block]

    resp.finish_reason = finish_reason

    billed = MagicMock()
    billed.input_tokens = input_tokens
    billed.output_tokens = output_tokens
    resp.usage = MagicMock()
    resp.usage.billed_units = billed

    return resp


def _make_embed_response(vector: list[float] | None = None) -> MagicMock:
    """Build a mock Cohere V2 embed response."""
    if vector is None:
        vector = [0.1, 0.2, 0.3, 0.4]

    resp = MagicMock()
    resp.embeddings = MagicMock()
    resp.embeddings.float = [vector]
    return resp


# ---------------------------------------------------------------------------
# Helper: import and reload aria.integrations.cohere with injected module
# ---------------------------------------------------------------------------

def _load_module(fake_cohere: ModuleType | None = None) -> ModuleType:
    """Reload aria.integrations.cohere with the supplied fake cohere module."""
    cohere_mod = fake_cohere if fake_cohere is not None else _make_cohere_module()
    with patch.dict(sys.modules, {"cohere": cohere_mod}):
        import aria.integrations.cohere as m
        importlib.reload(m)
        return m


def _build_aria_cohere(
    fake_cohere: ModuleType | None = None,
    auditor: MagicMock | None = None,
    aria: MagicMock | None = None,
    model_id: str | None = None,
    mock_client: MagicMock | None = None,
) -> tuple[object, MagicMock]:
    """Instantiate ARIACohere backed by an injected fake cohere module.

    Returns ``(aria_client, mock_cohere_client_instance)``.
    """
    if fake_cohere is None:
        fake_cohere = _make_cohere_module()

    if mock_client is None:
        mock_client = MagicMock()

    fake_cohere.ClientV2 = MagicMock(return_value=mock_client)

    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        import aria.integrations.cohere as m
        importlib.reload(m)
        client = m.ARIACohere(auditor=auditor, aria=aria, model_id=model_id)

    # Wire up the underlying client directly so test assertions are reliable
    client._client = mock_client
    return client, mock_client


def _build_aria_async_cohere(
    fake_cohere: ModuleType | None = None,
    auditor: MagicMock | None = None,
    aria: MagicMock | None = None,
    model_id: str | None = None,
    mock_client: MagicMock | None = None,
) -> tuple[object, MagicMock]:
    """Instantiate ARIAAsyncCohere backed by an injected fake cohere module."""
    if fake_cohere is None:
        fake_cohere = _make_cohere_module()

    if mock_client is None:
        mock_client = MagicMock()

    fake_cohere.AsyncClientV2 = MagicMock(return_value=mock_client)

    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        import aria.integrations.cohere as m
        importlib.reload(m)
        client = m.ARIAAsyncCohere(auditor=auditor, aria=aria, model_id=model_id)

    client._client = mock_client
    return client, mock_client


# ---------------------------------------------------------------------------
# 1. test_chat_records_inference
# ---------------------------------------------------------------------------

class TestChatRecordsInference:
    def test_chat_records_inference(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        chat_resp = _make_chat_response(text="Hi there", finish_reason="COMPLETE")
        mock_client.chat.return_value = chat_resp

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)
        resp = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="command-r-plus",
        )

        assert resp is chat_resp
        auditor.record.assert_called_once()

        # Verify positional args: model_id, input_data, output_data
        call_args = auditor.record.call_args
        model_id_arg = call_args[0][0]
        input_data = call_args[0][1]
        output_data = call_args[0][2]

        assert model_id_arg == "command-r-plus"
        assert "messages" in input_data
        assert output_data["text"] == "Hi there"
        assert output_data["finish_reason"] == "COMPLETE"


# ---------------------------------------------------------------------------
# 2. test_async_chat_records_inference
# ---------------------------------------------------------------------------

class TestAsyncChatRecordsInference:
    def test_async_chat_records_inference(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        chat_resp = _make_chat_response(text="Async reply")
        mock_client.chat = AsyncMock(return_value=chat_resp)

        client, _ = _build_aria_async_cohere(auditor=auditor, mock_client=mock_client)

        async def _run():
            return await client.chat(
                messages=[{"role": "user", "content": "Hello async"}],
                model="command-r",
            )

        resp = asyncio.get_event_loop().run_until_complete(_run())

        assert resp is chat_resp
        auditor.record.assert_called_once()

        call_args = auditor.record.call_args
        assert call_args[0][0] == "command-r"
        assert call_args[0][2]["text"] == "Async reply"


# ---------------------------------------------------------------------------
# 3. test_embed_records_inference
# ---------------------------------------------------------------------------

class TestEmbedRecordsInference:
    def test_embed_records_inference(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        vector = [0.1] * 1024
        embed_resp = _make_embed_response(vector=vector)
        mock_client.embed.return_value = embed_resp

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)
        resp = client.embed(
            texts=["hello world", "foo bar"],
            model="embed-english-v3.0",
            input_type="search_document",
        )

        assert resp is embed_resp
        auditor.record.assert_called_once()

        call_args = auditor.record.call_args
        input_data = call_args[0][1]
        output_data = call_args[0][2]

        assert input_data["texts"] == ["hello world", "foo bar"]
        assert input_data["input_type"] == "search_document"
        assert output_data["dimensions"] == 1024


# ---------------------------------------------------------------------------
# 4. test_confidence_is_none
# ---------------------------------------------------------------------------

class TestConfidenceIsNone:
    def test_confidence_is_none(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response()

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)
        client.chat(
            messages=[{"role": "user", "content": "ping"}],
            model="command-r",
        )

        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs.get("confidence") is None


# ---------------------------------------------------------------------------
# 5. test_model_id_override
# ---------------------------------------------------------------------------

class TestModelIdOverride:
    def test_model_id_override(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response()

        client, _ = _build_aria_cohere(
            auditor=auditor, mock_client=mock_client, model_id="my-prod-model"
        )
        client.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="command-r-plus",
        )

        call_args = auditor.record.call_args
        # model_id in ARIA record should be the override, not the API model
        assert call_args[0][0] == "my-prod-model"


# ---------------------------------------------------------------------------
# 6. test_aria_quick_integration
# ---------------------------------------------------------------------------

class TestARIAQuickIntegration:
    def test_aria_quick_integration(self):
        aria_quick = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response(text="quick reply")

        client, _ = _build_aria_cohere(aria=aria_quick, mock_client=mock_client)
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="command-r",
        )

        aria_quick.record.assert_called_once()
        call_kwargs = aria_quick.record.call_args[1]
        assert call_kwargs["model_id"] == "command-r"
        assert call_kwargs["output_data"]["text"] == "quick reply"


# ---------------------------------------------------------------------------
# 7. test_record_error_swallowed
# ---------------------------------------------------------------------------

class TestRecordErrorSwallowed:
    def test_record_error_swallowed(self):
        """A failure inside the recorder must not propagate to the caller."""
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage down")

        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response()

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)

        # Should NOT raise even though record() raises internally
        resp = client.chat(
            messages=[{"role": "user", "content": "test"}],
            model="command-r",
        )
        assert resp is not None


# ---------------------------------------------------------------------------
# 8. test_import_error_without_cohere
# ---------------------------------------------------------------------------

class TestImportErrorWithoutCohere:
    def test_import_error_without_cohere(self):
        with patch.dict(sys.modules, {"cohere": None}):
            import aria.integrations.cohere as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="cohere"):
                m.ARIACohere(auditor=MagicMock())

    def test_import_error_async_without_cohere(self):
        with patch.dict(sys.modules, {"cohere": None}):
            import aria.integrations.cohere as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="cohere"):
                m.ARIAAsyncCohere(auditor=MagicMock())


# ---------------------------------------------------------------------------
# 9. test_getattr_proxy
# ---------------------------------------------------------------------------

class TestGetattrProxy:
    def test_getattr_proxy(self):
        mock_client = MagicMock()
        mock_client.some_custom_attr = "sentinel-value"

        client, _ = _build_aria_cohere(auditor=MagicMock(), mock_client=mock_client)
        assert client.some_custom_attr == "sentinel-value"

    def test_getattr_proxy_async(self):
        mock_client = MagicMock()
        mock_client.another_attr = 42

        client, _ = _build_aria_async_cohere(
            auditor=MagicMock(), mock_client=mock_client
        )
        assert client.another_attr == 42


# ---------------------------------------------------------------------------
# 10. test_provider_metadata_is_cohere
# ---------------------------------------------------------------------------

class TestProviderMetadataIsCohere:
    def test_provider_metadata_is_cohere_chat(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_chat_response()

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="command-r-plus",
        )

        call_kwargs = auditor.record.call_args[1]
        metadata = call_kwargs.get("metadata", {})
        assert metadata["provider"] == "cohere"
        assert metadata["model"] == "command-r-plus"

    def test_provider_metadata_is_cohere_embed(self):
        auditor = MagicMock()
        mock_client = MagicMock()
        mock_client.embed.return_value = _make_embed_response()

        client, _ = _build_aria_cohere(auditor=auditor, mock_client=mock_client)
        client.embed(
            texts=["doc"],
            model="embed-english-v3.0",
            input_type="search_document",
        )

        call_kwargs = auditor.record.call_args[1]
        metadata = call_kwargs.get("metadata", {})
        assert metadata["provider"] == "cohere"
        assert metadata["model"] == "embed-english-v3.0"
