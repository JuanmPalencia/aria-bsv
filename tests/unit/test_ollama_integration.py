"""Tests for aria.integrations.ollama — ARIAOllama and ARIAAsyncOllama wrappers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("respx")
import respx  # noqa: E402

from aria.integrations.ollama import ARIAAsyncOllama, ARIAOllama, _ARIARecorder


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:11434"

_CHAT_RESPONSE = {
    "model": "llama3",
    "message": {"role": "assistant", "content": "Hello from Ollama!"},
    "done": True,
    "total_duration": 123456789,
}

_GENERATE_RESPONSE = {
    "model": "llama3",
    "response": "Once upon a time in a land far away.",
    "done": True,
    "total_duration": 987654321,
}

_TAGS_RESPONSE = {
    "models": [
        {"name": "llama3:latest", "size": 4700000000},
        {"name": "mistral:7b-instruct", "size": 3800000000},
    ]
}


def _auditor() -> MagicMock:
    m = MagicMock()
    m.record.return_value = "rec-id-1"
    return m


# ---------------------------------------------------------------------------
# ARIAOllama — synchronous
# ---------------------------------------------------------------------------

class TestARIAOllamaChat:
    def test_chat_records_inference(self):
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/api/chat").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
            resp = client.chat(
                model="llama3",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert resp["content"] == "Hello from Ollama!"
        assert resp["done"] is True
        auditor.record.assert_called_once()

    def test_generate_records_inference(self):
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/api/generate").mock(
                return_value=httpx.Response(200, json=_GENERATE_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
            resp = client.generate(model="llama3", prompt="Once upon a time")

        assert "Once upon a time" in resp["content"]
        auditor.record.assert_called_once()

    def test_list_models(self):
        with respx.mock:
            respx.get(f"{BASE_URL}/api/tags").mock(
                return_value=httpx.Response(200, json=_TAGS_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL)
            models = client.list_models()

        assert "llama3:latest" in models
        assert "mistral:7b-instruct" in models
        assert len(models) == 2

    def test_model_id_override(self):
        """Passing model= to chat() is used as the model_id in the audit record."""
        auditor = _auditor()
        override_response = dict(_CHAT_RESPONSE)
        override_response["model"] = "mistral:7b"
        with respx.mock:
            respx.post(f"{BASE_URL}/api/chat").mock(
                return_value=httpx.Response(200, json=override_response)
            )
            client = ARIAOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
            client.chat(
                model="mistral:7b",
                messages=[{"role": "user", "content": "hi"}],
            )

        call_kwargs = auditor.record.call_args
        assert call_kwargs[0][0] == "mistral:7b"

    def test_aria_quick_integration(self):
        """ARIAOllama works with ARIAQuick instead of InferenceAuditor."""
        aria = MagicMock()
        with respx.mock:
            respx.post(f"{BASE_URL}/api/chat").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL, aria=aria)
            client.chat(model="llama3", messages=[{"role": "user", "content": "hi"}])

        aria.record.assert_called_once()

    def test_record_error_swallowed(self):
        """If auditor.record raises, the exception is swallowed and response is returned."""
        auditor = _auditor()
        auditor.record.side_effect = RuntimeError("db exploded")
        with respx.mock:
            respx.post(f"{BASE_URL}/api/chat").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
            # Should NOT raise
            resp = client.chat(model="llama3", messages=[{"role": "user", "content": "hi"}])

        assert resp["content"] == "Hello from Ollama!"

    def test_provider_metadata_is_ollama(self):
        """The metadata dict recorded always contains provider='ollama'."""
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/api/chat").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
            client.chat(model="llama3", messages=[{"role": "user", "content": "hello"}])

        _, kwargs = auditor.record.call_args
        assert kwargs["metadata"]["provider"] == "ollama"

    def test_custom_base_url(self):
        """A custom base_url is used for the HTTP request and recorded in metadata."""
        custom_url = "http://my-ollama-host:11434"
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{custom_url}/api/chat").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAOllama(base_url=custom_url, model_id="llama3", auditor=auditor)
            client.chat(model="llama3", messages=[{"role": "user", "content": "hi"}])

        _, kwargs = auditor.record.call_args
        assert kwargs["metadata"]["base_url"] == custom_url


# ---------------------------------------------------------------------------
# ARIAAsyncOllama — asynchronous
# ---------------------------------------------------------------------------

class TestARIAAsyncOllamaChat:
    def test_async_chat_records_inference(self):
        auditor = _auditor()

        async def _run():
            with respx.mock:
                respx.post(f"{BASE_URL}/api/chat").mock(
                    return_value=httpx.Response(200, json=_CHAT_RESPONSE)
                )
                client = ARIAAsyncOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
                resp = await client.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": "hello"}],
                )
                await client.aclose()
            return resp

        resp = asyncio.get_event_loop().run_until_complete(_run())
        assert resp["content"] == "Hello from Ollama!"
        auditor.record.assert_called_once()

    def test_async_generate_records_inference(self):
        auditor = _auditor()

        async def _run():
            with respx.mock:
                respx.post(f"{BASE_URL}/api/generate").mock(
                    return_value=httpx.Response(200, json=_GENERATE_RESPONSE)
                )
                client = ARIAAsyncOllama(base_url=BASE_URL, model_id="llama3", auditor=auditor)
                resp = await client.generate(model="llama3", prompt="Once upon a time")
                await client.aclose()
            return resp

        resp = asyncio.get_event_loop().run_until_complete(_run())
        assert "Once upon a time" in resp["content"]
        auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# _ARIARecorder — unit tests
# ---------------------------------------------------------------------------

class TestARIARecorder:
    def test_calls_auditor(self):
        auditor = MagicMock()
        rec = _ARIARecorder(auditor=auditor)
        rec.record(
            model_id="llama3",
            input_data={"messages": []},
            output_data={"content": "hi"},
            confidence=None,
            latency_ms=75.0,
            metadata={"provider": "ollama"},
        )
        auditor.record.assert_called_once()
        assert auditor.record.call_args[0][0] == "llama3"

    def test_calls_aria(self):
        aria = MagicMock()
        rec = _ARIARecorder(aria=aria)
        rec.record(
            model_id="llama3",
            input_data={},
            output_data={},
            latency_ms=10,
            metadata={},
        )
        aria.record.assert_called_once()

    def test_exception_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("fail")
        rec = _ARIARecorder(auditor=auditor)
        # Should not raise
        rec.record(
            model_id="m",
            input_data={},
            output_data={},
            latency_ms=1,
            metadata={},
        )
