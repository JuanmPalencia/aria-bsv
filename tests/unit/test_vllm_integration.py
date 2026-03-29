"""Tests for aria.integrations.vllm — ARIAvLLM and ARIAAsyncvLLM wrappers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("respx")
import respx  # noqa: E402

from aria.integrations.vllm import ARIAAsyncvLLM, ARIAvLLM, _ARIARecorder


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"

_CHAT_RESPONSE = {
    "id": "cmpl-test",
    "object": "chat.completion",
    "model": "vllm-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from vLLM!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}

_COMPLETE_RESPONSE = {
    "id": "cmpl-test2",
    "object": "text_completion",
    "model": "vllm-model",
    "choices": [
        {
            "index": 0,
            "text": "The answer is 42.",
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}


def _auditor() -> MagicMock:
    m = MagicMock()
    m.record.return_value = "rec-id-1"
    return m


# ---------------------------------------------------------------------------
# ARIAvLLM — synchronous
# ---------------------------------------------------------------------------

class TestARIAvLLMChat:
    def test_chat_records_inference(self):
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
            resp = client.chat(messages=[{"role": "user", "content": "Hello"}])

        assert resp["content"] == "Hello from vLLM!"
        auditor.record.assert_called_once()

    def test_complete_records_inference(self):
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/completions").mock(
                return_value=httpx.Response(200, json=_COMPLETE_RESPONSE)
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
            resp = client.complete(prompt="The answer is")

        assert resp["content"] == "The answer is 42."
        auditor.record.assert_called_once()

    def test_model_id_override(self):
        """Passing model= to chat() overrides the constructor model_id in the record."""
        auditor = _auditor()
        override_response = dict(_CHAT_RESPONSE)
        override_response["model"] = "custom-model"
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=override_response)
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="default-model", auditor=auditor)
            client.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="custom-model",
            )

        call_kwargs = auditor.record.call_args
        # First positional arg is model_id
        assert call_kwargs[0][0] == "custom-model"

    def test_aria_quick_integration(self):
        """ARIAvLLM works with ARIAQuick instead of InferenceAuditor."""
        aria = MagicMock()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAvLLM(base_url=BASE_URL, aria=aria)
            client.chat(messages=[{"role": "user", "content": "hi"}])

        aria.record.assert_called_once()

    def test_record_error_swallowed(self):
        """If auditor.record raises, the exception is swallowed and response is returned."""
        auditor = _auditor()
        auditor.record.side_effect = RuntimeError("db exploded")
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
            # Should NOT raise
            resp = client.chat(messages=[{"role": "user", "content": "hi"}])

        assert resp["content"] == "Hello from vLLM!"

    def test_http_error_propagates(self):
        """An HTTP error from the server propagates as httpx.HTTPStatusError."""
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(500, json={"error": "internal"})
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
            with pytest.raises(httpx.HTTPStatusError):
                client.chat(messages=[{"role": "user", "content": "hi"}])

        auditor.record.assert_not_called()

    def test_provider_metadata_is_vllm(self):
        """The metadata dict recorded always contains provider='vllm'."""
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
            client.chat(messages=[{"role": "user", "content": "hello"}])

        _, kwargs = auditor.record.call_args
        assert kwargs["metadata"]["provider"] == "vllm"

    def test_custom_base_url(self):
        """A custom base_url is used for the HTTP request and recorded in metadata."""
        custom_url = "http://gpu-cluster:9000"
        auditor = _auditor()
        with respx.mock:
            respx.post(f"{custom_url}/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=_CHAT_RESPONSE)
            )
            client = ARIAvLLM(base_url=custom_url, model_id="vllm-model", auditor=auditor)
            client.chat(messages=[{"role": "user", "content": "hi"}])

        _, kwargs = auditor.record.call_args
        assert kwargs["metadata"]["base_url"] == custom_url


# ---------------------------------------------------------------------------
# ARIAAsyncvLLM — asynchronous
# ---------------------------------------------------------------------------

class TestARIAAsyncvLLMChat:
    def test_async_chat_records_inference(self):
        auditor = _auditor()

        async def _run():
            with respx.mock:
                respx.post(f"{BASE_URL}/v1/chat/completions").mock(
                    return_value=httpx.Response(200, json=_CHAT_RESPONSE)
                )
                client = ARIAAsyncvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
                resp = await client.chat(messages=[{"role": "user", "content": "hello"}])
                await client.aclose()
            return resp

        resp = asyncio.get_event_loop().run_until_complete(_run())
        assert resp["content"] == "Hello from vLLM!"
        auditor.record.assert_called_once()

    def test_async_complete_records_inference(self):
        auditor = _auditor()

        async def _run():
            with respx.mock:
                respx.post(f"{BASE_URL}/v1/completions").mock(
                    return_value=httpx.Response(200, json=_COMPLETE_RESPONSE)
                )
                client = ARIAAsyncvLLM(base_url=BASE_URL, model_id="vllm-model", auditor=auditor)
                resp = await client.complete(prompt="The answer is")
                await client.aclose()
            return resp

        resp = asyncio.get_event_loop().run_until_complete(_run())
        assert resp["content"] == "The answer is 42."
        auditor.record.assert_called_once()


# ---------------------------------------------------------------------------
# _ARIARecorder — unit tests
# ---------------------------------------------------------------------------

class TestARIARecorder:
    def test_calls_auditor(self):
        auditor = MagicMock()
        rec = _ARIARecorder(auditor=auditor)
        rec.record(
            model_id="vllm-model",
            input_data={"messages": []},
            output_data={"content": "hi"},
            confidence=None,
            latency_ms=50.0,
            metadata={"provider": "vllm"},
        )
        auditor.record.assert_called_once()
        assert auditor.record.call_args[0][0] == "vllm-model"

    def test_calls_aria(self):
        aria = MagicMock()
        rec = _ARIARecorder(aria=aria)
        rec.record(
            model_id="vllm-model",
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
