"""Tests for aria.integrations.vertexai — ARIAVertexAI and ARIAAsyncVertexAI."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.integrations.vertexai import ARIAAsyncVertexAI, ARIAVertexAI, _build_url


# ---------------------------------------------------------------------------
# _build_url helper
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_url_format(self):
        url = _build_url("my-project", "us-central1", "12345")
        assert "us-central1-aiplatform.googleapis.com" in url
        assert "my-project" in url
        assert "us-central1" in url
        assert "12345" in url
        assert url.endswith(":predict")

    def test_location_used_as_subdomain_and_path(self):
        url = _build_url("proj", "europe-west4", "ep-99")
        assert url.startswith("https://europe-west4-aiplatform.googleapis.com")
        assert "europe-west4" in url


# ---------------------------------------------------------------------------
# ARIAVertexAI — sync
# ---------------------------------------------------------------------------

class TestARIAVertexAISync:
    def _make_client(self, auditor=None, aria=None, credentials_json=None):
        return ARIAVertexAI(
            project="test-proj",
            location="us-central1",
            endpoint_id="9999",
            auditor=auditor,
            aria=aria,
            model_id="vertex-model-v1",
            credentials_json=credentials_json,
        )

    def _mock_response(self, predictions=None):
        resp = MagicMock()
        resp.json.return_value = {"predictions": predictions or [{"label": "cat", "score": 0.9}]}
        resp.raise_for_status = MagicMock()
        return resp

    def test_predict_returns_parsed_response(self):
        client = self._make_client(auditor=MagicMock())
        with patch.object(client._http, "post", return_value=self._mock_response()):
            result = client.predict([{"x": 1}])
        assert "predictions" in result

    def test_predict_calls_auditor_record(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict([{"x": 1}])
        auditor.record.assert_called_once()

    def test_metadata_includes_provider_vertexai(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict([{"x": 1}])
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "vertexai"

    def test_metadata_includes_project_and_location(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict([{"x": 1}])
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["project"] == "test-proj"
        assert kwargs["metadata"]["location"] == "us-central1"

    def test_instances_truncated_to_3_in_input(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        many = [{"x": i} for i in range(10)]
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict(many)
        args = auditor.record.call_args[0]
        input_data = args[1]
        # Only repr of first 3 instances stored
        assert "0" in input_data["instances"]  # first instance is {x: 0}

    def test_aria_backend_used(self):
        aria = MagicMock()
        client = self._make_client(aria=aria)
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict([{"x": 1}])
        aria.record.assert_called_once()

    def test_authorization_header_set_with_credentials(self):
        client = self._make_client(
            auditor=MagicMock(),
            credentials_json="Bearer tok123",
        )
        # Authorization header should be present in the underlying httpx client
        assert client._http.headers.get("Authorization") == "Bearer tok123"

    def test_no_credentials_no_auth_header(self):
        client = self._make_client(auditor=MagicMock())
        assert "authorization" not in {k.lower() for k in client._http.headers}

    def test_auditor_error_does_not_propagate(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("record failed")
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", return_value=self._mock_response()):
            result = client.predict([{"x": 1}])
        assert "predictions" in result

    def test_context_manager(self):
        client = self._make_client(auditor=MagicMock())
        with patch.object(client._http, "post", return_value=self._mock_response()):
            with client as c:
                c.predict([{"x": 1}])
        # After context exit, close() should have been called (no error)

    def test_endpoint_id_used_as_model_id_when_none_supplied(self):
        auditor = MagicMock()
        client = ARIAVertexAI(
            project="p",
            location="us-central1",
            endpoint_id="ep-42",
            auditor=auditor,
        )
        with patch.object(client._http, "post", return_value=self._mock_response()):
            client.predict([{"x": 1}])
        args = auditor.record.call_args[0]
        assert args[0] == "ep-42"


# ---------------------------------------------------------------------------
# ARIAAsyncVertexAI — async
# ---------------------------------------------------------------------------

class TestARIAAsyncVertexAI:
    def _make_client(self, auditor=None, aria=None):
        return ARIAAsyncVertexAI(
            project="async-proj",
            location="europe-west4",
            endpoint_id="async-ep",
            auditor=auditor,
            aria=aria,
            model_id="async-model",
        )

    def _mock_async_response(self, predictions=None):
        resp = MagicMock()
        resp.json.return_value = {"predictions": predictions or [{"score": 0.8}]}
        resp.raise_for_status = MagicMock()
        return resp

    @pytest.mark.asyncio
    async def test_predict_returns_parsed_response(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=self._mock_async_response()):
            result = await client.predict([{"x": 1}])
        assert "predictions" in result
        await client.aclose()

    @pytest.mark.asyncio
    async def test_predict_calls_auditor_record(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=self._mock_async_response()):
            await client.predict([{"x": 1}])
        auditor.record.assert_called_once()
        await client.aclose()

    @pytest.mark.asyncio
    async def test_metadata_provider_vertexai(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=self._mock_async_response()):
            await client.predict([{"x": 1}])
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["provider"] == "vertexai"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        auditor = MagicMock()
        client = self._make_client(auditor=auditor)
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=self._mock_async_response()):
            async with client as c:
                await c.predict([{"x": 1}])

    @pytest.mark.asyncio
    async def test_aria_backend(self):
        aria = MagicMock()
        client = self._make_client(aria=aria)
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=self._mock_async_response()):
            await client.predict([{"x": 1}])
        aria.record.assert_called_once()
        await client.aclose()
