"""Tests for aria.integrations.google_gemini — ARIAGemini and ARIAAsyncGemini."""

from __future__ import annotations

import asyncio
import importlib
import math
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a fake google / google.generativeai module hierarchy
# ---------------------------------------------------------------------------

def _make_google_modules():
    """Build minimal fake ``google`` and ``google.generativeai`` modules."""
    google_mod = ModuleType("google")
    genai_mod = ModuleType("google.generativeai")
    google_mod.generativeai = genai_mod
    genai_mod.GenerativeModel = MagicMock
    return google_mod, genai_mod


# ---------------------------------------------------------------------------
# Mock response factories
# ---------------------------------------------------------------------------

class MockUsageMetadata:
    def __init__(self, prompt=10, candidates=20, total=30):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.total_token_count = total


class MockCandidate:
    def __init__(self, finish_reason="STOP", avg_logprobs=None):
        self.finish_reason = finish_reason
        self.avg_logprobs = avg_logprobs


class MockGeminiResponse:
    def __init__(
        self,
        text="Gemini says hello",
        finish_reason="STOP",
        avg_logprobs=None,
        usage=None,
    ):
        self.text = text
        self.candidates = [MockCandidate(finish_reason=finish_reason, avg_logprobs=avg_logprobs)]
        self.usage_metadata = usage or MockUsageMetadata()


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

def _reload_module(google_mod=None, genai_mod=None):
    """Reload aria.integrations.google_gemini with injected fake modules."""
    if google_mod is None:
        google_mod, genai_mod = _make_google_modules()
    patches = {
        "google": google_mod,
        "google.generativeai": genai_mod,
    }
    with patch.dict(sys.modules, patches):
        import aria.integrations.google_gemini as m
        importlib.reload(m)
        return m, google_mod, genai_mod


# ---------------------------------------------------------------------------
# Test 1: basic generate_content records to auditor
# ---------------------------------------------------------------------------

class TestGenerateContentRecordsInference:
    def test_generate_content_records_inference(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MockGeminiResponse()
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-pro", auditor=auditor)
            client._model = mock_model_instance
            resp = client.generate_content("What is BSV?")

        assert resp is not None
        auditor.record.assert_called_once()
        call_args = auditor.record.call_args[0]
        assert call_args[0] == "gemini-1.5-pro"
        assert call_args[1] == {"prompt": "What is BSV?"}


# ---------------------------------------------------------------------------
# Test 2: avg_logprobs extraction into confidence
# ---------------------------------------------------------------------------

class TestGenerateContentExtractsConfidence:
    def test_generate_content_extracts_confidence(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        avg_lp = -0.3
        mock_model_instance.generate_content.return_value = MockGeminiResponse(
            avg_logprobs=avg_lp
        )
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-pro", auditor=auditor)
            client._model = mock_model_instance
            client.generate_content("Hello")

        kwargs = auditor.record.call_args[1]
        expected = round(math.exp(avg_lp), 4)
        assert kwargs["confidence"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3: no confidence when avg_logprobs unavailable
# ---------------------------------------------------------------------------

class TestNoConfidenceWhenUnavailable:
    def test_no_confidence_when_unavailable(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MockGeminiResponse(
            avg_logprobs=None
        )
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-flash", auditor=auditor)
            client._model = mock_model_instance
            client.generate_content("Ping")

        kwargs = auditor.record.call_args[1]
        assert kwargs["confidence"] is None


# ---------------------------------------------------------------------------
# Test 4: async generate_content_async records inference
# ---------------------------------------------------------------------------

class TestAsyncGenerateContentRecordsInference:
    def test_async_generate_content_records_inference(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content_async = AsyncMock(
            return_value=MockGeminiResponse()
        )
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAAsyncGemini(model_name="gemini-1.5-pro", auditor=auditor)
            client._model = mock_model_instance

            async def _run():
                return await client.generate_content_async("Async hello")

            resp = asyncio.get_event_loop().run_until_complete(_run())

        assert resp is not None
        auditor.record.assert_called_once()
        call_args = auditor.record.call_args[0]
        assert call_args[0] == "gemini-1.5-pro"


# ---------------------------------------------------------------------------
# Test 5: model_id kwarg overrides the model_name label
# ---------------------------------------------------------------------------

class TestModelIdOverride:
    def test_model_id_override(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MockGeminiResponse()
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(
                model_name="gemini-1.5-pro",
                auditor=auditor,
                model_id="my-prod-gemini",
            )
            client._model = mock_model_instance
            client.generate_content("Test override")

        call_args = auditor.record.call_args[0]
        assert call_args[0] == "my-prod-gemini"


# ---------------------------------------------------------------------------
# Test 6: works with ARIAQuick
# ---------------------------------------------------------------------------

class TestAriaQuickIntegration:
    def test_aria_quick_integration(self):
        aria = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MockGeminiResponse()
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-flash", aria=aria)
            client._model = mock_model_instance
            resp = client.generate_content("Quick test")

        assert resp is not None
        aria.record.assert_called_once()
        kwargs = aria.record.call_args[1]
        assert kwargs["model_id"] == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# Test 7: auditor error is swallowed — does not crash the caller
# ---------------------------------------------------------------------------

class TestRecordErrorSwallowed:
    def test_record_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("storage failure")
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = MockGeminiResponse()
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-pro", auditor=auditor)
            client._model = mock_model_instance
            # Must not raise
            resp = client.generate_content("Should not crash")

        assert resp is not None


# ---------------------------------------------------------------------------
# Test 8: ImportError raised when google.generativeai not installed
# ---------------------------------------------------------------------------

class TestImportErrorWithoutGoogle:
    def test_import_error_without_google(self):
        with patch.dict(sys.modules, {"google": None, "google.generativeai": None}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            with pytest.raises(ImportError, match="google-generativeai"):
                m.ARIAGemini(model_name="gemini-1.5-pro", auditor=MagicMock())


# ---------------------------------------------------------------------------
# Test 9: unknown attributes proxy to underlying model
# ---------------------------------------------------------------------------

class TestGetattrProxy:
    def test_getattr_proxy(self):
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        mock_model_instance.some_special_attr = "special-value"
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-pro", auditor=MagicMock())
            client._model = mock_model_instance

        assert client.some_special_attr == "special-value"


# ---------------------------------------------------------------------------
# Test 10: usage_metadata token counts appear in the output dict
# ---------------------------------------------------------------------------

class TestUsageMetadataExtracted:
    def test_usage_metadata_extracted(self):
        auditor = MagicMock()
        google_mod, genai_mod = _make_google_modules()
        mock_model_instance = MagicMock()
        usage = MockUsageMetadata(prompt=8, candidates=16, total=24)
        mock_model_instance.generate_content.return_value = MockGeminiResponse(usage=usage)
        genai_mod.GenerativeModel = MagicMock(return_value=mock_model_instance)

        with patch.dict(sys.modules, {"google": google_mod, "google.generativeai": genai_mod}):
            import aria.integrations.google_gemini as m
            importlib.reload(m)
            client = m.ARIAGemini(model_name="gemini-1.5-pro", auditor=auditor)
            client._model = mock_model_instance
            client.generate_content("Token count test")

        call_args = auditor.record.call_args[0]
        output_data = call_args[2]
        assert output_data["usage"]["prompt_tokens"] == 8
        assert output_data["usage"]["candidates_tokens"] == 16
        assert output_data["usage"]["total_tokens"] == 24
