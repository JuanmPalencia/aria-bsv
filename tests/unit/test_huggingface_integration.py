"""Tests for aria.integrations.huggingface — ARIAPipeline wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aria.integrations.huggingface import (
    ARIAPipeline,
    _extract_confidence,
    _serialize_output,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestExtractConfidence:
    def test_text_classification_result(self):
        result = [{"label": "POSITIVE", "score": 0.9876}]
        c = _extract_confidence(result)
        assert c == pytest.approx(0.9876)

    def test_nested_list(self):
        result = [[{"label": "NEG", "score": 0.123}]]
        c = _extract_confidence(result)
        assert c == pytest.approx(0.123)

    def test_no_score_key(self):
        result = [{"label": "POSITIVE"}]
        assert _extract_confidence(result) is None

    def test_empty_list(self):
        assert _extract_confidence([]) is None

    def test_string_result(self):
        assert _extract_confidence("hello") is None

    def test_none(self):
        assert _extract_confidence(None) is None

    def test_rounds_to_4_decimals(self):
        result = [{"score": 0.123456789}]
        c = _extract_confidence(result)
        assert c == pytest.approx(0.1235, abs=0.0001)


class TestSerializeOutput:
    def test_list(self):
        result = [{"label": "POS", "score": 0.9}]
        out = _serialize_output(result)
        assert out == {"result": result}

    def test_dict(self):
        result = {"answer": "yes", "score": 0.95}
        out = _serialize_output(result)
        assert out == {"result": result}

    def test_string_truncated(self):
        result = "x" * 2000
        out = _serialize_output(result)
        assert len(out["result"]) == 1000

    def test_non_list_dict_falls_back_to_string(self):
        result = 42  # int — not list or dict
        out = _serialize_output(result)
        assert "result" in out
        assert "42" in out["result"]


# ---------------------------------------------------------------------------
# ARIAPipeline
# ---------------------------------------------------------------------------

def _mock_pipeline(result=None, model_name="distilbert-base", task="text-classification"):
    pipe = MagicMock()
    pipe.return_value = result or [{"label": "POSITIVE", "score": 0.95}]
    pipe.task = task
    pipe.model = MagicMock()
    pipe.model.config = MagicMock()
    pipe.model.config._name_or_path = model_name
    return pipe


class TestARIAPipelineInit:
    def test_basic_init(self):
        pipe = _mock_pipeline()
        auditor = MagicMock()
        ap = ARIAPipeline(pipe, auditor=auditor)
        assert ap._model_id == "distilbert-base"

    def test_model_id_override(self):
        pipe = _mock_pipeline()
        ap = ARIAPipeline(pipe, auditor=MagicMock(), model_id="my-classifier")
        assert ap._model_id == "my-classifier"

    def test_no_auditor_no_aria(self):
        pipe = _mock_pipeline()
        ap = ARIAPipeline(pipe)
        assert ap._auditor is None
        assert ap._aria is None

    def test_getattr_proxy(self):
        pipe = _mock_pipeline()
        pipe.some_attr = "test-value"
        ap = ARIAPipeline(pipe, auditor=MagicMock())
        assert ap.some_attr == "test-value"


class TestARIAPipelineCall:
    def test_call_records_to_auditor(self):
        auditor = MagicMock()
        pipe = _mock_pipeline(result=[{"label": "POSITIVE", "score": 0.99}])
        ap = ARIAPipeline(pipe, auditor=auditor, model_id="test-model")

        result = ap("I love this!")
        assert result == [{"label": "POSITIVE", "score": 0.99}]
        auditor.record.assert_called_once()

    def test_call_records_to_aria(self):
        aria = MagicMock()
        pipe = _mock_pipeline()
        ap = ARIAPipeline(pipe, aria=aria, model_id="test-model")
        ap("hello")
        aria.record.assert_called_once()

    def test_confidence_extracted(self):
        auditor = MagicMock()
        pipe = _mock_pipeline(result=[{"label": "POS", "score": 0.87}])
        ap = ARIAPipeline(pipe, auditor=auditor, model_id="m")
        ap("text")
        call_kwargs = auditor.record.call_args[1]
        assert call_kwargs["confidence"] == pytest.approx(0.87)

    def test_record_error_swallowed(self):
        auditor = MagicMock()
        auditor.record.side_effect = RuntimeError("oops")
        pipe = _mock_pipeline()
        ap = ARIAPipeline(pipe, auditor=auditor, model_id="m")
        # Should not raise
        result = ap("test")
        assert result is not None

    def test_no_auditor_no_record(self):
        pipe = _mock_pipeline()
        ap = ARIAPipeline(pipe)
        # Should not raise even without auditor
        result = ap("hello")
        assert result is not None

    def test_metadata_has_task(self):
        auditor = MagicMock()
        pipe = _mock_pipeline(task="sentiment-analysis")
        ap = ARIAPipeline(pipe, auditor=auditor, model_id="m")
        ap("test")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["task"] == "sentiment-analysis"


# ---------------------------------------------------------------------------
# ARIAPipeline.from_pretrained
# ---------------------------------------------------------------------------

class TestARIAPipelineFromPretrained:
    def _fake_transformers(self, mock_pipe):
        """Build a fake transformers module with a mock pipeline function."""
        import sys
        from types import ModuleType
        mod = ModuleType("transformers")
        mock_pipeline_fn = MagicMock(return_value=mock_pipe)
        mod.pipeline = mock_pipeline_fn
        return mod, mock_pipeline_fn

    def test_creates_pipeline(self):
        auditor = MagicMock()
        mock_pipe = _mock_pipeline()
        fake_tf, mock_fn = self._fake_transformers(mock_pipe)

        import sys
        with patch.dict(sys.modules, {"transformers": fake_tf}):
            ap = ARIAPipeline.from_pretrained(
                task="text-classification",
                model="distilbert-base-uncased",
                auditor=auditor,
            )
            mock_fn.assert_called_once_with(
                task="text-classification",
                model="distilbert-base-uncased",
            )
            assert isinstance(ap, ARIAPipeline)

    def test_model_id_from_model_name(self):
        auditor = MagicMock()
        mock_pipe = _mock_pipeline(model_name="distilbert-base-uncased")
        fake_tf, _ = self._fake_transformers(mock_pipe)

        import sys
        with patch.dict(sys.modules, {"transformers": fake_tf}):
            ap = ARIAPipeline.from_pretrained(
                task="text-classification",
                model="distilbert-base-uncased",
                auditor=auditor,
            )
            assert ap._model_id == "distilbert-base-uncased"

    def test_model_id_override(self):
        auditor = MagicMock()
        mock_pipe = _mock_pipeline()
        fake_tf, _ = self._fake_transformers(mock_pipe)

        import sys
        with patch.dict(sys.modules, {"transformers": fake_tf}):
            ap = ARIAPipeline.from_pretrained(
                task="text-classification",
                model="distilbert-base-uncased",
                auditor=auditor,
                model_id="my-custom-id",
            )
            assert ap._model_id == "my-custom-id"

    def test_import_error(self):
        import sys
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                ARIAPipeline.from_pretrained(
                    task="text-classification",
                    model="dummy",
                    auditor=MagicMock(),
                )


# ---------------------------------------------------------------------------
# _infer_model_id static
# ---------------------------------------------------------------------------

class TestInferModelId:
    def test_uses_name_or_path(self):
        pipe = _mock_pipeline(model_name="bert-base")
        assert ARIAPipeline._infer_model_id(pipe) == "bert-base"

    def test_fallback_model_type(self):
        pipe = MagicMock()
        pipe.model.config._name_or_path = None
        pipe.model.config.model_type = "bert"
        assert ARIAPipeline._infer_model_id(pipe) == "bert"

    def test_fallback_exception(self):
        pipe = MagicMock()
        pipe.model = None
        assert ARIAPipeline._infer_model_id(pipe) == "hf-pipeline"
