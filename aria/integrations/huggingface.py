"""
aria.integrations.huggingface — HuggingFace Transformers pipeline wrapper.

Wraps ``transformers.pipeline()`` so every inference call is automatically
audited via ARIA without changing your existing pipeline code.

Usage::

    from aria.integrations.huggingface import ARIAPipeline

    # Wraps an existing pipeline
    pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    aria_pipe = ARIAPipeline(pipe, aria=aria)

    result = aria_pipe("I love ARIA!")   # ← automatically audited

    # Or create directly:
    aria_pipe = ARIAPipeline.from_pretrained(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        aria=aria,
    )

Supports all pipeline tasks: text-classification, text-generation,
token-classification, question-answering, summarization, translation, etc.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


def _extract_confidence(result: Any) -> float | None:
    """Extract top confidence score from pipeline result if available."""
    try:
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, list):
                item = item[0]
            if isinstance(item, dict) and "score" in item:
                return round(float(item["score"]), 4)
    except Exception:
        pass
    return None


def _serialize_output(result: Any) -> dict[str, Any]:
    try:
        if isinstance(result, (list, dict)):
            return {"result": result}
        return {"result": str(result)[:1000]}
    except Exception:
        return {"result": str(result)[:200]}


class ARIAPipeline:
    """ARIA-audited wrapper around a HuggingFace ``Pipeline``.

    Args:
        pipeline:   A ``transformers.Pipeline`` instance to wrap.
        auditor:    ``InferenceAuditor`` instance.
        aria:       ``ARIAQuick`` instance (alternative to auditor).
        model_id:   Label for ARIA records. Defaults to pipeline's model name.
    """

    def __init__(
        self,
        pipeline: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._auditor = auditor
        self._aria = aria
        self._model_id = model_id or self._infer_model_id(pipeline)

    @classmethod
    def from_pretrained(
        cls,
        task: str,
        model: str,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "ARIAPipeline":
        """Create an ARIAPipeline directly from task + model name.

        Args:
            task:     HuggingFace pipeline task string.
            model:    Model name or path.
            auditor:  ``InferenceAuditor`` instance.
            aria:     ``ARIAQuick`` instance.
            model_id: Override label for ARIA records.
            **kwargs: Forwarded to ``transformers.pipeline()``.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. pip install aria-bsv[huggingface]"
            )
        pipe = pipeline(task=task, model=model, **kwargs)
        return cls(pipe, auditor=auditor, aria=aria, model_id=model_id or model)

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """Run the pipeline and record the inference in ARIA."""
        t0 = time.time()
        result = self._pipeline(inputs, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        input_data = {
            "text": inputs[:500] if isinstance(inputs, str) else str(inputs)[:500]
        }
        output_data = _serialize_output(result)
        confidence = _extract_confidence(result)

        self._record(input_data, output_data, confidence, latency_ms)
        return result

    def _record(
        self,
        input_data: dict,
        output_data: dict,
        confidence: float | None,
        latency_ms: float,
    ) -> None:
        try:
            task = getattr(self._pipeline, "task", "unknown")
            metadata = {"provider": "huggingface", "task": task}
            if self._auditor is not None:
                self._auditor.record(
                    self._model_id, input_data, output_data,
                    confidence=confidence,
                    latency_ms=int(latency_ms),
                    metadata=metadata,
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=self._model_id,
                    input_data=input_data,
                    output_data=output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
        except Exception as exc:
            _log.warning("ARIAPipeline: record error: %s", exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    @staticmethod
    def _infer_model_id(pipeline: Any) -> str:
        try:
            cfg = pipeline.model.config
            return getattr(cfg, "_name_or_path", None) or getattr(cfg, "model_type", "hf-model")
        except Exception:
            return "hf-pipeline"
