"""
aria.integrations.google_gemini — Drop-in Google Gemini wrapper for ARIA auditing.

Wraps ``google.generativeai.GenerativeModel`` so that every call to
``generate_content()`` and ``generate_content_async()`` is automatically
audited via ARIA.

Zero config changes required — just replace ``GenerativeModel()`` with
``ARIAGemini()``.

Usage::

    from aria.integrations.google_gemini import ARIAGemini

    # Drop-in replacement — same API as google.generativeai.GenerativeModel
    model = ARIAGemini(
        model_name="gemini-1.5-pro",
        auditor=auditor,          # InferenceAuditor instance
        model_id="gemini-prod",   # overrides the model label in ARIA records
    )

    response = model.generate_content("What is BSV?")
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-app")
    model = ARIAGemini(model_name="gemini-1.5-flash", aria=aria)

Async support::

    from aria.integrations.google_gemini import ARIAAsyncGemini

    model = ARIAAsyncGemini(model_name="gemini-1.5-pro", aria=aria)
    response = await model.generate_content_async("Hello!")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_confidence(response: Any) -> float | None:
    """Extract a proxy confidence from Gemini response avg_logprobs if available."""
    try:
        candidate = response.candidates[0]
        avg_lp = getattr(candidate, "avg_logprobs", None)
        if avg_lp is not None:
            import math
            return round(math.exp(avg_lp), 4)
    except Exception:
        pass
    return None


def _contents_to_input(contents: Any) -> dict[str, Any]:
    """Convert Gemini contents argument to a serialisable input dict."""
    return {"prompt": str(contents)[:500]}


def _response_to_output(response: Any) -> dict[str, Any]:
    """Convert Gemini response to a serialisable output dict."""
    try:
        candidate = response.candidates[0]
        usage = getattr(response, "usage_metadata", None)
        return {
            "text": response.text[:500],
            "finish_reason": str(getattr(candidate, "finish_reason", "")),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "candidates_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
            },
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIAGeminiRecorder:
    """Shared recording logic for sync and async Gemini wrappers."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs: Any) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    kwargs["model_id"],
                    kwargs["input_data"],
                    kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=int(kwargs.get("latency_ms") or 0),
                    metadata=kwargs.get("metadata") or {},
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=kwargs["model_id"],
                    input_data=kwargs["input_data"],
                    output_data=kwargs["output_data"],
                    confidence=kwargs.get("confidence"),
                    latency_ms=kwargs.get("latency_ms"),
                    metadata=kwargs.get("metadata") or {},
                )
        except Exception as exc:
            _log.warning("ARIAGemini: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class ARIAGemini:
    """Drop-in replacement for ``google.generativeai.GenerativeModel`` with ARIA auditing.

    Args:
        model_name: Gemini model name (e.g. ``"gemini-1.5-pro"``).
        auditor:    ``InferenceAuditor`` instance.
        aria:       ``ARIAQuick`` instance (alternative to auditor).
        model_id:   Override for the model_id label in ARIA records.
                    If None, uses ``model_name``.
        **kwargs:   All keyword arguments are forwarded to ``GenerativeModel()``.

    Raises:
        ImportError: if ``google-generativeai`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import]
            self._model = genai.GenerativeModel(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "pip install aria-bsv[google-gemini]"
            )

        self._model_name = model_name
        self._recorder = _ARIAGeminiRecorder(auditor=auditor, aria=aria, model_id=model_id)

    def generate_content(self, contents: Any, **kwargs: Any) -> Any:
        """Generate content and record the inference in ARIA."""
        model_id = self._recorder.model_id or self._model_name
        t0 = time.time()
        response = self._model.generate_content(contents, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_contents_to_input(contents),
            output_data=_response_to_output(response),
            confidence=_extract_confidence(response),
            latency_ms=latency_ms,
            metadata={"provider": "google_gemini", "model": self._model_name},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying GenerativeModel."""
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class ARIAAsyncGemini:
    """Async drop-in replacement for ``google.generativeai.GenerativeModel``.

    Args:
        model_name: Gemini model name (e.g. ``"gemini-1.5-pro"``).
        auditor:    ``InferenceAuditor`` instance.
        aria:       ``ARIAQuick`` instance (alternative to auditor).
        model_id:   Override for the model_id label in ARIA records.
        **kwargs:   All keyword arguments are forwarded to ``GenerativeModel()``.

    Raises:
        ImportError: if ``google-generativeai`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import]
            self._model = genai.GenerativeModel(model_name=model_name, **kwargs)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "pip install aria-bsv[google-gemini]"
            )

        self._model_name = model_name
        self._recorder = _ARIAGeminiRecorder(auditor=auditor, aria=aria, model_id=model_id)

    async def generate_content_async(self, contents: Any, **kwargs: Any) -> Any:
        """Async generate content and record the inference in ARIA."""
        model_id = self._recorder.model_id or self._model_name
        t0 = time.time()
        response = await self._model.generate_content_async(contents, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_contents_to_input(contents),
            output_data=_response_to_output(response),
            confidence=_extract_confidence(response),
            latency_ms=latency_ms,
            metadata={"provider": "google_gemini", "model": self._model_name},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying GenerativeModel."""
        return getattr(self._model, name)
