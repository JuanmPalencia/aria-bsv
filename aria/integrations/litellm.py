"""
aria.integrations.litellm — ARIA audit integration for LiteLLM.

LiteLLM is a unified interface to 100+ LLM providers (OpenAI, Anthropic,
Gemini, Cohere, etc.). This integration wraps ``litellm.completion`` and
``litellm.acompletion``, and provides a ``make_litellm_callback`` factory
that returns a function compatible with litellm's ``success_callback`` list.

Usage::

    from aria.integrations.litellm import ARIALiteLLM

    client = ARIALiteLLM(auditor=auditor, model_id="gpt-4o")
    response = client.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is BSV?"}],
    )
    # ↑ Automatically recorded in ARIA

    # Async:
    response = await client.acompletion(model="gpt-4o", messages=[...])

    # Or register as a litellm success_callback:
    import litellm
    from aria.integrations.litellm import make_litellm_callback

    litellm.success_callback = [make_litellm_callback(auditor=auditor)]
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

def _messages_to_input(messages: list[dict]) -> dict[str, Any]:
    """Convert messages list to a serialisable input dict."""
    return {
        "messages": [
            {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
            for m in (messages or [])
        ]
    }


def _response_to_output(response: Any) -> dict[str, Any]:
    """Convert a LiteLLM ModelResponse to a serialisable output dict."""
    try:
        choice = response.choices[0]
        return {
            "content": getattr(choice.message, "content", "") or "",
            "finish_reason": getattr(choice, "finish_reason", "") or "",
            "model": getattr(response, "model", "") or "",
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            } if hasattr(response, "usage") and response.usage else {},
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIALiteLLMRecorder:
    """Shared recording logic for LiteLLM wrappers."""

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
            _log.warning("ARIALiteLLM: record error: %s", exc)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class ARIALiteLLM:
    """ARIA-audited wrapper around ``litellm.completion`` and ``litellm.acompletion``.

    Every call records an AuditRecord with:
    - input_hash: SHA-256 of the canonical messages
    - output_hash: SHA-256 of the canonical response content
    - latency_ms: wall-clock time of the call
    - metadata: provider, model, temperature

    Args:
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Override for the model_id label in ARIA records.
                  If None, uses the ``model`` argument from each call.
        **kwargs: Default keyword arguments merged into every completion call.

    Raises:
        ImportError: if the ``litellm`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm as _litellm  # type: ignore[import]
            self._litellm = _litellm
        except ImportError:
            raise ImportError(
                "litellm package not installed. pip install aria-bsv[litellm]"
            )
        self._recorder = _ARIALiteLLMRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._default_kwargs = kwargs

    def completion(self, model: str, messages: list[dict], **kwargs: Any) -> Any:
        """Call ``litellm.completion`` and record to ARIA.

        Args:
            model:    LiteLLM model string (e.g. ``"gpt-4o"``, ``"claude-3-opus-20240229"``).
            messages: OpenAI-style message list.
            **kwargs: Extra kwargs forwarded to ``litellm.completion``.

        Returns:
            The LiteLLM ``ModelResponse`` object.
        """
        model_id = self._recorder.model_id or model
        merged = {**self._default_kwargs, **kwargs}
        t0 = time.time()
        response = self._litellm.completion(model=model, messages=messages, **merged)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata={
                "provider": "litellm",
                "model": model,
                "temperature": merged.get("temperature"),
            },
        )
        return response

    async def acompletion(self, model: str, messages: list[dict], **kwargs: Any) -> Any:
        """Async call to ``litellm.acompletion`` and record to ARIA.

        Args:
            model:    LiteLLM model string.
            messages: OpenAI-style message list.
            **kwargs: Extra kwargs forwarded to ``litellm.acompletion``.

        Returns:
            The LiteLLM ``ModelResponse`` object.
        """
        model_id = self._recorder.model_id or model
        merged = {**self._default_kwargs, **kwargs}
        t0 = time.time()
        response = await self._litellm.acompletion(model=model, messages=messages, **merged)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response),
            latency_ms=latency_ms,
            metadata={
                "provider": "litellm",
                "model": model,
                "temperature": merged.get("temperature"),
            },
        )
        return response


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def make_litellm_callback(
    auditor: "InferenceAuditor | None" = None,
    aria: "ARIAQuick | None" = None,
    model_id: str | None = None,
) -> Any:
    """Return a litellm ``success_callback`` function that records to ARIA.

    The returned callable conforms to the litellm success_callback protocol::

        import litellm
        litellm.success_callback = [make_litellm_callback(auditor=auditor)]

    Args:
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Override model_id label in ARIA records.

    Returns:
        A callable compatible with the litellm ``success_callback`` list.
    """
    recorder = _ARIALiteLLMRecorder(auditor=auditor, aria=aria, model_id=model_id)

    def litellm_callback(
        kwargs: dict,
        completion_response: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """LiteLLM success_callback that records each call to ARIA."""
        try:
            model = kwargs.get("model", "unknown")
            mid = recorder.model_id or model
            messages = kwargs.get("messages", [])
            # start_time / end_time are datetime objects in litellm
            try:
                latency_ms = (end_time - start_time).total_seconds() * 1000
            except Exception:
                latency_ms = 0.0
            recorder.record(
                model_id=mid,
                input_data=_messages_to_input(messages),
                output_data=_response_to_output(completion_response),
                latency_ms=latency_ms,
                metadata={"provider": "litellm", "model": model},
            )
        except Exception as exc:
            _log.warning("litellm_callback: record error: %s", exc)

    return litellm_callback
