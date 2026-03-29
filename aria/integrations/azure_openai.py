"""
aria.integrations.azure_openai — Drop-in Azure OpenAI client wrapper for ARIA auditing.

Wraps the official ``openai.AzureOpenAI`` client so that every call to
``chat.completions.create()`` and ``embeddings.create()`` is automatically
audited via ARIA.

Zero config changes required — just replace ``AzureOpenAI()`` with
``ARIAAzureOpenAI()``.

Usage::

    from aria.integrations.azure_openai import ARIAAzureOpenAI

    # Drop-in replacement — same API as openai.AzureOpenAI
    client = ARIAAzureOpenAI(
        auditor=auditor,                              # InferenceAuditor instance
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_version="2024-02-01",
        api_key="...",
        model_id="gpt-4o",                            # overrides model label in records
    )

    response = client.chat.completions.create(
        model="my-gpt4o-deployment",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-app")
    client = ARIAAzureOpenAI(
        aria=aria,
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_version="2024-02-01",
    )

Async support::

    from aria.integrations.azure_openai import ARIAAsyncAzureOpenAI

    client = ARIAAsyncAzureOpenAI(
        aria=aria,
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_version="2024-02-01",
    )
    response = await client.chat.completions.create(...)
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
    """Extract a proxy confidence from Azure OpenAI response logprobs if available."""
    try:
        choice = response.choices[0]
        if hasattr(choice, "logprobs") and choice.logprobs:
            lp = choice.logprobs
            if hasattr(lp, "content") and lp.content:
                import math
                avg_lp = sum(t.logprob for t in lp.content) / len(lp.content)
                return round(math.exp(avg_lp), 4)
    except Exception:
        pass
    return None


def _messages_to_input(messages: list[dict]) -> dict[str, Any]:
    """Convert messages list to a serialisable input dict."""
    return {
        "messages": [
            {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
            for m in (messages or [])
        ]
    }


def _response_to_output(response: Any) -> dict[str, Any]:
    """Convert Azure OpenAI response to a serialisable output dict."""
    try:
        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "model": getattr(response, "model", ""),
            "usage": {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            } if hasattr(response, "usage") and response.usage else {},
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class _ARIAChatCompletions:
    """Wraps openai.resources.chat.completions.Completions for Azure."""

    def __init__(self, original: Any, recorder: "_ARIARecorder") -> None:
        self._orig = original
        self._recorder = recorder

    def create(self, **kwargs) -> Any:
        model_id = self._recorder.model_id or kwargs.get("model", "azure_openai")
        messages = kwargs.get("messages", [])
        t0 = time.time()
        response = self._orig.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response),
            confidence=_extract_confidence(response),
            latency_ms=latency_ms,
            metadata={
                "provider": "azure_openai",
                "model": kwargs.get("model", ""),
                "deployment": kwargs.get("model", ""),
                "temperature": kwargs.get("temperature"),
            },
        )
        return response


class _ARIAChat:
    def __init__(self, original: Any, recorder: "_ARIARecorder") -> None:
        self.completions = _ARIAChatCompletions(original.completions, recorder)


class _ARIAEmbeddings:
    def __init__(self, original: Any, recorder: "_ARIARecorder") -> None:
        self._orig = original
        self._recorder = recorder

    def create(self, **kwargs) -> Any:
        model_id = self._recorder.model_id or kwargs.get("model", "azure_openai-embedding")
        t0 = time.time()
        response = self._orig.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        input_text = kwargs.get("input", "")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        self._recorder.record(
            model_id=model_id,
            input_data={"text": str(input_text)[:500]},
            output_data={"dimensions": len(response.data[0].embedding) if response.data else 0},
            latency_ms=latency_ms,
            metadata={
                "provider": "azure_openai",
                "model": kwargs.get("model", ""),
                "deployment": kwargs.get("model", ""),
            },
        )
        return response


class _ARIARecorder:
    """Shared recording logic for sync and async wrappers."""

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self.model_id = model_id

    def record(self, **kwargs) -> None:
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
            _log.warning("ARIAAzureOpenAI: record error: %s", exc)


class ARIAAzureOpenAI:
    """Drop-in replacement for ``openai.AzureOpenAI`` with automatic ARIA auditing.

    Args:
        auditor:         ``InferenceAuditor`` instance.
        aria:            ``ARIAQuick`` instance (alternative to auditor).
        model_id:        Override for the model_id label in ARIA records.
                         If None, uses the ``model`` field from the API request.
        azure_endpoint:  Azure OpenAI resource endpoint URL, e.g.
                         ``"https://my-resource.openai.azure.com/"``.
        api_version:     Azure OpenAI API version string, e.g. ``"2024-02-01"``.
        **kwargs:        All remaining keyword arguments are forwarded to
                         ``openai.AzureOpenAI()``.

    Raises:
        ImportError: if the ``openai`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
            self._client = openai.AzureOpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "openai package not installed. pip install aria-bsv[openai]"
            )

        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.chat = _ARIAChat(self._client.chat, self._recorder)
        self.embeddings = _ARIAEmbeddings(self._client.embeddings, self._recorder)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying client."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class _ARIAAsyncChatCompletions:
    def __init__(self, original: Any, recorder: "_ARIARecorder") -> None:
        self._orig = original
        self._recorder = recorder

    async def create(self, **kwargs) -> Any:
        model_id = self._recorder.model_id or kwargs.get("model", "azure_openai")
        messages = kwargs.get("messages", [])
        t0 = time.time()
        response = await self._orig.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_response_to_output(response),
            confidence=_extract_confidence(response),
            latency_ms=latency_ms,
            metadata={
                "provider": "azure_openai",
                "model": kwargs.get("model", ""),
                "deployment": kwargs.get("model", ""),
            },
        )
        return response


class _ARIAAsyncChat:
    def __init__(self, original: Any, recorder: "_ARIARecorder") -> None:
        self.completions = _ARIAAsyncChatCompletions(original.completions, recorder)


class ARIAAsyncAzureOpenAI:
    """Async drop-in replacement for ``openai.AsyncAzureOpenAI``.

    Args:
        auditor:         ``InferenceAuditor`` instance.
        aria:            ``ARIAQuick`` instance (alternative to auditor).
        model_id:        Override for the model_id label in ARIA records.
        azure_endpoint:  Azure OpenAI resource endpoint URL.
        api_version:     Azure OpenAI API version string.
        **kwargs:        All remaining keyword arguments are forwarded to
                         ``openai.AsyncAzureOpenAI()``.

    Raises:
        ImportError: if the ``openai`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import openai
            self._client = openai.AsyncAzureOpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "openai package not installed. pip install aria-bsv[openai]"
            )

        self._recorder = _ARIARecorder(auditor=auditor, aria=aria, model_id=model_id)
        self.chat = _ARIAAsyncChat(self._client.chat, self._recorder)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying client."""
        return getattr(self._client, name)
