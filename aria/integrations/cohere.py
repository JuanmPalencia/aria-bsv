"""
aria.integrations.cohere — Drop-in Cohere client wrapper for ARIA auditing.

Wraps the official ``cohere`` V2 client so that every call to ``chat()``
and ``embed()`` is automatically audited via ARIA.

Zero config changes required — just replace ``cohere.ClientV2()`` with
``ARIACohere()``.

Usage::

    from aria.integrations.cohere import ARIACohere

    # Drop-in replacement — same API as cohere.ClientV2
    client = ARIACohere(
        api_key="...",
        auditor=auditor,        # InferenceAuditor instance
        model_id="command-r",   # overrides the model label in ARIA records
    )

    response = client.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # ↑ This call is automatically recorded in ARIA

    # Or use ARIAQuick for zero-setup:
    from aria.quick import ARIAQuick
    aria = ARIAQuick("my-app")
    client = ARIACohere(api_key="...", aria=aria)

Async support::

    from aria.integrations.cohere import ARIAAsyncCohere

    client = ARIAAsyncCohere(api_key="...", aria=aria)
    response = await client.chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Notes:
    - Cohere V2 API does not expose per-token logprobs, so ``confidence``
      is always ``None``.
    - Only the first 3 texts are captured in embed input metadata to keep
      record sizes manageable.
    - ``response.message.content[0].text`` is truncated to 500 characters
      in the output metadata.

Raises:
    ImportError: if the ``cohere`` package is not installed.
                 Install with ``pip install aria-bsv[cohere]``.
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

def _messages_to_input(messages: list[Any]) -> dict[str, Any]:
    """Convert a Cohere V2 messages list to a serialisable input dict."""
    serialised: list[dict[str, str]] = []
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = str(m.get("content", ""))[:500]
        else:
            role = getattr(m, "role", "")
            content = str(getattr(m, "content", ""))[:500]
        serialised.append({"role": role, "content": content})
    return {"messages": serialised}


def _chat_response_to_output(response: Any) -> dict[str, Any]:
    """Convert a Cohere V2 chat response to a serialisable output dict."""
    try:
        text = ""
        try:
            text = response.message.content[0].text[:500]
        except Exception:
            pass

        finish_reason = getattr(response, "finish_reason", "")

        usage: dict[str, Any] = {}
        try:
            bu = response.usage.billed_units
            usage = {
                "input_tokens": getattr(bu, "input_tokens", 0),
                "output_tokens": getattr(bu, "output_tokens", 0),
            }
        except Exception:
            pass

        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage,
        }
    except Exception:
        return {"raw": str(response)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIACohereRecorder:
    """Shared recording logic for sync and async Cohere wrappers."""

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
            _log.warning("ARIACohere: record error: %s", exc)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class ARIACohere:
    """Drop-in replacement for ``cohere.ClientV2`` with automatic ARIA auditing.

    Intercepts ``chat()`` and ``embed()`` calls and records them via ARIA.
    All other attribute accesses are proxied to the underlying ``ClientV2``
    instance unchanged.

    Args:
        auditor:   ``InferenceAuditor`` instance.
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        model_id:  Override for the model_id label stored in ARIA records.
                   If ``None``, the ``model`` argument from each call is used.
        **kwargs:  All remaining keyword arguments are forwarded to
                   ``cohere.ClientV2()``.  Pass ``api_key`` here.

    Raises:
        ImportError: if the ``cohere`` package is not installed.
                     Run ``pip install aria-bsv[cohere]`` to fix this.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import cohere
            self._client = cohere.ClientV2(**kwargs)
        except ImportError:
            raise ImportError(
                "cohere package not installed. pip install aria-bsv[cohere]"
            )

        self._recorder = _ARIACohereRecorder(auditor=auditor, aria=aria, model_id=model_id)

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Call ``ClientV2.chat()`` and record the inference in ARIA.

        Args:
            messages:  Conversation history (list of message dicts or objects).
            model:     Cohere model name (e.g. ``"command-r-plus"``).
            **kwargs:  Additional arguments forwarded to the underlying client.

        Returns:
            The original Cohere chat response object, unmodified.
        """
        model_id = self._recorder.model_id or model
        t0 = time.time()
        response = self._client.chat(messages=messages, model=model, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_chat_response_to_output(response),
            confidence=None,
            latency_ms=latency_ms,
            metadata={"provider": "cohere", "model": model},
        )
        return response

    def embed(
        self,
        texts: list[str],
        model: str,
        input_type: str,
        **kwargs: Any,
    ) -> Any:
        """Call ``ClientV2.embed()`` and record the inference in ARIA.

        Args:
            texts:       List of strings to embed.
            model:       Cohere embedding model name.
            input_type:  Cohere input type (e.g. ``"search_document"``).
            **kwargs:    Additional arguments forwarded to the underlying client.

        Returns:
            The original Cohere embed response object, unmodified.
        """
        model_id = self._recorder.model_id or model
        t0 = time.time()
        response = self._client.embed(
            texts=texts, model=model, input_type=input_type, **kwargs
        )
        latency_ms = (time.time() - t0) * 1000

        dimensions = 0
        try:
            dimensions = len(response.embeddings.float[0]) if response.embeddings.float else 0
        except Exception:
            pass

        self._recorder.record(
            model_id=model_id,
            input_data={"texts": texts[:3], "input_type": input_type},
            output_data={"dimensions": dimensions},
            confidence=None,
            latency_ms=latency_ms,
            metadata={"provider": "cohere", "model": model},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying ``ClientV2`` instance."""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

class ARIAAsyncCohere:
    """Async drop-in replacement for ``cohere.AsyncClientV2`` with ARIA auditing.

    Intercepts ``chat()`` and ``embed()`` coroutines and records them via ARIA.

    Args:
        auditor:   ``InferenceAuditor`` instance.
        aria:      ``ARIAQuick`` instance (alternative to *auditor*).
        model_id:  Override for the model_id label stored in ARIA records.
        **kwargs:  Forwarded to ``cohere.AsyncClientV2()``.  Pass ``api_key`` here.

    Raises:
        ImportError: if the ``cohere`` package is not installed.
                     Run ``pip install aria-bsv[cohere]`` to fix this.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import cohere
            self._client = cohere.AsyncClientV2(**kwargs)
        except ImportError:
            raise ImportError(
                "cohere package not installed. pip install aria-bsv[cohere]"
            )

        self._recorder = _ARIACohereRecorder(auditor=auditor, aria=aria, model_id=model_id)

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Any],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Async call to ``AsyncClientV2.chat()`` with ARIA recording.

        Args:
            messages:  Conversation history.
            model:     Cohere model name.
            **kwargs:  Additional arguments forwarded to the underlying client.

        Returns:
            The original Cohere chat response object, unmodified.
        """
        model_id = self._recorder.model_id or model
        t0 = time.time()
        response = await self._client.chat(messages=messages, model=model, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=model_id,
            input_data=_messages_to_input(messages),
            output_data=_chat_response_to_output(response),
            confidence=None,
            latency_ms=latency_ms,
            metadata={"provider": "cohere", "model": model},
        )
        return response

    async def embed(
        self,
        texts: list[str],
        model: str,
        input_type: str,
        **kwargs: Any,
    ) -> Any:
        """Async call to ``AsyncClientV2.embed()`` with ARIA recording.

        Args:
            texts:       List of strings to embed.
            model:       Cohere embedding model name.
            input_type:  Cohere input type.
            **kwargs:    Additional arguments forwarded to the underlying client.

        Returns:
            The original Cohere embed response object, unmodified.
        """
        model_id = self._recorder.model_id or model
        t0 = time.time()
        response = await self._client.embed(
            texts=texts, model=model, input_type=input_type, **kwargs
        )
        latency_ms = (time.time() - t0) * 1000

        dimensions = 0
        try:
            dimensions = len(response.embeddings.float[0]) if response.embeddings.float else 0
        except Exception:
            pass

        self._recorder.record(
            model_id=model_id,
            input_data={"texts": texts[:3], "input_type": input_type},
            output_data={"dimensions": dimensions},
            confidence=None,
            latency_ms=latency_ms,
            metadata={"provider": "cohere", "model": model},
        )
        return response

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying ``AsyncClientV2`` instance."""
        return getattr(self._client, name)
