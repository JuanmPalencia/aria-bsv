"""
aria.integrations.llamaindex — LlamaIndex callback handler for ARIA.

Hooks into LlamaIndex's ``CallbackManager`` to capture every LLM call,
embedding, and query engine execution, auditing them via ARIA.

Usage::

    from aria.integrations.llamaindex import ARIACallbackHandler
    from llama_index.core import Settings

    # Register ARIA as a callback handler
    handler = ARIACallbackHandler(aria=aria)
    Settings.callback_manager = CallbackManager([handler])

    # Now all LLM + embedding calls are automatically audited
    index = VectorStoreIndex.from_documents(docs)
    response = index.as_query_engine().query("What is BSV?")

Also works with older LlamaIndex (v0.8.x) via ``ServiceContext``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


class ARIACallbackHandler:
    """LlamaIndex ``BaseCallbackHandler`` subclass that audits via ARIA.

    Args:
        auditor:    ``InferenceAuditor`` instance.
        aria:       ``ARIAQuick`` instance (alternative to auditor).
        model_id:   Default model label. If None, auto-detected from event.
    """

    # LlamaIndex event types we care about
    _LLM_EVENTS    = {"llm_predict", "llm_chat", "LLM_PREDICT", "LLM_CHAT"}
    _EMBED_EVENTS  = {"embedding", "EMBEDDING"}
    _QUERY_EVENTS  = {"query", "QUERY"}

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self._model_id = model_id
        self._start_times: dict[str, float] = {}

        # Try to register as a proper BaseCallbackHandler
        self._base_class = self._get_base_class()

    def _get_base_class(self) -> Any:
        try:
            from llama_index.core.callbacks import BaseCallbackHandler  # v0.10+
            return BaseCallbackHandler
        except ImportError:
            try:
                from llama_index.callbacks import BaseCallbackHandler  # v0.8/0.9
                return BaseCallbackHandler
            except ImportError:
                return object

    # ------------------------------------------------------------------
    # LlamaIndex callback protocol
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: dict | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        self._start_times[event_id] = time.time()
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: dict | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        t0 = self._start_times.pop(event_id, time.time())
        latency_ms = (time.time() - t0) * 1000
        event_name = str(event_type.value if hasattr(event_type, "value") else event_type)
        payload = payload or {}

        if event_name.upper() in {e.upper() for e in self._LLM_EVENTS}:
            self._record_llm(event_name, payload, latency_ms)
        elif event_name.upper() in {e.upper() for e in self._EMBED_EVENTS}:
            self._record_embedding(payload, latency_ms)
        elif event_name.upper() in {e.upper() for e in self._QUERY_EVENTS}:
            self._record_query(payload, latency_ms)

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict | None = None,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _record_llm(self, event_name: str, payload: dict, latency_ms: float) -> None:
        messages = payload.get("messages", [])
        response = payload.get("response", payload.get("output", ""))

        input_data = {
            "messages": [
                {"role": getattr(m, "role", ""), "content": str(getattr(m, "content", m))[:500]}
                for m in (messages if isinstance(messages, list) else [])
            ]
        }
        output_data = {
            "content": str(getattr(response, "message", response))[:1000] if response else ""
        }

        model_id = (
            self._model_id
            or str(payload.get("model_name", payload.get("model", "llamaindex-llm")))
        )
        self._record(model_id, input_data, output_data, None, latency_ms,
                     {"provider": "llamaindex", "event": event_name})

    def _record_embedding(self, payload: dict, latency_ms: float) -> None:
        chunks = payload.get("chunks", [])
        self._record(
            self._model_id or "llamaindex-embedding",
            {"chunks_count": len(chunks), "sample": str(chunks[0])[:200] if chunks else ""},
            {"embeddings_count": len(payload.get("embeddings", []))},
            None, latency_ms,
            {"provider": "llamaindex", "event": "embedding"},
        )

    def _record_query(self, payload: dict, latency_ms: float) -> None:
        query_str = str(payload.get("query_str", payload.get("query", "")))[:500]
        response = str(payload.get("response", ""))[:1000]
        self._record(
            self._model_id or "llamaindex-query",
            {"query": query_str},
            {"response": response},
            None, latency_ms,
            {"provider": "llamaindex", "event": "query"},
        )

    def _record(
        self,
        model_id: str,
        input_data: dict,
        output_data: dict,
        confidence: float | None,
        latency_ms: float,
        metadata: dict,
    ) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    model_id, input_data, output_data,
                    confidence=confidence,
                    latency_ms=int(latency_ms),
                    metadata=metadata,
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=model_id,
                    input_data=input_data,
                    output_data=output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    metadata=metadata,
                )
        except Exception as exc:
            _log.warning("ARIACallbackHandler: record error: %s", exc)
