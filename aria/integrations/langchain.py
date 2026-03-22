"""
aria.integrations.langchain — ARIA audit integration for LangChain/LangGraph.

Usage:
    from aria.integrations.langchain import ARIACallbackHandler, ARIAAuditedLLM

    handler = ARIACallbackHandler(auditor=auditor, model_id="gpt-4-audit")
    llm = ChatOpenAI(callbacks=[handler])

    # Or use the wrapper directly:
    audited = ARIAAuditedLLM(llm=llm, auditor=auditor, model_id="gpt-4-audit")
    result = audited.invoke("What is the EU AI Act?")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Union
from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core: pip install aria-bsv[langchain]"
    )

if TYPE_CHECKING:
    from aria.auditor import InferenceAuditor


class ARIACallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records every LLM call to ARIA.

    Attach to any LangChain LLM, Chat model, or chain. Each invocation
    is recorded as an AuditRecord with:
    - input_hash: SHA-256 of the canonical prompt/messages
    - output_hash: SHA-256 of the canonical response text
    - latency_ms: wall-clock time of the LLM call
    - confidence: token log-probability if available, else None
    - metadata: model name, finish reason, token usage

    Args:
        auditor:   An initialised InferenceAuditor instance.
        model_id:  The model_id to use for AuditRecords.  MUST be a key in
                   auditor's model_hashes.
        pii_strip: Additional field names to strip from prompt metadata.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor",
        model_id: str,
        pii_strip: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._auditor = auditor
        self._model_id = model_id
        self._pii_strip = set(pii_strip or [])
        # Per-run state keyed by run_id
        self._start_times: dict[str, float] = {}
        self._inputs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._start_times[key] = time.monotonic()
        self._inputs[key] = {"prompts": prompts, "model": serialized.get("name", self._model_id)}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._start_times[key] = time.monotonic()
        # Represent messages as list of dicts for canonical hashing
        msg_dicts = [
            [{"role": m.type, "content": m.content} for m in batch]
            for batch in messages
        ]
        self._inputs[key] = {"messages": msg_dicts, "model": serialized.get("name", self._model_id)}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        start = self._start_times.pop(key, None)
        inp = self._inputs.pop(key, {})
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0

        # Extract output text and metadata
        generations = response.generations
        if not generations:
            return

        output_texts = [
            g.text if hasattr(g, "text") else str(g)
            for batch in generations
            for g in batch
        ]

        # Extract confidence from token log-probs if available
        confidence: float | None = None
        llm_output = response.llm_output or {}
        if "token_usage" in llm_output:
            # Some providers expose log-prob-derived confidence
            usage = llm_output["token_usage"]
            if "logprob_confidence" in usage:
                confidence = float(usage["logprob_confidence"])

        metadata = {
            "model": inp.get("model", self._model_id),
            "finish_reason": llm_output.get("finish_reason"),
            "token_usage": llm_output.get("token_usage"),
        }
        # Strip PII metadata fields
        for field in self._pii_strip:
            metadata.pop(field, None)

        output = {"texts": output_texts}

        self._auditor.record(
            self._model_id,
            inp,
            output,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._start_times.pop(key, None)
        self._inputs.pop(key, None)


class ARIAAuditedLLM:
    """Thin wrapper that attaches ARIACallbackHandler to any LangChain Runnable.

    Usage:
        from langchain_openai import ChatOpenAI
        from aria.integrations.langchain import ARIAAuditedLLM

        llm = ChatOpenAI(model="gpt-4o")
        audited = ARIAAuditedLLM(llm=llm, auditor=auditor, model_id="gpt-4o")
        result = audited.invoke("Explain ARIA protocol")
    """

    def __init__(
        self,
        llm: Any,
        auditor: "InferenceAuditor",
        model_id: str,
        pii_strip: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._handler = ARIACallbackHandler(
            auditor=auditor, model_id=model_id, pii_strip=pii_strip
        )

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        cfg = config or {}
        existing = cfg.get("callbacks", [])
        cfg["callbacks"] = list(existing) + [self._handler]
        return self._llm.invoke(input, config=cfg, **kwargs)

    async def ainvoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        cfg = config or {}
        existing = cfg.get("callbacks", [])
        cfg["callbacks"] = list(existing) + [self._handler]
        return await self._llm.ainvoke(input, config=cfg, **kwargs)

    def stream(self, input: Any, config: dict | None = None, **kwargs: Any):
        cfg = config or {}
        existing = cfg.get("callbacks", [])
        cfg["callbacks"] = list(existing) + [self._handler]
        yield from self._llm.stream(input, config=cfg, **kwargs)

    def batch(self, inputs: list[Any], config: dict | None = None, **kwargs: Any) -> list[Any]:
        cfg = config or {}
        existing = cfg.get("callbacks", [])
        cfg["callbacks"] = list(existing) + [self._handler]
        return self._llm.batch(inputs, config=cfg, **kwargs)
