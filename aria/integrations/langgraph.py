"""
aria.integrations.langgraph — ARIA audit integration for LangGraph.

Hooks into LangGraph state-graph execution at the node level, recording
every node invocation as an AuditRecord.  No LangGraph import is required
at module load time — this is a pure Python adapter.

Usage::

    from aria.integrations.langgraph import ARIALangGraphCallback
    from aria.quick import ARIAQuick

    aria = ARIAQuick(model_id="langgraph-agent", db_path="aria.db")
    callback = ARIALangGraphCallback(aria=aria)

    # Wire into your graph's event loop manually, e.g.:
    #   callback.on_node_start("my_node", inputs, run_id="run-1")
    #   result = my_node_fn(inputs)
    #   callback.on_node_end("my_node", result, run_id="run-1")

    # Or use as a base class to satisfy a specific LangGraph hook interface.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


class ARIALangGraphCallback:
    """Callback handler for LangGraph state-graph execution.

    Audits every node invocation via ARIA.  Each node start/end pair
    produces one AuditRecord with:

    - ``model_id``: ``f"{model_id}/{node_name}"``
    - ``input_data``: ``{"node": node_name, "inputs": repr(inputs)[:400]}``
    - ``output_data``: ``{"node": node_name, "outputs": repr(outputs)[:400]}``
    - ``latency_ms``: wall-clock milliseconds between ``on_node_start`` and
      ``on_node_end`` for the same *run_id*.
    - ``metadata``: ``{"provider": "langgraph", "node": node_name, "run_id": run_id}``

    Args:
        auditor:  An initialised ``InferenceAuditor`` instance.
        aria:     An ``ARIAQuick`` instance (alternative to *auditor*).
        model_id: Base label prepended to each node name.  Defaults to
                  ``"langgraph-agent"``.

    Example::

        callback = ARIALangGraphCallback(aria=aria, model_id="my-graph")
        callback.on_graph_start("graph-abc", {"query": "hello"})
        callback.on_node_start("retrieve", {"query": "hello"}, run_id="r1")
        # ... node executes ...
        callback.on_node_end("retrieve", {"docs": [...]}, run_id="r1")
        callback.on_graph_end("graph-abc", {"answer": "world"})
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str = "langgraph-agent",
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self._model_id = model_id
        # Keyed by run_id → monotonic start time
        self._start_times: dict[str, float] = {}
        # Keyed by run_id → (node_name, inputs_repr) captured at on_node_start
        self._pending: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Node-level hooks
    # ------------------------------------------------------------------

    def on_node_start(
        self,
        node_name: str,
        inputs: dict,
        run_id: str,
    ) -> None:
        """Record the start time for a node execution.

        Args:
            node_name: Name of the LangGraph node being entered.
            inputs:    State/input dict passed to the node.
            run_id:    Unique identifier for this execution run.
        """
        self._start_times[run_id] = time.monotonic()
        self._pending[run_id] = (node_name, repr(inputs)[:400])

    def on_node_end(
        self,
        node_name: str,
        outputs: dict,
        run_id: str,
    ) -> None:
        """Record the completed node execution as an AuditRecord.

        Args:
            node_name: Name of the LangGraph node that finished.
            outputs:   State/output dict returned by the node.
            run_id:    Must match the *run_id* from the corresponding
                       ``on_node_start`` call.
        """
        start = self._start_times.pop(run_id, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0

        pending = self._pending.pop(run_id, None)
        inputs_repr = pending[1] if pending is not None else ""

        input_data: dict[str, Any] = {
            "node": node_name,
            "inputs": inputs_repr,
        }
        output_data: dict[str, Any] = {
            "node": node_name,
            "outputs": repr(outputs)[:400],
        }
        metadata: dict[str, Any] = {
            "provider": "langgraph",
            "node": node_name,
            "run_id": run_id,
        }
        self._record(
            f"{self._model_id}/{node_name}",
            input_data,
            output_data,
            None,
            latency_ms,
            metadata,
        )

    def on_error(
        self,
        node_name: str,
        error: Exception,
        run_id: str,
    ) -> None:
        """Record a node execution that raised an exception.

        Args:
            node_name: Name of the node that raised the error.
            error:     The exception that was raised.
            run_id:    Execution run identifier.
        """
        start = self._start_times.pop(run_id, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0
        self._pending.pop(run_id, None)

        input_data: dict[str, Any] = {"node": node_name, "inputs": ""}
        output_data: dict[str, Any] = {
            "node": node_name,
            "error": type(error).__name__,
            "message": str(error)[:400],
        }
        metadata: dict[str, Any] = {
            "provider": "langgraph",
            "node": node_name,
            "run_id": run_id,
            "status": "error",
        }
        self._record(
            f"{self._model_id}/{node_name}",
            input_data,
            output_data,
            None,
            latency_ms,
            metadata,
        )

    # ------------------------------------------------------------------
    # Graph-level hooks
    # ------------------------------------------------------------------

    def on_graph_start(self, graph_id: str, inputs: dict) -> None:
        """Emit a trace-start log entry when the graph begins execution.

        Args:
            graph_id: Identifier for the graph instance.
            inputs:   Initial inputs to the graph.
        """
        _log.info("ARIALangGraphCallback: graph start graph_id=%s", graph_id)

    def on_graph_end(self, graph_id: str, outputs: dict) -> None:
        """Emit a trace-end log entry when the graph finishes execution.

        Args:
            graph_id: Identifier for the graph instance.
            outputs:  Final outputs produced by the graph.
        """
        _log.info("ARIALangGraphCallback: graph end graph_id=%s", graph_id)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _record(
        self,
        model_id: str,
        input_data: dict,
        output_data: dict,
        confidence: float | None,
        latency_ms: int,
        metadata: dict,
    ) -> None:
        try:
            if self._auditor is not None:
                self._auditor.record(
                    model_id,
                    input_data,
                    output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
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
            _log.warning("ARIALangGraphCallback: record error: %s", exc)
