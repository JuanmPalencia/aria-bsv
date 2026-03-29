"""
aria.integrations.crewai — ARIA audit integration for CrewAI.

Hooks into CrewAI multi-agent task execution, recording each agent task
and tool action as an AuditRecord.  No CrewAI import is required at module
load time — this is a pure Python adapter.

Usage::

    from aria.integrations.crewai import ARIACrewCallback
    from aria.quick import ARIAQuick

    aria = ARIAQuick(model_id="crewai-agent", db_path="aria.db")
    callback = ARIACrewCallback(aria=aria)

    # Instrument task execution:
    callback.on_task_start("Analyse the dataset", agent_role="Data Analyst")
    result = agent.execute_task(task)
    callback.on_task_end("Analyse the dataset", agent_role="Data Analyst", output=result)

    # Tool use is recorded separately:
    callback.on_agent_action("Data Analyst", action="search_web", input_str="BSV price")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


class ARIACrewCallback:
    """Callback handler for CrewAI multi-agent task execution.

    Audits every agent task and tool action via ARIA.  Each task start/end
    pair produces one AuditRecord with:

    - ``model_id``: ``f"{model_id}/{agent_role}"``
    - ``input_data``: ``{"task": task_description[:400], "agent": agent_role}``
    - ``output_data``: ``{"output": output[:400], "agent": agent_role}``
    - ``latency_ms``: wall-clock milliseconds between ``on_task_start`` and
      ``on_task_end``, keyed by ``f"{task_description[:40]}:{agent_role}"``.
    - ``metadata``: ``{"provider": "crewai", "agent_role": agent_role}``

    Args:
        auditor:  An initialised ``InferenceAuditor`` instance.
        aria:     An ``ARIAQuick`` instance (alternative to *auditor*).
        model_id: Base label prepended to each agent role.  Defaults to
                  ``"crewai-agent"``.

    Example::

        callback = ARIACrewCallback(aria=aria, model_id="my-crew")
        callback.on_task_start("Write a blog post", agent_role="Writer")
        # ... agent runs ...
        callback.on_task_end("Write a blog post", agent_role="Writer",
                             output="Here is the post...")
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str = "crewai-agent",
    ) -> None:
        self._auditor = auditor
        self._aria = aria
        self._model_id = model_id
        # Keyed by "{task_description[:40]}:{agent_role}" → monotonic start time
        self._start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Task-level hooks
    # ------------------------------------------------------------------

    def on_task_start(self, task_description: str, agent_role: str) -> None:
        """Record the start time for a task execution.

        Args:
            task_description: Human-readable description of the task.
            agent_role:       Role name of the agent executing the task.
        """
        key = f"{task_description[:40]}:{agent_role}"
        self._start_times[key] = time.monotonic()

    def on_task_end(
        self,
        task_description: str,
        agent_role: str,
        output: str,
    ) -> None:
        """Record the completed task execution as an AuditRecord.

        Args:
            task_description: Human-readable description of the task.
            agent_role:       Role name of the agent that executed the task.
            output:           Text output produced by the agent.
        """
        key = f"{task_description[:40]}:{agent_role}"
        start = self._start_times.pop(key, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0

        input_data: dict[str, Any] = {
            "task": task_description[:400],
            "agent": agent_role,
        }
        output_data: dict[str, Any] = {
            "output": output[:400],
            "agent": agent_role,
        }
        metadata: dict[str, Any] = {
            "provider": "crewai",
            "agent_role": agent_role,
        }
        self._record(
            f"{self._model_id}/{agent_role}",
            input_data,
            output_data,
            None,
            latency_ms,
            metadata,
        )

    # ------------------------------------------------------------------
    # Agent-level hooks
    # ------------------------------------------------------------------

    def on_agent_action(
        self,
        agent_role: str,
        action: str,
        input_str: str,
    ) -> None:
        """Record an agent tool-use action as an AuditRecord.

        Args:
            agent_role: Role name of the agent performing the action.
            action:     Name of the tool or action being invoked.
            input_str:  Input string passed to the tool.
        """
        input_data: dict[str, Any] = {
            "agent": agent_role,
            "action": action,
            "input": input_str[:400],
        }
        output_data: dict[str, Any] = {
            "agent": agent_role,
            "action": action,
        }
        metadata: dict[str, Any] = {
            "provider": "crewai",
            "agent_role": agent_role,
            "event": "agent_action",
        }
        self._record(
            f"{self._model_id}/{agent_role}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

    def on_agent_finish(self, agent_role: str, output: str) -> None:
        """Record an agent's final output as an AuditRecord.

        Args:
            agent_role: Role name of the agent that finished.
            output:     Final output text produced by the agent.
        """
        input_data: dict[str, Any] = {"agent": agent_role}
        output_data: dict[str, Any] = {
            "output": output[:400],
            "agent": agent_role,
        }
        metadata: dict[str, Any] = {
            "provider": "crewai",
            "agent_role": agent_role,
            "event": "agent_finish",
        }
        self._record(
            f"{self._model_id}/{agent_role}",
            input_data,
            output_data,
            None,
            0,
            metadata,
        )

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
            _log.warning("ARIACrewCallback: record error: %s", exc)
