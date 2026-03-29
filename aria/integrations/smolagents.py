"""
aria.integrations.smolagents — ARIA audit integration for HuggingFace SmolAgents.

SmolAgents is HuggingFace's lightweight agent framework that supports
``CodeAgent`` and ``ToolCallingAgent``. This integration wraps the agent's
``run()`` method and individual tools so that every call is recorded in ARIA.

Usage::

    from smolagents import CodeAgent, HfApiModel
    from aria.integrations.smolagents import ARIASmolAgent

    model = HfApiModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
    agent = CodeAgent(tools=[], model=model)

    audited = ARIASmolAgent(agent=agent, auditor=auditor, model_id="smol-coder")
    result = audited.run("What is 42 * 17?")
    # ↑ Automatically recorded in ARIA

    # Wrap individual tools:
    from aria.integrations.smolagents import ARIAToolWrapper

    wrapped_tool = ARIAToolWrapper(tool=my_tool, auditor=auditor)
    output = wrapped_tool(my_input="hello")

    # Async:
    result = await audited.arun("Compute something")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..auditor import InferenceAuditor
    from ..quick import ARIAQuick


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIASmolAgentsRecorder:
    """Shared recording logic for SmolAgents wrappers."""

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
            _log.warning("ARIASmolAgent: record error: %s", exc)


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

class ARIASmolAgent:
    """Wraps a SmolAgents agent to record every ``run()`` call in ARIA.

    Works with ``smolagents.CodeAgent`` and ``smolagents.ToolCallingAgent``.
    All attributes not handled by the wrapper are proxied to the underlying agent.

    Args:
        agent:    A SmolAgents agent instance (``CodeAgent`` or ``ToolCallingAgent``).
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Model ID for ARIA records. If None, uses the agent class name.

    Raises:
        ImportError: if the ``smolagents`` package is not installed.
    """

    def __init__(
        self,
        agent: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import smolagents  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "smolagents package not installed. pip install aria-bsv[smolagents]"
            )
        self._agent = agent
        self._recorder = _ARIASmolAgentsRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._model_id = model_id or type(agent).__name__

    def run(self, task: str, **kwargs: Any) -> Any:
        """Run the agent on the given task and record to ARIA.

        Args:
            task:     The natural-language task string.
            **kwargs: Extra kwargs forwarded to ``agent.run()``.

        Returns:
            The agent's final answer (string or structured result).
        """
        t0 = time.time()
        result = self._agent.run(task, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=self._model_id,
            input_data={"task": str(task)[:500]},
            output_data={"answer": str(result)[:500]},
            latency_ms=latency_ms,
            metadata={
                "provider": "smolagents",
                "agent_type": type(self._agent).__name__,
            },
        )
        return result

    async def arun(self, task: str, **kwargs: Any) -> Any:
        """Async run the agent on the given task and record to ARIA.

        If the underlying agent exposes ``arun()``, it is awaited directly.
        Otherwise the synchronous ``run()`` is executed in the default thread
        executor to avoid blocking the event loop.

        Args:
            task:     The natural-language task string.
            **kwargs: Extra kwargs forwarded to the agent's run method.

        Returns:
            The agent's final answer.
        """
        t0 = time.time()
        if hasattr(self._agent, "arun"):
            result = await self._agent.arun(task, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._agent.run(task, **kwargs)
            )
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=self._model_id,
            input_data={"task": str(task)[:500]},
            output_data={"answer": str(result)[:500]},
            latency_ms=latency_ms,
            metadata={
                "provider": "smolagents",
                "agent_type": type(self._agent).__name__,
                "async": True,
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying agent."""
        return getattr(self._agent, name)


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------

class ARIAToolWrapper:
    """Wraps an individual SmolAgents tool to record each call in ARIA.

    The wrapper is callable, so it can be used as a drop-in replacement for
    the tool in agent tool lists.

    Args:
        tool:     A SmolAgents tool instance (any callable with a ``name`` attribute).
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Model ID for ARIA records. If None, uses the tool name.

    Raises:
        ImportError: if the ``smolagents`` package is not installed.
    """

    def __init__(
        self,
        tool: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import smolagents  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "smolagents package not installed. pip install aria-bsv[smolagents]"
            )
        self._tool = tool
        self._recorder = _ARIASmolAgentsRecorder(auditor=auditor, aria=aria, model_id=model_id)
        tool_name = getattr(tool, "name", None) or type(tool).__name__
        self._model_id = model_id or tool_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the tool and record to ARIA.

        Returns:
            The tool's result.
        """
        t0 = time.time()
        result = self._tool(*args, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        all_inputs: dict[str, Any] = {f"arg_{i}": str(a)[:500] for i, a in enumerate(args)}
        all_inputs.update({k: str(v)[:500] for k, v in kwargs.items()})

        self._recorder.record(
            model_id=self._model_id,
            input_data=all_inputs,
            output_data={"result": str(result)[:500]},
            latency_ms=latency_ms,
            metadata={
                "provider": "smolagents",
                "tool": getattr(self._tool, "name", type(self._tool).__name__),
            },
        )
        return result

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Async invoke the tool and record to ARIA.

        If the underlying tool exposes ``acall()``, it is awaited directly.
        Otherwise the synchronous ``__call__`` is executed in the thread executor.

        Returns:
            The tool's result.
        """
        t0 = time.time()
        if hasattr(self._tool, "acall"):
            result = await self._tool.acall(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._tool(*args, **kwargs)
            )
        latency_ms = (time.time() - t0) * 1000

        all_inputs: dict[str, Any] = {f"arg_{i}": str(a)[:500] for i, a in enumerate(args)}
        all_inputs.update({k: str(v)[:500] for k, v in kwargs.items()})

        self._recorder.record(
            model_id=self._model_id,
            input_data=all_inputs,
            output_data={"result": str(result)[:500]},
            latency_ms=latency_ms,
            metadata={
                "provider": "smolagents",
                "tool": getattr(self._tool, "name", type(self._tool).__name__),
                "async": True,
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying tool."""
        return getattr(self._tool, name)
