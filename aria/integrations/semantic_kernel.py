"""
aria.integrations.semantic_kernel — ARIA audit integration for Microsoft Semantic Kernel.

Semantic Kernel is Microsoft's AI SDK for orchestrating LLM calls, plugins,
and multi-step workflows. This integration adds ARIA recording via a function
invocation filter (middleware) that captures every kernel function call.

Usage::

    from aria.integrations.semantic_kernel import ARIASemanticKernel

    # Drop-in Kernel wrapper — registers ARIA middleware automatically
    kernel = ARIASemanticKernel(auditor=auditor, model_id="sk-production")

    # Add services and plugins exactly as you would with a plain Kernel
    kernel.add_service(...)
    kernel.add_plugin(my_plugin, plugin_name="MyPlugin")

    result = await kernel.invoke("MyPlugin", "my_function", input="Hello!")
    # ↑ Automatically recorded in ARIA

    # Or register the middleware manually on an existing Kernel:
    from semantic_kernel import Kernel
    from aria.integrations.semantic_kernel import ARIAKernelMiddleware

    kernel = Kernel()
    middleware = ARIAKernelMiddleware(auditor=auditor, model_id="sk-kernel")
    kernel.add_filter("function_invocation", middleware)
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
# Helpers
# ---------------------------------------------------------------------------

def _kernel_args_to_input(arguments: Any) -> dict[str, Any]:
    """Convert KernelArguments (or any mapping) to a serialisable input dict."""
    try:
        if arguments is None:
            return {}
        if hasattr(arguments, "items"):
            return {k: str(v)[:500] for k, v in arguments.items()}
        return {"input": str(arguments)[:500]}
    except Exception:
        return {}


def _kernel_result_to_output(result: Any) -> dict[str, Any]:
    """Convert a KernelFunctionResult (or any value) to a serialisable output dict."""
    try:
        if result is None:
            return {"result": ""}
        if hasattr(result, "value"):
            return {"result": str(result.value)[:500]}
        return {"result": str(result)[:500]}
    except Exception:
        return {"result": str(result)[:500]}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIASemanticKernelRecorder:
    """Shared recording logic for Semantic Kernel wrappers."""

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
            _log.warning("ARIASemanticKernel: record error: %s", exc)


# ---------------------------------------------------------------------------
# Middleware (function invocation filter)
# ---------------------------------------------------------------------------

class ARIAKernelMiddleware:
    """Semantic Kernel function invocation filter that records every call to ARIA.

    Register with ``kernel.add_filter("function_invocation", middleware)`` on an
    existing Kernel, or use ``ARIASemanticKernel`` which does this automatically.

    The filter intercepts every kernel function call (plugins, prompts, native
    functions) and records the function name, arguments, result, and latency.

    Args:
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Override for the model_id label in ARIA records.
                  If None, the qualified function name (``plugin:function``) is used.

    Raises:
        ImportError: if the ``semantic-kernel`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import semantic_kernel  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "semantic-kernel package not installed. "
                "pip install aria-bsv[semantic_kernel]"
            )
        self._recorder = _ARIASemanticKernelRecorder(
            auditor=auditor, aria=aria, model_id=model_id
        )
        self._model_id = model_id

    async def on_function_invocation(self, context: Any, next: Any) -> None:
        """Async filter hook called before and after every kernel function.

        Args:
            context: The ``FunctionInvocationContext`` provided by Semantic Kernel.
            next:    The callable to invoke to continue the filter pipeline.
        """
        function = getattr(context, "function", None)
        if function is not None:
            plugin_name = getattr(function, "plugin_name", "") or ""
            fn_name = getattr(function, "name", str(function)) or ""
            fn_full = f"{plugin_name}:{fn_name}" if plugin_name else fn_name
        else:
            fn_full = "unknown"

        model_id = self._model_id or fn_full
        arguments = getattr(context, "arguments", None)

        t0 = time.time()
        await next(context)
        latency_ms = (time.time() - t0) * 1000

        result = getattr(context, "result", None)
        self._recorder.record(
            model_id=model_id,
            input_data=_kernel_args_to_input(arguments),
            output_data=_kernel_result_to_output(result),
            latency_ms=latency_ms,
            metadata={
                "provider": "semantic_kernel",
                "function": fn_full,
            },
        )


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

class ARIASemanticKernel:
    """Wraps ``semantic_kernel.Kernel`` and adds ARIA recording via middleware.

    Every kernel function invocation is automatically recorded, including
    plugin calls, prompt functions, and native functions.

    Args:
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Override for the model_id label in ARIA records.
        **kwargs: Extra keyword arguments forwarded to ``semantic_kernel.Kernel()``.

    Raises:
        ImportError: if the ``semantic-kernel`` package is not installed.
    """

    def __init__(
        self,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import semantic_kernel as sk  # type: ignore[import]
            self._kernel = sk.Kernel(**kwargs)
        except ImportError:
            raise ImportError(
                "semantic-kernel package not installed. "
                "pip install aria-bsv[semantic_kernel]"
            )

        self._middleware = ARIAKernelMiddleware(
            auditor=auditor, aria=aria, model_id=model_id
        )
        # Register the ARIA middleware as a function invocation filter
        try:
            self._kernel.add_filter("function_invocation", self._middleware)
        except Exception as exc:
            _log.warning(
                "ARIASemanticKernel: could not register function invocation filter: %s", exc
            )

    async def invoke(self, plugin_name: str, function_name: str, **kwargs: Any) -> Any:
        """Async invoke a kernel function and return its result.

        Args:
            plugin_name:    Name of the plugin containing the function.
            function_name:  Name of the function to invoke.
            **kwargs:       Extra kwargs forwarded to ``kernel.invoke()``.

        Returns:
            The ``KernelFunctionResult`` from the kernel.
        """
        return await self._kernel.invoke(plugin_name, function_name, **kwargs)

    def invoke_sync(self, plugin_name: str, function_name: str, **kwargs: Any) -> Any:
        """Synchronously invoke a kernel function.

        Runs the async ``invoke()`` in a new event loop via ``asyncio.run()``.

        Args:
            plugin_name:    Name of the plugin.
            function_name:  Name of the function.
            **kwargs:       Extra kwargs forwarded to ``kernel.invoke()``.

        Returns:
            The ``KernelFunctionResult`` from the kernel.
        """
        return asyncio.run(self.invoke(plugin_name, function_name, **kwargs))

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying Kernel instance."""
        return getattr(self._kernel, name)
