"""
aria.integrations.dspy — ARIA audit integration for DSPy.

DSPy is Stanford's framework for programmatic LLM prompting. This integration
wraps DSPy modules and optimizers so that every ``forward()`` call and every
optimization run is recorded in ARIA.

Usage::

    import dspy
    from aria.integrations.dspy import ARIADSPyModule

    class QAProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predict(question=question)

    program = QAProgram()
    audited = ARIADSPyModule(module=program, auditor=auditor, model_id="dspy-qa")
    result = audited(question="What is BSV?")
    # ↑ Automatically recorded in ARIA

    # For optimizers:
    from aria.integrations.dspy import ARIADSPyOptimizer
    from dspy.teleprompt import BootstrapFewShot

    audited_optimizer = ARIADSPyOptimizer(
        optimizer=BootstrapFewShot(metric=my_metric),
        auditor=auditor,
        model_id="dspy-optimizer",
    )
    compiled = audited_optimizer.compile(program, trainset=trainset)
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

def _prediction_to_output(prediction: Any) -> dict[str, Any]:
    """Convert a DSPy Prediction to a serialisable output dict."""
    try:
        if hasattr(prediction, "toDict"):
            return {k: str(v)[:500] for k, v in prediction.toDict().items()}
        elif hasattr(prediction, "__dict__"):
            return {
                k: str(v)[:500]
                for k, v in vars(prediction).items()
                if not k.startswith("_")
            }
        return {"result": str(prediction)[:500]}
    except Exception:
        return {"result": str(prediction)[:500]}


def _kwargs_to_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert forward() kwargs to a serialisable input dict (truncated values)."""
    return {k: str(v)[:500] for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# Shared recorder
# ---------------------------------------------------------------------------

class _ARIADSPyRecorder:
    """Shared recording logic for DSPy wrappers."""

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
            _log.warning("ARIADSPy: record error: %s", exc)


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------

class ARIADSPyModule:
    """Wraps any ``dspy.Module`` to record every ``forward()`` call in ARIA.

    Works with both compiled (optimized) and uncompiled modules. The wrapper
    is transparent — all attribute accesses not handled by the wrapper are
    proxied to the underlying DSPy module.

    Args:
        module:   Any ``dspy.Module`` instance (compiled or uncompiled).
        auditor:  ``InferenceAuditor`` instance.
        aria:     ``ARIAQuick`` instance (alternative to auditor).
        model_id: Model ID for ARIA records. If None, uses the module class name.

    Raises:
        ImportError: if the ``dspy-ai`` package is not installed.
    """

    def __init__(
        self,
        module: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import dspy  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "dspy-ai package not installed. pip install aria-bsv[dspy]"
            )
        self._module = module
        self._recorder = _ARIADSPyRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._model_id = model_id or type(module).__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke forward() and record to ARIA."""
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped module's forward() and record to ARIA."""
        # Merge positional args into kwargs using placeholder keys
        all_kwargs: dict[str, Any] = {f"arg_{i}": str(a)[:500] for i, a in enumerate(args)}
        all_kwargs.update(kwargs)

        t0 = time.time()
        result = self._module(*args, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=self._model_id,
            input_data=_kwargs_to_input(all_kwargs),
            output_data=_prediction_to_output(result),
            latency_ms=latency_ms,
            metadata={
                "provider": "dspy",
                "module": type(self._module).__name__,
                "compiled": bool(getattr(self._module, "_compiled", False)),
            },
        )
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying module."""
        return getattr(self._module, name)


# ---------------------------------------------------------------------------
# Optimizer wrapper
# ---------------------------------------------------------------------------

class ARIADSPyOptimizer:
    """Records DSPy optimization (compile) runs to ARIA.

    Wraps any DSPy optimizer (e.g. ``BootstrapFewShot``, ``MIPRO``) and
    records each ``compile()`` invocation, capturing the module types and
    elapsed time.

    Args:
        optimizer:  Any DSPy optimizer instance.
        auditor:    ``InferenceAuditor`` instance.
        aria:       ``ARIAQuick`` instance (alternative to auditor).
        model_id:   Model ID for ARIA records. If None, uses the optimizer class name.

    Raises:
        ImportError: if the ``dspy-ai`` package is not installed.
    """

    def __init__(
        self,
        optimizer: Any,
        auditor: "InferenceAuditor | None" = None,
        aria: "ARIAQuick | None" = None,
        model_id: str | None = None,
    ) -> None:
        try:
            import dspy  # type: ignore[import]  # noqa: F401
        except ImportError:
            raise ImportError(
                "dspy-ai package not installed. pip install aria-bsv[dspy]"
            )
        self._optimizer = optimizer
        self._recorder = _ARIADSPyRecorder(auditor=auditor, aria=aria, model_id=model_id)
        self._model_id = model_id or type(optimizer).__name__

    def compile(self, student: Any, **kwargs: Any) -> Any:
        """Compile (optimize) a DSPy program and record the run to ARIA.

        Args:
            student:  The DSPy module to optimize.
            **kwargs: Extra kwargs forwarded to the optimizer's ``compile()``.

        Returns:
            The compiled DSPy module.
        """
        t0 = time.time()
        compiled = self._optimizer.compile(student, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        self._recorder.record(
            model_id=self._model_id,
            input_data={
                "student_module": type(student).__name__,
                "optimizer": type(self._optimizer).__name__,
            },
            output_data={
                "compiled_module": type(compiled).__name__,
                "compiled": True,
            },
            latency_ms=latency_ms,
            metadata={
                "provider": "dspy",
                "optimizer": type(self._optimizer).__name__,
            },
        )
        return compiled

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying optimizer."""
        return getattr(self._optimizer, name)
