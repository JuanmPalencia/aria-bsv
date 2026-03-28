"""
aria.shadow_mode — Shadow deployment for safe model evaluation.

Routes each inference to both the live model and a shadow model simultaneously.
The shadow model's response is audited but not returned to the caller, enabling
risk-free performance evaluation before promoting a new model version.

Usage::

    from aria.shadow_mode import ShadowRunner

    runner = ShadowRunner(
        live_fn=live_model.predict,
        shadow_fn=new_model.predict,
        auditor=auditor,
        shadow_model_id="v2-shadow",
    )

    # Returns live result only; shadow is audited asynchronously
    result = runner.run("What is BSV?")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .auditor import InferenceAuditor
    from .quick import ARIAQuick

_log = logging.getLogger(__name__)


@dataclass
class ShadowResult:
    """Holds both live and shadow inference results."""
    live_result:      Any
    shadow_result:    Any | None = None
    shadow_error:     str | None = None
    live_latency_ms:  float = 0.0
    shadow_latency_ms: float = 0.0


class ShadowRunner:
    """Routes inferences to both live and shadow models.

    The live model's result is always returned. The shadow model is called
    in a background thread and its output is recorded via ARIA. If the shadow
    model raises, the error is logged but the live result is unaffected.

    Args:
        live_fn:          Callable for the live (production) model.
        shadow_fn:        Callable for the shadow (candidate) model.
        auditor:          InferenceAuditor for recording shadow inferences.
        aria:             ARIAQuick instance (alternative to auditor).
        live_model_id:    Label for live model records (default "live").
        shadow_model_id:  Label for shadow model records (default "shadow").
        async_shadow:     Run shadow in a background thread (default True).
    """

    def __init__(
        self,
        live_fn:         Callable,
        shadow_fn:       Callable,
        auditor:         "InferenceAuditor | None" = None,
        aria:            "ARIAQuick | None" = None,
        live_model_id:   str = "live",
        shadow_model_id: str = "shadow",
        async_shadow:    bool = True,
    ) -> None:
        self._live    = live_fn
        self._shadow  = shadow_fn
        self._auditor = auditor
        self._aria    = aria
        self._live_id    = live_model_id
        self._shadow_id  = shadow_model_id
        self._async      = async_shadow
        self._shadow_results: list[ShadowResult] = []

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run both models; return only the live result.

        Args:
            *args, **kwargs: Forwarded to both live_fn and shadow_fn.

        Returns:
            Result from live_fn.
        """
        t0 = time.time()
        live_result = self._live(*args, **kwargs)
        live_latency = (time.time() - t0) * 1000

        sr = ShadowResult(live_result=live_result, live_latency_ms=live_latency)
        self._shadow_results.append(sr)

        if self._async:
            t = threading.Thread(
                target=self._run_shadow,
                args=(sr, args, kwargs),
                daemon=True,
            )
            t.start()
        else:
            self._run_shadow(sr, args, kwargs)

        return live_result

    def _run_shadow(self, sr: ShadowResult, args: tuple, kwargs: dict) -> None:
        t0 = time.time()
        try:
            result = self._shadow(*args, **kwargs)
            sr.shadow_result = result
            sr.shadow_latency_ms = (time.time() - t0) * 1000
            self._record(result, sr.shadow_latency_ms, args, kwargs)
        except Exception as exc:
            sr.shadow_error = str(exc)
            _log.warning("ShadowRunner: shadow model error: %s", exc)

    def _record(self, result: Any, latency_ms: float, args: tuple, kwargs: dict) -> None:
        try:
            input_data  = {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
            output_data = {"result": str(result)[:1000]}
            if self._auditor is not None:
                self._auditor.record(
                    self._shadow_id, input_data, output_data,
                    latency_ms=int(latency_ms),
                    metadata={"mode": "shadow"},
                )
            elif self._aria is not None:
                self._aria.record(
                    model_id=self._shadow_id,
                    input_data=input_data,
                    output_data=output_data,
                    latency_ms=latency_ms,
                    metadata={"mode": "shadow"},
                )
        except Exception as exc:
            _log.warning("ShadowRunner: record error: %s", exc)

    @property
    def shadow_results(self) -> list[ShadowResult]:
        """All shadow results collected so far."""
        return list(self._shadow_results)

    def error_rate(self) -> float:
        """Fraction of shadow calls that raised an exception."""
        if not self._shadow_results:
            return 0.0
        errors = sum(1 for r in self._shadow_results if r.shadow_error)
        return errors / len(self._shadow_results)
