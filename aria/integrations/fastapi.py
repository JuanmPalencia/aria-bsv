"""FastAPI integration helpers for ARIA audit recording."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from ..core.errors import ARIAError

_log = logging.getLogger(__name__)

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False


def audit_inference(model_id: str, auditor: "InferenceAuditor") -> Callable:  # type: ignore[type-arg]
    """Decorator for FastAPI endpoint functions that records each request as an inference.

    Records the request body (as input) and response body (as output) as hashed
    AuditRecords via the provided ``InferenceAuditor``.

    Both synchronous and asynchronous endpoint functions are supported.

    Args:
        model_id: Model identifier to associate with this endpoint's inferences.
        auditor:  A running ``InferenceAuditor`` instance.

    Example::

        from aria.integrations.fastapi import audit_inference

        @app.post("/predict")
        @audit_inference("classifier", auditor)
        async def predict(body: PredictRequest) -> PredictResponse:
            return model.run(body)
    """
    import asyncio  # local import so the module is importable without asyncio

    def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.monotonic()
                result = await func(*args, **kwargs)
                latency = int((time.monotonic() - t0) * 1000)
                try:
                    auditor.record(model_id, kwargs, result, latency_ms=latency)
                except ARIAError as exc:
                    _log.warning("ARIA audit_inference record failed: %s", exc)
                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.monotonic()
                result = func(*args, **kwargs)
                latency = int((time.monotonic() - t0) * 1000)
                try:
                    auditor.record(model_id, kwargs, result, latency_ms=latency)
                except ARIAError as exc:
                    _log.warning("ARIA audit_inference record failed: %s", exc)
                return result

            return sync_wrapper

    return decorator


class ARIAMiddleware:
    """Starlette / FastAPI middleware that adds ARIA audit headers to responses.

    Adds the following response headers for every request handled by the
    decorated app:

    * ``X-ARIA-Epoch-ID``   — current open epoch ID (empty if not yet open).
    * ``X-ARIA-System-ID``  — system_id from the auditor configuration.

    This middleware does NOT record inferences — use ``@audit_inference`` for
    that.  It only exposes the epoch context for client-side receipt tracking.

    Args:
        app:     The ASGI application to wrap.
        auditor: A running ``InferenceAuditor`` instance.

    Example::

        app = FastAPI()
        app.add_middleware(ARIAMiddleware, auditor=auditor)
    """

    def __init__(self, app: "ASGIApp", auditor: "InferenceAuditor") -> None:
        if not _FASTAPI_AVAILABLE:
            raise ImportError(  # pragma: no cover
                "fastapi is required for ARIAMiddleware — pip install fastapi"
            )
        self._app = app
        self._auditor = auditor

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                open_result = self._auditor._batch._current_open
                epoch_id = open_result.epoch_id if open_result else ""
                headers.append(
                    (b"x-aria-epoch-id", epoch_id.encode())
                )
                headers.append(
                    (b"x-aria-system-id", self._auditor._config.system_id.encode())
                )
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_headers)
