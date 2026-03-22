"""
aria.integrations.flask — ARIA audit integration for Flask.

Usage:
    from aria.integrations.flask import ARIAFlask, audit_route

    aria_ext = ARIAFlask(auditor=auditor)
    aria_ext.init_app(app)

    @app.route("/predict", methods=["POST"])
    @audit_route("my-model", auditor)
    def predict():
        data = request.get_json()
        result = model.predict(data)
        return jsonify(result)
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TYPE_CHECKING

try:
    from flask import Flask, Request, Response, g, request as flask_request
except ImportError:
    raise ImportError(
        "Flask integration requires Flask: pip install aria-bsv[flask]"
    )

if TYPE_CHECKING:
    from aria.auditor import InferenceAuditor


class ARIAFlask:
    """Flask extension that adds X-ARIA-* response headers.

    Usage:
        aria = ARIAFlask(auditor=auditor)
        aria.init_app(app)

    Or with application factory:
        aria = ARIAFlask()
        # later:
        aria.init_app(app, auditor=auditor)
    """

    def __init__(self, auditor: "InferenceAuditor | None" = None) -> None:
        self._auditor = auditor

    def init_app(self, app: Flask, auditor: "InferenceAuditor | None" = None) -> None:
        if auditor is not None:
            self._auditor = auditor

        _auditor = self._auditor

        @app.after_request
        def _add_aria_headers(response: Response) -> Response:
            if _auditor is None:
                return response
            try:
                response.headers["X-ARIA-System-ID"] = _auditor._config.system_id
                epoch_id = getattr(_auditor, "_epoch_id", "")
                if epoch_id:
                    response.headers["X-ARIA-Epoch-ID"] = epoch_id
            except Exception:
                pass
            return response

        app.extensions["aria"] = self


def audit_route(
    model_id: str,
    auditor: "InferenceAuditor",
    extract_confidence: Callable | None = None,
    pii_fields: list[str] | None = None,
) -> Callable:
    """Decorator for Flask route functions that records request/response to ARIA.

    The JSON request body is used as input_data. The JSON response body is
    used as output_data.

    Args:
        model_id:           Model ID for AuditRecord. Must be in auditor.model_hashes.
        auditor:            InferenceAuditor instance.
        extract_confidence: Optional callable(response) -> float | None.
        pii_fields:         Field names to strip from request JSON before hashing.

    Example:
        @app.route("/triage", methods=["POST"])
        @audit_route("triage-v3", auditor)
        def triage():
            data = request.get_json()
            result = model.predict(data)
            return jsonify(result)
    """
    import json

    _pii = set(pii_fields or [])

    def decorator(view_func: Callable) -> Callable:
        @functools.wraps(view_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()

            # Capture input from current request context
            input_data: Any
            try:
                input_data = flask_request.get_json(silent=True) or {}
            except Exception:
                input_data = {}

            if isinstance(input_data, dict):
                input_data = {k: v for k, v in input_data.items() if k not in _pii}

            response = view_func(*args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)

            # Normalise response to a Flask Response object
            from flask import current_app
            resp_obj = current_app.make_response(response)

            # Parse output
            try:
                output_data = json.loads(resp_obj.get_data(as_text=True))
            except (ValueError, TypeError):
                output_data = {"status": resp_obj.status_code}

            confidence = extract_confidence(resp_obj) if extract_confidence else None

            try:
                auditor.record(
                    model_id,
                    input_data,
                    output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )
            except Exception:
                pass  # Never let auditing break the route

            return response

        return wrapper
    return decorator
