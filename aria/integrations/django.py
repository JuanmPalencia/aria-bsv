"""
aria.integrations.django — ARIA audit integration for Django.

Usage in settings.py:
    MIDDLEWARE = [
        ...
        "aria.integrations.django.ARIAMiddleware",
    ]
    ARIA_AUDITOR = auditor_instance  # set at startup

Usage in views.py:
    from aria.integrations.django import audit_view

    @audit_view("my-model", auditor)
    def predict(request):
        ...
        return JsonResponse(result)
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TYPE_CHECKING

try:
    from django.http import HttpRequest, HttpResponse
    from django.utils.deprecation import MiddlewareMixin
except ImportError:
    raise ImportError(
        "Django integration requires Django: pip install aria-bsv[django]"
    )

if TYPE_CHECKING:
    from aria.auditor import InferenceAuditor


class ARIAMiddleware(MiddlewareMixin):
    """Django middleware that adds X-ARIA-* headers and optionally records inferences.

    Add to MIDDLEWARE in settings.py. Requires ARIA_AUDITOR to be set in
    django.conf.settings or passed at init.

    Adds response headers:
        X-ARIA-System-ID: <system_id from config>
        X-ARIA-Epoch-ID:  <current epoch id>
    """

    def __init__(self, get_response: Callable, auditor: "InferenceAuditor | None" = None) -> None:
        self._auditor = auditor
        super().__init__(get_response)

    def _get_auditor(self) -> "InferenceAuditor | None":
        if self._auditor is not None:
            return self._auditor
        try:
            from django.conf import settings
            return getattr(settings, "ARIA_AUDITOR", None)
        except Exception:
            return None

    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        auditor = self._get_auditor()
        if auditor is None:
            return response
        try:
            response["X-ARIA-System-ID"] = auditor._config.system_id
            # Expose epoch_id if available via a lightweight accessor
            epoch_id = getattr(auditor, "_epoch_id", "")
            if epoch_id:
                response["X-ARIA-Epoch-ID"] = epoch_id
        except Exception:
            pass
        return response


def audit_view(
    model_id: str,
    auditor: "InferenceAuditor",
    extract_confidence: Callable | None = None,
    pii_fields: list[str] | None = None,
) -> Callable:
    """Decorator for Django views that records the request/response pair to ARIA.

    The request body (parsed as JSON if possible, else raw) is used as input_data.
    The response body (parsed as JSON if possible, else raw) is used as output_data.

    Args:
        model_id:           Model ID for AuditRecord. Must be in auditor.model_hashes.
        auditor:            InferenceAuditor instance.
        extract_confidence: Optional callable(response) -> float | None.
        pii_fields:         Additional field names to strip from request data.

    Example:
        @audit_view("triage-v3", auditor)
        def triage(request):
            data = json.loads(request.body)
            result = model.predict(data)
            return JsonResponse(result)
    """
    import json

    _pii = set(pii_fields or [])

    def decorator(view_func: Callable) -> Callable:
        @functools.wraps(view_func)
        def wrapper(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
            start = time.monotonic()

            # Parse input
            try:
                input_data = json.loads(request.body)
            except (ValueError, AttributeError):
                input_data = {"raw": str(request.body)}

            # Strip PII from input
            if isinstance(input_data, dict):
                input_data = {k: v for k, v in input_data.items() if k not in _pii}

            response = view_func(request, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)

            # Parse output
            try:
                output_data = json.loads(response.content)
            except (ValueError, AttributeError):
                output_data = {"status": response.status_code}

            confidence = extract_confidence(response) if extract_confidence else None

            try:
                auditor.record(
                    model_id,
                    input_data,
                    output_data,
                    confidence=confidence,
                    latency_ms=latency_ms,
                )
            except Exception:
                pass  # Never let auditing break the view

            return response

        return wrapper
    return decorator
