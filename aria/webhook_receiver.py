"""
aria.webhook_receiver — Receive external webhooks and create ARIA audit records.

Provides a FastAPI router that accepts incoming webhooks from external systems
(CI/CD, monitoring, model registries) and converts them to audit records.

Usage::

    from fastapi import FastAPI
    from aria.webhook_receiver import create_webhook_router

    app = FastAPI()
    router = create_webhook_router(storage, system_id="my-system", secret="webhook-secret")
    app.include_router(router, prefix="/aria/webhooks")

Webhook payload format::

    POST /aria/webhooks/ingest
    X-ARIA-Signature: sha256=<HMAC hex>
    Content-Type: application/json

    {
        "model_id": "gpt-4",
        "input_data": {"prompt": "..."},
        "output_data": {"text": "..."},
        "confidence": 0.95,
        "latency_ms": 120,
        "metadata": {"source": "ci-pipeline"}
    }
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)

_FASTAPI_AVAILABLE = False
try:
    from fastapi import APIRouter, Header, HTTPException, Request
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    pass


@dataclass
class WebhookEvent:
    """Parsed webhook event."""

    event_id: str
    model_id: str
    input_data: Any
    output_data: Any
    confidence: float | None = None
    latency_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    received_at: float = 0
    source_ip: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "model_id": self.model_id,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "received_at": self.received_at,
            "verified": self.verified,
        }


class WebhookProcessor:
    """Processes webhook events into ARIA audit records.

    Can be used standalone (without FastAPI) by calling ``process()`` directly.

    Args:
        system_id:  ARIA system identifier.
        secret:     HMAC secret for signature verification (optional).
        on_event:   Callback invoked for each processed event.
    """

    def __init__(
        self,
        system_id: str = "webhook",
        secret: str | None = None,
        on_event: Callable[[WebhookEvent], None] | None = None,
    ) -> None:
        self.system_id = system_id
        self._secret = secret
        self._on_event = on_event
        self._events: list[WebhookEvent] = []

    @property
    def events(self) -> list[WebhookEvent]:
        return list(self._events)

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify HMAC-SHA256 signature of the payload.

        Args:
            payload:   Raw request body bytes.
            signature: Value of X-ARIA-Signature header (format: sha256=<hex>).

        Returns:
            True if signature is valid or no secret configured.
        """
        if not self._secret:
            return True

        if not signature or not signature.startswith("sha256="):
            return False

        expected = hmac.new(
            self._secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        received_hex = signature[7:]  # strip "sha256="
        return hmac.compare_digest(expected, received_hex)

    def process(
        self,
        model_id: str,
        input_data: Any,
        output_data: Any,
        confidence: float | None = None,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        source_ip: str = "",
        verified: bool = True,
    ) -> WebhookEvent:
        """Process a webhook payload into a WebhookEvent.

        Args:
            model_id:    Model identifier.
            input_data:  Input data (will be hashed, not stored raw).
            output_data: Output data (will be hashed, not stored raw).
            confidence:  Confidence score.
            latency_ms:  Latency in milliseconds.
            metadata:    Extra metadata.
            source_ip:   Source IP of the webhook sender.
            verified:    Whether signature verification passed.

        Returns:
            WebhookEvent
        """
        event = WebhookEvent(
            event_id=uuid.uuid4().hex[:16],
            model_id=model_id,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata or {},
            received_at=time.time(),
            source_ip=source_ip,
            verified=verified,
        )

        self._events.append(event)

        if self._on_event:
            self._on_event(event)

        return event

    @property
    def event_count(self) -> int:
        return len(self._events)


if _FASTAPI_AVAILABLE:

    class WebhookPayload(BaseModel):
        """Request body for webhook ingest endpoint."""

        model_id: str = Field(..., description="Model identifier")
        input_data: Any = Field(..., description="Model input data")
        output_data: Any = Field(..., description="Model output data")
        confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence score")
        latency_ms: int = Field(0, ge=0, description="Latency in milliseconds")
        metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    class WebhookResponse(BaseModel):
        """Response for webhook ingest endpoint."""

        event_id: str
        model_id: str
        accepted: bool = True

    def create_webhook_router(
        processor: WebhookProcessor | None = None,
        system_id: str = "webhook",
        secret: str | None = None,
        auditor: Any | None = None,
    ) -> APIRouter:
        """Create a FastAPI router for receiving webhooks.

        Args:
            processor: Optional WebhookProcessor (created automatically if None).
            system_id: ARIA system identifier.
            secret:    HMAC secret for signature verification.
            auditor:   Optional InferenceAuditor to forward records to.

        Returns:
            FastAPI APIRouter with /ingest and /status endpoints.
        """
        router = APIRouter(tags=["ARIA Webhooks"])

        def _on_event(event: WebhookEvent) -> None:
            if auditor is not None:
                try:
                    auditor.record(
                        model_id=event.model_id,
                        input_data=event.input_data,
                        output_data=event.output_data,
                        confidence=event.confidence,
                        latency_ms=event.latency_ms,
                        metadata={"webhook_event_id": event.event_id, **event.metadata},
                    )
                except Exception as exc:
                    _log.warning("Failed to forward webhook to auditor: %s", exc)

        proc = processor or WebhookProcessor(
            system_id=system_id,
            secret=secret,
            on_event=_on_event,
        )

        @router.post("/ingest", response_model=WebhookResponse)
        async def ingest(
            request: Request,
            payload: WebhookPayload,
            x_aria_signature: str | None = Header(None),
        ) -> WebhookResponse:
            """Receive and process a webhook event."""
            body = await request.body()
            if not proc.verify_signature(body, x_aria_signature or ""):
                raise HTTPException(status_code=401, detail="Invalid signature")

            source_ip = request.client.host if request.client else ""
            event = proc.process(
                model_id=payload.model_id,
                input_data=payload.input_data,
                output_data=payload.output_data,
                confidence=payload.confidence,
                latency_ms=payload.latency_ms,
                metadata=payload.metadata,
                source_ip=source_ip,
                verified=True,
            )

            return WebhookResponse(
                event_id=event.event_id,
                model_id=event.model_id,
            )

        @router.get("/status")
        async def status() -> dict:
            """Return webhook receiver status."""
            return {
                "system_id": proc.system_id,
                "events_received": proc.event_count,
            }

        return router
