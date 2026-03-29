"""Tests for aria.webhook_receiver — webhook processing and HMAC verification."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from aria.webhook_receiver import WebhookEvent, WebhookProcessor


class TestWebhookProcessor:
    """Tests for WebhookProcessor."""

    def test_create(self):
        proc = WebhookProcessor(system_id="test")
        assert proc.system_id == "test"
        assert proc.event_count == 0

    def test_process(self):
        proc = WebhookProcessor()
        event = proc.process(
            model_id="gpt-4",
            input_data={"prompt": "hi"},
            output_data={"text": "hello"},
            confidence=0.95,
            latency_ms=50,
        )
        assert isinstance(event, WebhookEvent)
        assert event.model_id == "gpt-4"
        assert event.confidence == 0.95
        assert event.latency_ms == 50
        assert event.received_at > 0
        assert event.verified is True

    def test_process_multiple(self):
        proc = WebhookProcessor()
        proc.process("m1", "i1", "o1")
        proc.process("m2", "i2", "o2")
        assert proc.event_count == 2
        assert len(proc.events) == 2

    def test_process_with_metadata(self):
        proc = WebhookProcessor()
        event = proc.process("m", "i", "o", metadata={"source": "ci"})
        assert event.metadata == {"source": "ci"}

    def test_on_event_callback(self):
        received = []
        proc = WebhookProcessor(on_event=lambda e: received.append(e))
        proc.process("m", "i", "o")
        assert len(received) == 1
        assert received[0].model_id == "m"

    def test_verify_signature_no_secret(self):
        proc = WebhookProcessor()
        assert proc.verify_signature(b"any", "sha256=abc") is True

    def test_verify_signature_valid(self):
        secret = "my-webhook-secret"
        proc = WebhookProcessor(secret=secret)
        payload = b'{"model_id": "gpt-4"}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert proc.verify_signature(payload, sig) is True

    def test_verify_signature_invalid(self):
        proc = WebhookProcessor(secret="secret")
        assert proc.verify_signature(b"payload", "sha256=wrong") is False

    def test_verify_signature_missing_prefix(self):
        proc = WebhookProcessor(secret="secret")
        assert proc.verify_signature(b"payload", "no-prefix") is False

    def test_verify_signature_empty(self):
        proc = WebhookProcessor(secret="secret")
        assert proc.verify_signature(b"payload", "") is False

    def test_source_ip(self):
        proc = WebhookProcessor()
        event = proc.process("m", "i", "o", source_ip="1.2.3.4")
        assert event.source_ip == "1.2.3.4"


class TestWebhookEvent:
    """Tests for WebhookEvent dataclass."""

    def test_to_dict(self):
        e = WebhookEvent(
            event_id="evt-1",
            model_id="gpt-4",
            input_data={"a": 1},
            output_data={"b": 2},
            confidence=0.9,
            latency_ms=100,
            received_at=1234567890.0,
            verified=True,
        )
        d = e.to_dict()
        assert d["event_id"] == "evt-1"
        assert d["model_id"] == "gpt-4"
        assert d["verified"] is True
        assert d["confidence"] == 0.9

    def test_defaults(self):
        e = WebhookEvent(
            event_id="e", model_id="m",
            input_data=None, output_data=None,
        )
        assert e.confidence is None
        assert e.latency_ms == 0
        assert e.metadata == {}
        assert e.verified is False
