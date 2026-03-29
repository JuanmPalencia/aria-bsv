"""Tests for aria.notifications — Event-driven notification system."""

from __future__ import annotations

import pytest

from aria.notifications import (
    NotificationManager,
    NotificationRule,
    Notification,
)


class _FakeRecord:
    def __init__(self, confidence=0.9, record_id="rec-1", model_id="gpt-4"):
        self.confidence = confidence
        self.record_id = record_id
        self.model_id = model_id


class TestNotification:
    def test_to_dict(self):
        n = Notification(event="test", message="hello", severity="info")
        d = n.to_dict()
        assert d["event"] == "test"
        assert d["message"] == "hello"
        assert d["severity"] == "info"


class TestNotificationManager:
    def test_no_rules_no_notifications(self):
        nm = NotificationManager()
        fired = nm.process_record(_FakeRecord(confidence=0.5))
        assert fired == []

    def test_low_confidence_fires(self):
        nm = NotificationManager()
        nm.on_low_confidence(threshold=0.7, channel="log")
        fired = nm.process_record(_FakeRecord(confidence=0.5))
        assert len(fired) == 1
        assert fired[0].event == "low_confidence"
        assert "0.500" in fired[0].message

    def test_low_confidence_does_not_fire_above_threshold(self):
        nm = NotificationManager()
        nm.on_low_confidence(threshold=0.7, channel="log")
        fired = nm.process_record(_FakeRecord(confidence=0.9))
        assert fired == []

    def test_process_event(self):
        nm = NotificationManager()
        nm.on_epoch_close(channel="log")
        fired = nm.process_event("epoch_close", {"epoch_id": "ep-1"})
        assert len(fired) == 1
        assert fired[0].event == "epoch_close"

    def test_process_event_no_match(self):
        nm = NotificationManager()
        nm.on_epoch_close(channel="log")
        fired = nm.process_event("drift", {})
        assert fired == []

    def test_custom_channel(self):
        captured = []
        nm = NotificationManager()
        nm.add_channel("test", lambda n: captured.append(n))
        nm.on_event("test_event", channel="test")
        nm.process_event("test_event")
        assert len(captured) == 1

    def test_history(self):
        nm = NotificationManager()
        nm.on_event("x", channel="log")
        nm.process_event("x")
        nm.process_event("x")
        assert len(nm.history) == 2

    def test_clear_history(self):
        nm = NotificationManager()
        nm.on_event("x", channel="log")
        nm.process_event("x")
        nm.clear_history()
        assert len(nm.history) == 0

    def test_disabled_rule(self):
        nm = NotificationManager()
        nm._rules.append(NotificationRule(event="x", channel="log", enabled=False))
        fired = nm.process_event("x")
        assert fired == []

    def test_from_config(self):
        nm = NotificationManager.from_config({
            "rules": [
                {"event": "epoch_close", "channel": "log"},
                {"event": "low_confidence", "threshold": 0.6},
            ]
        })
        assert len(nm._rules) == 2

    def test_on_drift(self):
        nm = NotificationManager()
        nm.on_drift(channel="log")
        fired = nm.process_event("drift", {"metric": "ks"})
        assert len(fired) == 1

    def test_on_compliance_fail(self):
        nm = NotificationManager()
        nm.on_compliance_fail(channel="log")
        fired = nm.process_event("compliance_fail")
        assert len(fired) == 1

    def test_multiple_rules_same_event(self):
        captured_a = []
        captured_b = []
        nm = NotificationManager()
        nm.add_channel("a", lambda n: captured_a.append(n))
        nm.add_channel("b", lambda n: captured_b.append(n))
        nm.on_event("x", channel="a")
        nm.on_event("x", channel="b")
        nm.process_event("x")
        assert len(captured_a) == 1
        assert len(captured_b) == 1
