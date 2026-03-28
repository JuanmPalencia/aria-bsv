"""Tests for aria.events — InMemoryEventBus and RedisEventBus."""

from __future__ import annotations

import threading
import time

import pytest

from aria.events import (
    ARIAEvent,
    EventBusInterface,
    EventType,
    InMemoryEventBus,
)


# ---------------------------------------------------------------------------
# ARIAEvent
# ---------------------------------------------------------------------------

class TestARIAEvent:
    def test_to_dict_has_required_keys(self):
        evt = ARIAEvent(type=EventType.RECORD_CREATED, data={"x": 1})
        d = evt.to_dict()
        assert d["type"] == "record.created"
        assert d["data"] == {"x": 1}
        assert "timestamp" in d
        assert "event_id" in d

    def test_timestamp_is_recent(self):
        before = time.time()
        evt = ARIAEvent(type=EventType.EPOCH_OPENED, data={})
        after = time.time()
        assert before <= evt.timestamp <= after

    def test_system_id_default_empty(self):
        evt = ARIAEvent(type=EventType.ALERT_FIRED, data={})
        assert evt.system_id == ""


# ---------------------------------------------------------------------------
# InMemoryEventBus — basic pub/sub
# ---------------------------------------------------------------------------

class TestInMemoryEventBus:
    def test_subscribe_and_receive_event(self):
        bus = InMemoryEventBus()
        received = []

        @bus.subscribe(EventType.RECORD_CREATED)
        def handler(event: ARIAEvent):
            received.append(event)

        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={"a": 1}))
        assert len(received) == 1
        assert received[0].data == {"a": 1}

    def test_none_subscription_receives_all(self):
        bus = InMemoryEventBus()
        received = []

        @bus.subscribe(None)
        def catch_all(event: ARIAEvent):
            received.append(event.type)

        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))
        bus.publish(ARIAEvent(type=EventType.EPOCH_OPENED, data={}))
        bus.publish(ARIAEvent(type=EventType.DRIFT_DETECTED, data={}))

        assert EventType.RECORD_CREATED in received
        assert EventType.EPOCH_OPENED in received
        assert EventType.DRIFT_DETECTED in received

    def test_wrong_type_does_not_trigger(self):
        bus = InMemoryEventBus()
        received = []

        @bus.subscribe(EventType.ALERT_FIRED)
        def handler(event: ARIAEvent):
            received.append(event)

        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))
        assert len(received) == 0

    def test_event_id_increments(self):
        bus = InMemoryEventBus()
        ids = []

        @bus.subscribe(None)
        def capture(evt):
            ids.append(evt.event_id)

        for _ in range(5):
            bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

        assert ids == [1, 2, 3, 4, 5]

    def test_history_accumulates(self):
        bus = InMemoryEventBus(max_history=100)
        for i in range(10):
            bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={"i": i}))
        assert len(bus.history) == 10

    def test_history_capped_at_max(self):
        bus = InMemoryEventBus(max_history=5)
        for i in range(20):
            bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))
        assert len(bus.history) <= 5

    def test_events_of_type_filters(self):
        bus = InMemoryEventBus()
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))
        bus.publish(ARIAEvent(type=EventType.EPOCH_OPENED, data={}))
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

        records = bus.events_of_type(EventType.RECORD_CREATED)
        opened = bus.events_of_type(EventType.EPOCH_OPENED)

        assert len(records) == 2
        assert len(opened) == 1

    def test_clear_history(self):
        bus = InMemoryEventBus()
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))
        bus.clear_history()
        assert bus.history == []

    def test_listener_exception_does_not_propagate(self):
        bus = InMemoryEventBus()

        @bus.subscribe(None)
        def bad_handler(_):
            raise RuntimeError("boom")

        # Should not raise
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

    def test_add_listener_api(self):
        bus = InMemoryEventBus()
        received = []
        bus.add_listener(received.append, EventType.ALERT_FIRED)
        bus.publish(ARIAEvent(type=EventType.ALERT_FIRED, data={"x": 9}))
        assert len(received) == 1

    def test_thread_safe_concurrent_publish(self):
        bus = InMemoryEventBus(max_history=10_000)
        threads = []

        def publish_many():
            for _ in range(50):
                bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

        for _ in range(10):
            t = threading.Thread(target=publish_many)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(bus.history) == 500

    # ------------------------------------------------------------------
    # Convenience emit helpers
    # ------------------------------------------------------------------

    def test_emit_record(self):
        bus = InMemoryEventBus()
        received = []

        @bus.subscribe(EventType.RECORD_CREATED)
        def h(e):
            received.append(e)

        class FakeRecord:
            record_id = "r1"
            epoch_id = "e1"
            model_id = "gpt"
            latency_ms = 42.0
            confidence = 0.9
            sequence = 1

        bus.emit_record(FakeRecord())
        assert len(received) == 1
        assert received[0].data["model_id"] == "gpt"
        assert received[0].data["confidence"] == 0.9

    def test_emit_epoch_opened(self):
        bus = InMemoryEventBus()
        received = []
        bus.add_listener(received.append, EventType.EPOCH_OPENED)
        bus.emit_epoch_opened("ep1", "sys1", "txid-abc")
        assert received[0].data["epoch_id"] == "ep1"
        assert received[0].data["txid"] == "txid-abc"

    def test_emit_epoch_closed(self):
        bus = InMemoryEventBus()
        received = []
        bus.add_listener(received.append, EventType.EPOCH_CLOSED)
        bus.emit_epoch_closed("ep1", "sys1", "tx2", 100, "deadbeef")
        d = received[0].data
        assert d["records_count"] == 100
        assert d["merkle_root"] == "deadbeef"

    def test_emit_alert(self):
        bus = InMemoryEventBus()
        received = []
        bus.add_listener(received.append, EventType.ALERT_FIRED)

        class FakeAlert:
            kind = "LATENCY_SPIKE"
            severity = "WARNING"
            message = "Too slow"
            epoch_id = "ep1"

        bus.emit_alert(FakeAlert())
        assert received[0].data["kind"] == "LATENCY_SPIKE"


# ---------------------------------------------------------------------------
# EventBusInterface — abstract methods
# ---------------------------------------------------------------------------

class TestEventBusInterface:
    def test_publish_raises(self):
        bus = EventBusInterface()
        with pytest.raises(NotImplementedError):
            bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

    def test_add_listener_raises(self):
        bus = EventBusInterface()
        with pytest.raises(NotImplementedError):
            bus.add_listener(lambda e: None)


# ---------------------------------------------------------------------------
# RedisEventBus — graceful fallback when redis not installed
# ---------------------------------------------------------------------------

class TestRedisEventBusNoOp:
    def test_import_and_instantiate(self):
        from aria.events import RedisEventBus
        bus = RedisEventBus()
        assert bus is not None
        assert isinstance(bus._available, bool)

    def test_publish_does_not_raise_without_redis(self):
        from aria.events import RedisEventBus
        bus = RedisEventBus()
        # Should not raise regardless of whether redis is installed
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={}))

    def test_local_listeners_work_without_redis(self):
        from aria.events import RedisEventBus
        bus = RedisEventBus()
        received = []
        bus.add_listener(received.append, EventType.RECORD_CREATED)
        bus.publish(ARIAEvent(type=EventType.RECORD_CREATED, data={"y": 2}))
        assert len(received) == 1
        assert received[0].data["y"] == 2
