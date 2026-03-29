"""Tests for aria.redis_bus — Redis-backed event bus (mocked Redis)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# Patch redis import before importing RedisEventBus
mock_redis = MagicMock()
mock_redis_module = MagicMock()


class TestRedisEventBusWithMock:
    """Tests for RedisEventBus using mocked Redis client."""

    def _make_bus(self):
        """Create a RedisEventBus with mocked Redis."""
        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis_module.Redis.from_url.return_value = mock_client

            from aria.redis_bus import RedisEventBus
            bus = RedisEventBus.__new__(RedisEventBus)
            bus._redis_url = "redis://localhost:6379/0"
            bus._prefix = "aria:events:"
            bus._redis = mock_client
            bus._callbacks = {}
            bus._lock = __import__("threading").Lock()
            bus._listener_thread = None
            bus._running = False
            bus._counter = 0
            return bus, mock_client

    def test_publish(self):
        from aria.events import ARIAEvent, EventType

        bus, mock_client = self._make_bus()
        event = ARIAEvent(
            type=EventType.RECORD_CREATED,
            data={"record_id": "r1"},
            system_id="test",
        )
        bus.publish(event)
        mock_client.publish.assert_called_once()
        channel, payload = mock_client.publish.call_args[0]
        assert channel == "aria:events:record.created"
        data = json.loads(payload)
        assert data["type"] == "record.created"

    def test_subscribe_decorator(self):
        from aria.events import EventType

        bus, _ = self._make_bus()

        @bus.subscribe(EventType.EPOCH_OPENED)
        def handler(event):
            pass

        assert EventType.EPOCH_OPENED in bus._callbacks
        assert handler in bus._callbacks[EventType.EPOCH_OPENED]

    def test_add_callback(self):
        from aria.events import EventType

        bus, _ = self._make_bus()

        def my_handler(event):
            pass

        bus.add_callback(EventType.ALERT_FIRED, my_handler)
        assert my_handler in bus._callbacks[EventType.ALERT_FIRED]

    def test_subscribe_all(self):
        bus, _ = self._make_bus()

        @bus.subscribe(None)
        def catch_all(event):
            pass

        assert None in bus._callbacks
        assert catch_all in bus._callbacks[None]

    def test_ping(self):
        bus, mock_client = self._make_bus()
        assert bus.ping() is True
        mock_client.ping.return_value = True
        assert bus.ping() is True

    def test_ping_failure(self):
        bus, mock_client = self._make_bus()
        mock_client.ping.side_effect = ConnectionError("down")
        assert bus.ping() is False

    def test_dispatch(self):
        from aria.events import ARIAEvent, EventType

        bus, _ = self._make_bus()
        received = []

        bus.add_callback(EventType.RECORD_CREATED, lambda e: received.append(e))

        event = ARIAEvent(
            type=EventType.RECORD_CREATED,
            data={"test": True},
            system_id="test",
        )
        bus._dispatch(event)
        assert len(received) == 1
        assert received[0].data["test"] is True

    def test_dispatch_catch_all(self):
        from aria.events import ARIAEvent, EventType

        bus, _ = self._make_bus()
        received = []

        bus.add_callback(None, lambda e: received.append(e))

        event = ARIAEvent(
            type=EventType.DRIFT_DETECTED,
            data={},
            system_id="test",
        )
        bus._dispatch(event)
        assert len(received) == 1

    def test_dispatch_callback_error_doesnt_propagate(self):
        from aria.events import ARIAEvent, EventType

        bus, _ = self._make_bus()

        def bad_handler(event):
            raise RuntimeError("oops")

        bus.add_callback(EventType.RECORD_CREATED, bad_handler)

        event = ARIAEvent(
            type=EventType.RECORD_CREATED,
            data={},
            system_id="test",
        )
        # Should not raise
        bus._dispatch(event)

    def test_close(self):
        bus, mock_client = self._make_bus()
        bus.close()
        mock_client.close.assert_called_once()

    def test_publish_assigns_event_id(self):
        from aria.events import ARIAEvent, EventType

        bus, mock_client = self._make_bus()
        event = ARIAEvent(
            type=EventType.EPOCH_CLOSED,
            data={},
            system_id="test",
            event_id=0,
        )
        bus.publish(event)
        assert event.event_id == 1

    def test_counter_increments(self):
        from aria.events import ARIAEvent, EventType

        bus, mock_client = self._make_bus()

        for i in range(3):
            event = ARIAEvent(
                type=EventType.RECORD_CREATED,
                data={},
                system_id="test",
                event_id=0,
            )
            bus.publish(event)

        assert bus._counter == 3
