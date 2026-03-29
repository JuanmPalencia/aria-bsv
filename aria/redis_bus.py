"""
aria.redis_bus — Redis-backed event bus for distributed ARIA deployments.

Publishes ARIA events to Redis Pub/Sub channels, enabling cross-process
and cross-machine event distribution.

Requires: ``pip install aria-bsv[redis]``

Usage::

    from aria.redis_bus import RedisEventBus
    from aria.events import EventType

    bus = RedisEventBus("redis://localhost:6379/0")

    @bus.subscribe(EventType.RECORD_CREATED)
    def on_record(event):
        print(f"Record: {event.data}")

    bus.start_listener()  # background thread for receiving
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from .events import ARIAEvent, EventBusInterface, EventType

_log = logging.getLogger(__name__)

_CHANNEL_PREFIX = "aria:events:"
_ALL_CHANNEL = "aria:events:*"


class RedisEventBus(EventBusInterface):
    """Redis Pub/Sub backed event bus.

    Publishes events to Redis channels named ``aria:events:{event_type}``.
    Subscribers receive events via a background listener thread.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        channel_prefix: Prefix for Redis channel names.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        channel_prefix: str = _CHANNEL_PREFIX,
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "RedisEventBus requires the 'redis' package. "
                "Install it with: pip install aria-bsv[redis]"
            )

        self._redis_url = redis_url
        self._prefix = channel_prefix
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._callbacks: dict[EventType | None, list[Callable]] = {}
        self._lock = threading.Lock()
        self._listener_thread: threading.Thread | None = None
        self._running = False
        self._counter = 0

    def publish(self, event: ARIAEvent) -> None:
        """Publish an event to the Redis channel."""
        if event.event_id == 0:
            with self._lock:
                self._counter += 1
                event.event_id = self._counter

        channel = f"{self._prefix}{event.type.value}"
        payload = json.dumps(event.to_dict())
        self._redis.publish(channel, payload)
        _log.debug("Published event %s to %s", event.event_id, channel)

    def subscribe(self, event_type: EventType | None = None) -> Callable:
        """Decorator to register a callback for an event type."""
        def decorator(fn: Callable) -> Callable:
            with self._lock:
                self._callbacks.setdefault(event_type, []).append(fn)
            return fn
        return decorator

    def add_callback(self, event_type: EventType | None, fn: Callable) -> None:
        """Register a callback programmatically."""
        with self._lock:
            self._callbacks.setdefault(event_type, []).append(fn)

    def start_listener(self) -> None:
        """Start a background thread that listens for events on Redis."""
        if self._running:
            return
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="aria-redis-listener"
        )
        self._listener_thread.start()
        _log.info("Redis event listener started on %s", self._redis_url)

    def stop_listener(self) -> None:
        """Stop the background listener."""
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5)
        self._listener_thread = None

    def _listen_loop(self) -> None:
        import redis

        pubsub = self._redis.pubsub()
        pattern = f"{self._prefix}*"
        pubsub.psubscribe(pattern)

        try:
            while self._running:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    continue
                if msg["type"] not in ("pmessage", "message"):
                    continue

                try:
                    data = json.loads(msg["data"])
                    event = ARIAEvent(
                        type=EventType(data["type"]),
                        data=data.get("data", {}),
                        system_id=data.get("system_id", ""),
                        timestamp=data.get("timestamp", time.time()),
                        event_id=data.get("event_id", 0),
                    )
                    self._dispatch(event)
                except Exception as exc:
                    _log.warning("Error processing Redis message: %s", exc)
        finally:
            pubsub.punsubscribe(pattern)
            pubsub.close()

    def _dispatch(self, event: ARIAEvent) -> None:
        """Dispatch event to registered callbacks."""
        with self._lock:
            callbacks = list(self._callbacks.get(event.type, []))
            callbacks.extend(self._callbacks.get(None, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                _log.error("Callback error for %s: %s", event.type, exc)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._redis.ping()
        except Exception:
            return False

    def close(self) -> None:
        """Stop listener and close Redis connection."""
        self.stop_listener()
        self._redis.close()
