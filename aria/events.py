"""
aria.events — Event bus for real-time ARIA audit stream.

Every inference, epoch open, epoch close, and watchdog alert fires an event.
Consumers subscribe to receive these events and can react in real-time —
trigger dashboards, push to Kafka, write to a webhook, etc.

Two implementations are provided:

  InMemoryEventBus  — synchronous, thread-safe, zero dependencies.
                      Default for development and testing.

  RedisEventBus     — publishes to a Redis pub-sub channel.
                      Requires: pip install aria-bsv[redis]

Usage::

    from aria.events import InMemoryEventBus, EventType

    bus = InMemoryEventBus()

    @bus.subscribe(EventType.RECORD_CREATED)
    def on_record(event):
        print(f"New inference: {event.data['model_id']} in {event.data['epoch_id']}")

    # Wire into the auditor:
    auditor.add_record_hook(bus.emit_record)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    RECORD_CREATED   = "record.created"
    EPOCH_OPENED     = "epoch.opened"
    EPOCH_CLOSED     = "epoch.closed"
    ALERT_FIRED      = "alert.fired"
    DRIFT_DETECTED   = "drift.detected"
    COMPLIANCE_FAIL  = "compliance.fail"


@dataclass
class ARIAEvent:
    """A single event emitted by the ARIA system.

    Attributes:
        type:       Category of the event.
        data:       Event payload — JSON-serialisable dict.
        system_id:  Which ARIA system produced it.
        timestamp:  Unix seconds (float) when the event occurred.
        event_id:   Monotonically increasing counter for ordering.
    """
    type: EventType
    data: dict[str, Any]
    system_id: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "system_id": self.system_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class EventBusInterface:
    """Base class for ARIA event buses.

    Sub-class and implement ``publish()`` to create a custom backend.
    """

    def publish(self, event: ARIAEvent) -> None:
        """Publish one event.  Must be thread-safe and non-blocking."""
        raise NotImplementedError

    def subscribe(
        self,
        event_type: EventType | None = None,
    ) -> Callable:
        """Decorator — register a callback for *event_type* (None = all types).

        Example::

            @bus.subscribe(EventType.RECORD_CREATED)
            def handler(event: ARIAEvent) -> None:
                ...
        """
        def decorator(func: Callable) -> Callable:
            self.add_listener(func, event_type)
            return func
        return decorator

    def add_listener(
        self,
        callback: Callable[[ARIAEvent], None],
        event_type: EventType | None = None,
    ) -> None:
        """Register *callback* for *event_type*.  None = subscribe to all."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience emit helpers — called by InferenceAuditor / WatchdogDaemon
    # ------------------------------------------------------------------

    def emit_record(self, record: Any) -> None:
        """Emit a RECORD_CREATED event from an AuditRecord object."""
        self.publish(ARIAEvent(
            type=EventType.RECORD_CREATED,
            data={
                "record_id": record.record_id,
                "epoch_id": record.epoch_id,
                "model_id": record.model_id,
                "latency_ms": record.latency_ms,
                "confidence": record.confidence,
                "sequence": record.sequence,
            },
        ))

    def emit_epoch_opened(self, epoch_id: str, system_id: str, txid: str) -> None:
        self.publish(ARIAEvent(
            type=EventType.EPOCH_OPENED,
            system_id=system_id,
            data={"epoch_id": epoch_id, "txid": txid},
        ))

    def emit_epoch_closed(
        self, epoch_id: str, system_id: str, txid: str,
        records_count: int, merkle_root: str,
    ) -> None:
        self.publish(ARIAEvent(
            type=EventType.EPOCH_CLOSED,
            system_id=system_id,
            data={
                "epoch_id": epoch_id,
                "txid": txid,
                "records_count": records_count,
                "merkle_root": merkle_root,
            },
        ))

    def emit_alert(self, alert: Any) -> None:
        """Emit an ALERT_FIRED event from a watchdog Alert object."""
        self.publish(ARIAEvent(
            type=EventType.ALERT_FIRED,
            data={
                "kind": alert.kind,
                "severity": str(alert.severity),
                "message": alert.message,
                "epoch_id": alert.epoch_id,
            },
        ))


# ---------------------------------------------------------------------------
# In-memory implementation (default)
# ---------------------------------------------------------------------------

class InMemoryEventBus(EventBusInterface):
    """Thread-safe in-memory event bus.

    Calls all registered listeners synchronously in the publishing thread.
    Listener exceptions are caught and logged — they never propagate to the
    caller.

    Args:
        max_history: Number of recent events to keep in memory (default 1000).
                     Access via ``bus.history``.
    """

    def __init__(self, max_history: int = 1_000) -> None:
        self._lock = threading.Lock()
        self._listeners: dict[EventType | None, list[Callable]] = {}
        self._history: list[ARIAEvent] = []
        self._max_history = max_history
        self._counter = 0

    def add_listener(
        self,
        callback: Callable[[ARIAEvent], None],
        event_type: EventType | None = None,
    ) -> None:
        with self._lock:
            self._listeners.setdefault(event_type, []).append(callback)

    def publish(self, event: ARIAEvent) -> None:
        with self._lock:
            self._counter += 1
            event.event_id = self._counter
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            # Collect callbacks while holding lock, then call outside
            callbacks = list(self._listeners.get(event.type, []))
            callbacks += list(self._listeners.get(None, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                _log.warning("EventBus listener error: %s", exc)

    @property
    def history(self) -> list[ARIAEvent]:
        """Snapshot of recent events (newest last)."""
        with self._lock:
            return list(self._history)

    def events_of_type(self, event_type: EventType) -> list[ARIAEvent]:
        return [e for e in self.history if e.type == event_type]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Redis implementation (optional dep)
# ---------------------------------------------------------------------------

class RedisEventBus(EventBusInterface):
    """Publishes ARIA events to a Redis pub-sub channel.

    Requires: pip install aria-bsv[redis]

    Each event is JSON-serialised and published to ``channel``.
    Subscribers use standard Redis SUBSCRIBE to consume.

    Args:
        url:     Redis connection URL, e.g. ``redis://localhost:6379/0``.
        channel: Redis channel name (default ``"aria.events"``).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        channel: str = "aria.events",
    ) -> None:
        self._channel = channel
        self._listeners: dict[EventType | None, list[Callable]] = {}
        self._lock = threading.Lock()
        try:
            import redis  # type: ignore[import]
            self._redis = redis.from_url(url)
            self._available = True
        except ImportError:
            _log.warning(
                "RedisEventBus: redis package not installed. "
                "pip install aria-bsv[redis]  — falling back to no-op."
            )
            self._available = False

    def add_listener(
        self,
        callback: Callable[[ARIAEvent], None],
        event_type: EventType | None = None,
    ) -> None:
        """Register a local listener (in-process only — not over Redis)."""
        with self._lock:
            self._listeners.setdefault(event_type, []).append(callback)

    def publish(self, event: ARIAEvent) -> None:
        import json
        payload = json.dumps(event.to_dict())

        # Local listeners (same process)
        with self._lock:
            callbacks = list(self._listeners.get(event.type, []))
            callbacks += list(self._listeners.get(None, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                _log.warning("RedisEventBus local listener error: %s", exc)

        # Remote publish
        if self._available:
            try:
                self._redis.publish(self._channel, payload)
            except Exception as exc:
                _log.warning("RedisEventBus publish failed: %s", exc)
