"""
portal.backend.websocket — Real-time inference streaming via WebSocket.

Provides a WebSocket endpoint that streams live inference audit records
as they are committed by an :class:`aria.auditor.InferenceAuditor`. Clients
subscribe to a *system_id* channel and receive JSON-encoded events.

Event types
-----------
``inference.record``
    Emitted after each successful ``auditor.record()`` call.
``epoch.open``
    Emitted when a new epoch is opened.
``epoch.close``
    Emitted when an epoch is closed and anchored on BSV.
``system.heartbeat``
    Periodic keep-alive, sent every ``heartbeat_interval`` seconds.
``system.error``
    Emitted when the channel encounters an error.

Usage (server side)::

    from portal.backend.websocket import ARIAWebSocketServer, ARIAWebSocketEvent

    server = ARIAWebSocketServer()
    server.emit(ARIAWebSocketEvent(event="inference.record", system_id="acme",
                                  payload={"model": "gpt-4", "confidence": 0.99}))

Usage (FastAPI integration)::

    from fastapi import WebSocket
    from portal.backend.websocket import ws_router

    app.include_router(ws_router)
    # ws://host/ws/{system_id}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HEARTBEAT_INTERVAL = 30  # seconds
DEFAULT_MAX_QUEUE_SIZE = 256
DEFAULT_MAX_CONNECTIONS_PER_CHANNEL = 50


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class WSEventType(str, Enum):
    INFERENCE_RECORD = "inference.record"
    EPOCH_OPEN       = "epoch.open"
    EPOCH_CLOSE      = "epoch.close"
    HEARTBEAT        = "system.heartbeat"
    ERROR            = "system.error"
    SUBSCRIBE_ACK    = "system.subscribe_ack"
    DISCONNECT       = "system.disconnect"


@dataclass
class ARIAWebSocketEvent:
    """A single event pushed to WebSocket subscribers."""
    event:     str
    system_id: str
    payload:   dict = field(default_factory=dict)
    timestamp: str = ""
    event_id:  str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        return json.dumps({
            "event":     self.event,
            "system_id": self.system_id,
            "payload":   self.payload,
            "timestamp": self.timestamp,
            "event_id":  self.event_id,
        })

    @classmethod
    def from_json(cls, raw: str) -> "ARIAWebSocketEvent":
        d = json.loads(raw)
        return cls(
            event=d.get("event", ""),
            system_id=d.get("system_id", ""),
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", ""),
            event_id=d.get("event_id", ""),
        )


# ---------------------------------------------------------------------------
# Channel — one per system_id
# ---------------------------------------------------------------------------


class WSChannel:
    """
    Manages all WebSocket connections subscribed to a given *system_id*.

    Maintains per-connection asyncio Queues and broadcasts events to
    all active connections. Dead connections are pruned on next send.
    """

    def __init__(
        self,
        system_id: str,
        max_connections: int = DEFAULT_MAX_CONNECTIONS_PER_CHANNEL,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self.system_id = system_id
        self._max_connections = max_connections
        self._max_queue_size = max_queue_size
        # connection_id → asyncio.Queue
        self._queues: dict[str, asyncio.Queue] = {}
        self._connection_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, connection_id: str, meta: dict | None = None) -> asyncio.Queue:
        """Register a new subscriber. Returns its event queue."""
        if len(self._queues) >= self._max_connections:
            raise ConnectionError(
                f"Channel {self.system_id!r} at capacity ({self._max_connections} connections)"
            )
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._queues[connection_id] = q
        self._connection_meta[connection_id] = meta or {}
        _log.debug("WSChannel[%s]: %s connected (%d total)",
                   self.system_id, connection_id, len(self._queues))
        return q

    def disconnect(self, connection_id: str) -> None:
        self._queues.pop(connection_id, None)
        self._connection_meta.pop(connection_id, None)

    @property
    def connection_count(self) -> int:
        return len(self._queues)

    def connection_ids(self) -> list[str]:
        return list(self._queues.keys())

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast(self, event: ARIAWebSocketEvent) -> int:
        """
        Enqueue *event* for all subscribers.

        Returns the number of connections that received the event.
        Slow/full queues are dropped (non-blocking put_nowait).
        """
        sent = 0
        dead: list[str] = []
        for cid, q in self._queues.items():
            try:
                q.put_nowait(event)
                sent += 1
            except asyncio.QueueFull:
                _log.warning("WSChannel[%s]: queue full for %s, dropping event", self.system_id, cid)
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)
        return sent

    def is_empty(self) -> bool:
        return len(self._queues) == 0


# ---------------------------------------------------------------------------
# ARIAWebSocketServer
# ---------------------------------------------------------------------------


class ARIAWebSocketServer:
    """
    Central registry and event dispatcher for ARIA WebSocket channels.

    One instance is shared across all WebSocket connections in a process.
    Thread-safe for channel registration; individual channels use asyncio
    Queues and are not thread-safe for broadcasting.

    Usage::

        server = ARIAWebSocketServer()

        # Register a connection (called from the WS endpoint coroutine)
        q = server.subscribe("acme", connection_id="conn-1")

        # Push events from the auditor callback
        server.emit(ARIAWebSocketEvent("inference.record", "acme", payload={...}))

        # In the endpoint coroutine: consume from q
        event = await q.get()
        await websocket.send_text(event.to_json())
    """

    def __init__(
        self,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._channels: dict[str, WSChannel] = {}
        self._heartbeat_interval = heartbeat_interval
        self._max_queue_size = max_queue_size
        self._emitted: list[ARIAWebSocketEvent] = []  # for testing
        self._event_counter = 0

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def _get_or_create_channel(self, system_id: str) -> WSChannel:
        if system_id not in self._channels:
            self._channels[system_id] = WSChannel(
                system_id=system_id,
                max_queue_size=self._max_queue_size,
            )
        return self._channels[system_id]

    def subscribe(
        self,
        system_id: str,
        connection_id: str,
        meta: dict | None = None,
    ) -> asyncio.Queue:
        """Subscribe *connection_id* to *system_id*. Returns the event queue."""
        ch = self._get_or_create_channel(system_id)
        q = ch.connect(connection_id, meta)
        # Send ack event
        ack = ARIAWebSocketEvent(
            event=WSEventType.SUBSCRIBE_ACK,
            system_id=system_id,
            payload={"connection_id": connection_id, "system_id": system_id},
        )
        try:
            q.put_nowait(ack)
        except asyncio.QueueFull:
            pass
        return q

    def unsubscribe(self, system_id: str, connection_id: str) -> None:
        ch = self._channels.get(system_id)
        if ch:
            ch.disconnect(connection_id)
            if ch.is_empty():
                del self._channels[system_id]

    def channel_ids(self) -> list[str]:
        return list(self._channels.keys())

    def connection_count(self, system_id: str) -> int:
        ch = self._channels.get(system_id)
        return ch.connection_count if ch else 0

    def total_connections(self) -> int:
        return sum(ch.connection_count for ch in self._channels.values())

    # ------------------------------------------------------------------
    # Emitting events
    # ------------------------------------------------------------------

    def emit(self, event: ARIAWebSocketEvent) -> int:
        """Broadcast *event* to all subscribers of *event.system_id*."""
        self._event_counter += 1
        if not event.event_id:
            event.event_id = f"evt-{self._event_counter:08d}"
        self._emitted.append(event)
        ch = self._channels.get(event.system_id)
        if ch is None:
            return 0
        return ch.broadcast(event)

    def emit_inference_record(
        self,
        system_id: str,
        model_id: str,
        record_id: str,
        confidence: float,
        latency_ms: int = 0,
        extra: dict | None = None,
    ) -> int:
        payload = {
            "record_id": record_id,
            "model_id": model_id,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }
        if extra:
            payload.update(extra)
        return self.emit(ARIAWebSocketEvent(
            event=WSEventType.INFERENCE_RECORD,
            system_id=system_id,
            payload=payload,
        ))

    def emit_epoch_open(self, system_id: str, epoch_id: str, extra: dict | None = None) -> int:
        payload = {"epoch_id": epoch_id, **(extra or {})}
        return self.emit(ARIAWebSocketEvent(
            event=WSEventType.EPOCH_OPEN,
            system_id=system_id,
            payload=payload,
        ))

    def emit_epoch_close(
        self,
        system_id: str,
        epoch_id: str,
        bsv_tx: str = "",
        records_count: int = 0,
        extra: dict | None = None,
    ) -> int:
        payload = {
            "epoch_id": epoch_id,
            "bsv_tx": bsv_tx,
            "records_count": records_count,
            **(extra or {}),
        }
        return self.emit(ARIAWebSocketEvent(
            event=WSEventType.EPOCH_CLOSE,
            system_id=system_id,
            payload=payload,
        ))

    def emit_error(self, system_id: str, message: str) -> int:
        return self.emit(ARIAWebSocketEvent(
            event=WSEventType.ERROR,
            system_id=system_id,
            payload={"message": message},
        ))

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def heartbeat_loop(self, system_id: str) -> None:
        """
        Coroutine that sends periodic heartbeat events to all subscribers
        of *system_id*. Run as a background task.
        """
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            ch = self._channels.get(system_id)
            if ch is None or ch.is_empty():
                break
            self.emit(ARIAWebSocketEvent(
                event=WSEventType.HEARTBEAT,
                system_id=system_id,
                payload={"ts": time.time()},
            ))

    # ------------------------------------------------------------------
    # FastAPI integration helper
    # ------------------------------------------------------------------

    async def handle_connection(
        self,
        websocket: Any,
        system_id: str,
        connection_id: str,
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        """
        Manages the full lifecycle of a single WebSocket connection.

        Subscribes the connection, relays queued events to the WebSocket,
        and cleans up on disconnect.

        Intended to be called from a FastAPI WebSocket endpoint::

            @app.websocket("/ws/{system_id}")
            async def ws_endpoint(websocket: WebSocket, system_id: str):
                await server.handle_connection(websocket, system_id, str(id(websocket)))
        """
        await websocket.accept()
        q = self.subscribe(system_id, connection_id)
        try:
            while True:
                event: ARIAWebSocketEvent = await q.get()
                await websocket.send_text(event.to_json())
                if event.event == WSEventType.DISCONNECT:
                    break
        except Exception as exc:
            _log.debug("WS connection %s closed: %s", connection_id, exc)
        finally:
            self.unsubscribe(system_id, connection_id)
            if on_disconnect:
                on_disconnect()


# ---------------------------------------------------------------------------
# FastAPI router (optional, imported lazily)
# ---------------------------------------------------------------------------


def make_ws_router(server: ARIAWebSocketServer):  # type: ignore[return]
    """
    Build a FastAPI APIRouter with the WebSocket endpoint.

    Usage::

        from fastapi import FastAPI
        from portal.backend.websocket import ARIAWebSocketServer, make_ws_router

        ws_server = ARIAWebSocketServer()
        app = FastAPI()
        app.include_router(make_ws_router(ws_server))
    """
    try:
        from fastapi import APIRouter, WebSocket
        from fastapi.websockets import WebSocketDisconnect
    except ImportError:
        return None

    router = APIRouter()

    @router.websocket("/ws/{system_id}")
    async def ws_endpoint(websocket: WebSocket, system_id: str) -> None:
        connection_id = f"conn-{id(websocket)}"
        await server.handle_connection(websocket, system_id, connection_id)

    return router
