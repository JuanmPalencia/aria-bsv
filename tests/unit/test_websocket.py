"""Tests for portal.backend.websocket — WebSocket event server."""

from __future__ import annotations

import asyncio
import json

import pytest

from portal.backend.websocket import (
    ARIAWebSocketEvent,
    ARIAWebSocketServer,
    WSChannel,
    WSEventType,
)


# ---------------------------------------------------------------------------
# ARIAWebSocketEvent
# ---------------------------------------------------------------------------

class TestARIAWebSocketEvent:
    def test_auto_timestamp(self):
        e = ARIAWebSocketEvent(event="inference.record", system_id="acme")
        assert e.timestamp != ""

    def test_to_json(self):
        e = ARIAWebSocketEvent(
            event="inference.record",
            system_id="acme",
            payload={"model": "gpt-4", "confidence": 0.99},
        )
        d = json.loads(e.to_json())
        assert d["event"] == "inference.record"
        assert d["system_id"] == "acme"
        assert d["payload"]["model"] == "gpt-4"

    def test_from_json_roundtrip(self):
        e = ARIAWebSocketEvent(
            event="epoch.close", system_id="test",
            payload={"epoch_id": "ep-1"}, event_id="evt-001",
        )
        e2 = ARIAWebSocketEvent.from_json(e.to_json())
        assert e2.event == e.event
        assert e2.system_id == e.system_id
        assert e2.payload["epoch_id"] == "ep-1"
        assert e2.event_id == "evt-001"

    def test_from_json_missing_keys(self):
        raw = json.dumps({"event": "x", "system_id": "s"})
        e = ARIAWebSocketEvent.from_json(raw)
        assert e.payload == {}
        assert e.event_id == ""

    def test_to_json_has_timestamp(self):
        e = ARIAWebSocketEvent(event="x", system_id="s")
        d = json.loads(e.to_json())
        assert "timestamp" in d

    def test_default_payload_empty(self):
        e = ARIAWebSocketEvent(event="x", system_id="s")
        assert e.payload == {}


# ---------------------------------------------------------------------------
# WSChannel
# ---------------------------------------------------------------------------

class TestWSChannel:
    def test_connect_returns_queue(self):
        ch = WSChannel("acme")
        q = ch.connect("c1")
        assert isinstance(q, asyncio.Queue)

    def test_connect_increments_count(self):
        ch = WSChannel("acme")
        ch.connect("c1")
        ch.connect("c2")
        assert ch.connection_count == 2

    def test_disconnect_removes(self):
        ch = WSChannel("acme")
        ch.connect("c1")
        ch.disconnect("c1")
        assert ch.connection_count == 0

    def test_disconnect_unknown_is_noop(self):
        ch = WSChannel("acme")
        ch.disconnect("ghost")  # should not raise

    def test_is_empty_initially(self):
        ch = WSChannel("acme")
        assert ch.is_empty() is True

    def test_is_empty_after_connect(self):
        ch = WSChannel("acme")
        ch.connect("c1")
        assert ch.is_empty() is False

    def test_is_empty_after_all_disconnect(self):
        ch = WSChannel("acme")
        ch.connect("c1")
        ch.disconnect("c1")
        assert ch.is_empty() is True

    def test_max_connections_raises(self):
        ch = WSChannel("acme", max_connections=2)
        ch.connect("c1")
        ch.connect("c2")
        with pytest.raises(ConnectionError):
            ch.connect("c3")

    def test_broadcast_delivers_event(self):
        ch = WSChannel("acme")
        q = ch.connect("c1")
        evt = ARIAWebSocketEvent(event="inference.record", system_id="acme", payload={"k": "v"})
        sent = ch.broadcast(evt)
        assert sent == 1
        item = q.get_nowait()
        assert item.event == "inference.record"

    def test_broadcast_to_multiple(self):
        ch = WSChannel("acme")
        q1 = ch.connect("c1")
        q2 = ch.connect("c2")
        evt = ARIAWebSocketEvent(event="epoch.open", system_id="acme")
        sent = ch.broadcast(evt)
        assert sent == 2
        assert not q1.empty()
        assert not q2.empty()

    def test_broadcast_full_queue_drops_connection(self):
        ch = WSChannel("acme", max_queue_size=1)
        q = ch.connect("c1")
        evt = ARIAWebSocketEvent(event="x", system_id="acme")
        # Fill the queue first
        q.put_nowait(evt)
        # Now it's full; broadcast should drop this connection
        sent = ch.broadcast(evt)
        assert sent == 0
        assert ch.connection_count == 0

    def test_connection_ids(self):
        ch = WSChannel("acme")
        ch.connect("c1")
        ch.connect("c2")
        assert set(ch.connection_ids()) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# ARIAWebSocketServer — channel management
# ---------------------------------------------------------------------------

class TestARIAWebSocketServerChannels:
    def test_subscribe_creates_channel(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("acme", "c1")
        assert "acme" in srv.channel_ids()

    def test_subscribe_returns_queue(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        assert isinstance(q, asyncio.Queue)

    def test_subscribe_sends_ack(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        ack = q.get_nowait()
        assert ack.event == WSEventType.SUBSCRIBE_ACK
        assert ack.payload["connection_id"] == "c1"

    def test_unsubscribe_removes_connection(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("acme", "c1")
        srv.unsubscribe("acme", "c1")
        assert srv.connection_count("acme") == 0

    def test_unsubscribe_removes_empty_channel(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("acme", "c1")
        srv.unsubscribe("acme", "c1")
        assert "acme" not in srv.channel_ids()

    def test_unsubscribe_unknown_channel_noop(self):
        srv = ARIAWebSocketServer()
        srv.unsubscribe("ghost", "c1")  # should not raise

    def test_multiple_systems(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("sys-a", "c1")
        srv.subscribe("sys-b", "c2")
        assert set(srv.channel_ids()) == {"sys-a", "sys-b"}

    def test_connection_count(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("acme", "c1")
        srv.subscribe("acme", "c2")
        assert srv.connection_count("acme") == 2

    def test_connection_count_unknown(self):
        srv = ARIAWebSocketServer()
        assert srv.connection_count("ghost") == 0

    def test_total_connections(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("a", "c1")
        srv.subscribe("a", "c2")
        srv.subscribe("b", "c3")
        assert srv.total_connections() == 3


# ---------------------------------------------------------------------------
# ARIAWebSocketServer — emit
# ---------------------------------------------------------------------------

class TestARIAWebSocketServerEmit:
    def test_emit_no_subscribers_returns_zero(self):
        srv = ARIAWebSocketServer()
        n = srv.emit(ARIAWebSocketEvent(event="x", system_id="acme"))
        assert n == 0

    def test_emit_delivers_to_subscriber(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # consume ack
        evt = ARIAWebSocketEvent(event="inference.record", system_id="acme", payload={"m": "gpt-4"})
        srv.emit(evt)
        item = q.get_nowait()
        assert item.event == "inference.record"
        assert item.payload["m"] == "gpt-4"

    def test_emit_assigns_event_id(self):
        srv = ARIAWebSocketServer()
        evt = ARIAWebSocketEvent(event="x", system_id="acme")
        assert evt.event_id == ""
        srv.emit(evt)
        assert evt.event_id != ""

    def test_emit_preserves_explicit_event_id(self):
        srv = ARIAWebSocketServer()
        evt = ARIAWebSocketEvent(event="x", system_id="acme", event_id="my-id")
        srv.emit(evt)
        assert evt.event_id == "my-id"

    def test_emit_stores_in_emitted(self):
        srv = ARIAWebSocketServer()
        srv.emit(ARIAWebSocketEvent(event="x", system_id="acme"))
        assert len(srv._emitted) == 1

    def test_emit_inference_record(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack
        srv.emit_inference_record("acme", "gpt-4", "rec-001", 0.95, latency_ms=50)
        item = q.get_nowait()
        assert item.event == WSEventType.INFERENCE_RECORD
        assert item.payload["model_id"] == "gpt-4"
        assert item.payload["confidence"] == 0.95
        assert item.payload["latency_ms"] == 50

    def test_emit_epoch_open(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack
        srv.emit_epoch_open("acme", "ep-001")
        item = q.get_nowait()
        assert item.event == WSEventType.EPOCH_OPEN
        assert item.payload["epoch_id"] == "ep-001"

    def test_emit_epoch_close(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack
        srv.emit_epoch_close("acme", "ep-001", bsv_tx="abc123", records_count=100)
        item = q.get_nowait()
        assert item.event == WSEventType.EPOCH_CLOSE
        assert item.payload["bsv_tx"] == "abc123"
        assert item.payload["records_count"] == 100

    def test_emit_error(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack
        srv.emit_error("acme", "something went wrong")
        item = q.get_nowait()
        assert item.event == WSEventType.ERROR
        assert "something went wrong" in item.payload["message"]

    def test_emit_inference_record_extra_payload(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack
        srv.emit_inference_record(
            "acme", "gpt-4", "rec-1", 0.9, extra={"session": "s123"}
        )
        item = q.get_nowait()
        assert item.payload["session"] == "s123"

    def test_emit_returns_delivery_count(self):
        srv = ARIAWebSocketServer()
        srv.subscribe("acme", "c1")
        srv.subscribe("acme", "c2")
        evt = ARIAWebSocketEvent(event="x", system_id="acme")
        n = srv.emit(evt)
        assert n == 2

    def test_emit_different_system_not_delivered(self):
        srv = ARIAWebSocketServer()
        q = srv.subscribe("sys-a", "c1")
        q.get_nowait()  # ack
        srv.emit(ARIAWebSocketEvent(event="x", system_id="sys-b"))
        assert q.empty()


# ---------------------------------------------------------------------------
# WSEventType enum
# ---------------------------------------------------------------------------

class TestWSEventType:
    def test_values(self):
        assert WSEventType.INFERENCE_RECORD == "inference.record"
        assert WSEventType.EPOCH_OPEN == "epoch.open"
        assert WSEventType.EPOCH_CLOSE == "epoch.close"
        assert WSEventType.HEARTBEAT == "system.heartbeat"
        assert WSEventType.ERROR == "system.error"
        assert WSEventType.SUBSCRIBE_ACK == "system.subscribe_ack"
        assert WSEventType.DISCONNECT == "system.disconnect"


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    def test_heartbeat_emits_to_channel(self):
        srv = ARIAWebSocketServer(heartbeat_interval=0.05)
        q = srv.subscribe("acme", "c1")
        q.get_nowait()  # ack

        async def _run():
            task = asyncio.create_task(srv.heartbeat_loop("acme"))
            await asyncio.sleep(0.12)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        # At least one heartbeat should have been enqueued
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        heartbeats = [i for i in items if i.event == WSEventType.HEARTBEAT]
        assert len(heartbeats) >= 1

    def test_heartbeat_stops_when_channel_empty(self):
        srv = ARIAWebSocketServer(heartbeat_interval=0.05)
        srv.subscribe("acme", "c1")
        srv.unsubscribe("acme", "c1")

        async def _run():
            # Should exit quickly since channel is empty
            await asyncio.wait_for(srv.heartbeat_loop("acme"), timeout=0.2)

        asyncio.run(_run())
