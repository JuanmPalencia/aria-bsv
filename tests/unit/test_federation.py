"""Tests for aria.federation — FederationHub."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.federation import (
    FederatedEpoch,
    FederationHub,
    FederationSyncResult,
    PeerNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epoch(epoch_id="ep-1", node_id="node-a"):
    return FederatedEpoch(
        node_id=node_id,
        epoch_id=epoch_id,
        model_id="gpt-4o",
        records_count=50,
        commitment_hash="0x" + "ab" * 32,
        bsv_tx_close="bsv-tx-abc",
        opened_at="2025-01-01T00:00:00",
        closed_at="2025-01-01T01:00:00",
    )


def _hub(node_id="node-a"):
    return FederationHub(node_id=node_id, api_base="https://aria.test")


# ---------------------------------------------------------------------------
# FederatedEpoch
# ---------------------------------------------------------------------------

class TestFederatedEpoch:
    def test_to_dict(self):
        ep = _epoch()
        d = ep.to_dict()
        assert d["epoch_id"] == "ep-1"
        assert d["node_id"] == "node-a"

    def test_from_dict(self):
        ep = _epoch()
        d = ep.to_dict()
        ep2 = FederatedEpoch.from_dict(d)
        assert ep2.epoch_id == ep.epoch_id
        assert ep2.model_id == ep.model_id

    def test_roundtrip(self):
        ep = _epoch()
        ep2 = FederatedEpoch.from_dict(ep.to_dict())
        assert ep == ep2

    def test_to_json(self):
        ep = _epoch()
        j = ep.to_json()
        d = json.loads(j)
        assert d["epoch_id"] == "ep-1"

    def test_from_dict_ignores_unknown_keys(self):
        ep = _epoch()
        d = ep.to_dict()
        d["unknown_key"] = "should be ignored"
        ep2 = FederatedEpoch.from_dict(d)
        assert ep2.epoch_id == "ep-1"


# ---------------------------------------------------------------------------
# PeerNode
# ---------------------------------------------------------------------------

class TestPeerNode:
    def test_endpoint(self):
        peer = PeerNode(node_id="p", api_base="https://aria.peer.io")
        assert peer.endpoint("/epochs") == "https://aria.peer.io/epochs"

    def test_endpoint_strips_slash(self):
        peer = PeerNode(node_id="p", api_base="https://aria.peer.io/")
        assert peer.endpoint("/epochs") == "https://aria.peer.io/epochs"


# ---------------------------------------------------------------------------
# FederationHub — peer management
# ---------------------------------------------------------------------------

class TestFederationHubPeers:
    def test_register_peer(self):
        hub = _hub()
        peer = hub.register_peer("globex", "https://aria.globex.io")
        assert peer.node_id == "globex"
        assert hub.get_peer("globex") is peer

    def test_register_duplicate_raises(self):
        hub = _hub()
        hub.register_peer("globex", "https://a.io")
        with pytest.raises(ValueError, match="globex"):
            hub.register_peer("globex", "https://b.io")

    def test_register_replace(self):
        hub = _hub()
        hub.register_peer("globex", "https://old.io")
        hub.register_peer("globex", "https://new.io", replace=True)
        assert hub.get_peer("globex").api_base == "https://new.io"

    def test_deregister_peer(self):
        hub = _hub()
        hub.register_peer("globex", "https://a.io")
        assert hub.deregister_peer("globex") is True
        assert hub.get_peer("globex") is None

    def test_deregister_unknown(self):
        hub = _hub()
        assert hub.deregister_peer("unknown") is False

    def test_list_peers(self):
        hub = _hub()
        hub.register_peer("p1", "https://p1.io")
        hub.register_peer("p2", "https://p2.io")
        assert set(hub.list_peers()) == {"p1", "p2"}

    def test_list_peers_active_only(self):
        hub = _hub()
        hub.register_peer("p1", "https://p1.io")
        hub.register_peer("p2", "https://p2.io")
        hub.get_peer("p2").active = False
        assert hub.list_peers(active_only=True) == ["p1"]


# ---------------------------------------------------------------------------
# FederationHub — local registry
# ---------------------------------------------------------------------------

class TestFederationHubRegistry:
    def test_publish_and_get(self):
        hub = _hub()
        ep = _epoch("ep-1")
        hub.publish_local(ep)
        assert hub.get_epoch("ep-1") is ep

    def test_get_unknown(self):
        hub = _hub()
        assert hub.get_epoch("unknown") is None

    def test_list_epochs(self):
        hub = _hub()
        hub.publish_local(_epoch("ep-1"))
        hub.publish_local(_epoch("ep-2"))
        assert set(hub.list_epochs()) == {"ep-1", "ep-2"}

    def test_list_epochs_by_node(self):
        hub = _hub()
        hub.publish_local(_epoch("ep-1", node_id="node-a"))
        hub.publish_local(_epoch("ep-2", node_id="node-b"))
        epochs = hub.list_epochs(node_id="node-a")
        assert epochs == ["ep-1"]


# ---------------------------------------------------------------------------
# FederationHub — publish_epoch
# ---------------------------------------------------------------------------

class TestFederationHubPublish:
    def test_publish_to_peer_success(self):
        hub = _hub()
        hub.register_peer("globex", "https://g.io")
        ep = _epoch()

        async def _run():
            with patch.object(hub, "_post_epoch", new=AsyncMock()):
                return await hub.publish_epoch(ep, peers=["globex"])

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results["globex"] is True
        assert hub.get_epoch("ep-1") is ep  # also stored locally

    def test_publish_to_peer_failure(self):
        hub = _hub()
        hub.register_peer("globex", "https://g.io")
        ep = _epoch()

        async def _run():
            with patch.object(hub, "_post_epoch", new=AsyncMock(side_effect=RuntimeError("fail"))):
                return await hub.publish_epoch(ep, peers=["globex"])

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results["globex"] is False

    def test_publish_unknown_peer(self):
        hub = _hub()
        ep = _epoch()

        async def _run():
            return await hub.publish_epoch(ep, peers=["unknown"])

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results["unknown"] is False

    def test_publish_to_all_active_peers(self):
        hub = _hub()
        hub.register_peer("p1", "https://p1.io")
        hub.register_peer("p2", "https://p2.io")
        hub.get_peer("p2").active = False
        ep = _epoch()

        call_log = []

        async def _fake_post(peer, epoch):
            call_log.append(peer.node_id)

        async def _run():
            with patch.object(hub, "_post_epoch", side_effect=_fake_post):
                return await hub.publish_epoch(ep)

        asyncio.get_event_loop().run_until_complete(_run())
        assert "p1" in call_log
        assert "p2" not in call_log


# ---------------------------------------------------------------------------
# FederationHub — fetch_epoch
# ---------------------------------------------------------------------------

class TestFederationHubFetch:
    def test_fetch_success(self):
        hub = _hub()
        hub.register_peer("globex", "https://g.io")
        remote_ep = _epoch("ep-remote", node_id="globex")

        async def _run():
            with patch.object(hub, "_get_epoch", new=AsyncMock(return_value=remote_ep)):
                return await hub.fetch_epoch("globex", "ep-remote")

        ep = asyncio.get_event_loop().run_until_complete(_run())
        assert ep is remote_ep

    def test_fetch_unknown_peer(self):
        hub = _hub()

        async def _run():
            return await hub.fetch_epoch("unknown", "ep-1")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result is None

    def test_fetch_failure_returns_none(self):
        hub = _hub()
        hub.register_peer("globex", "https://g.io")

        async def _run():
            with patch.object(hub, "_get_epoch", new=AsyncMock(side_effect=RuntimeError("err"))):
                return await hub.fetch_epoch("globex", "ep-1")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result is None


# ---------------------------------------------------------------------------
# FederationSyncResult
# ---------------------------------------------------------------------------

class TestFederationSyncResult:
    def test_success_no_errors(self):
        r = FederationSyncResult(peer_id="p", epochs_sent=3)
        assert r.success is True

    def test_failure_with_errors(self):
        r = FederationSyncResult(peer_id="p", errors=["fail1"])
        assert r.success is False
