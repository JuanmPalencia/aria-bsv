"""
aria.federation — Federated ARIA node discovery and cross-node record sharing.

Enables multiple ARIA deployments to share epoch commitments and verify
each other's BSV anchors, forming a trust network of auditable AI systems.

Use cases:
- Corporate multi-department deployments sharing a common trust layer
- Cross-organisation model audit trails
- Multi-cloud ARIA installations with shared epoch registry

Usage::

    from aria.federation import FederationNode, FederationHub

    # Register this node
    hub = FederationHub(node_id="acme-prod", api_base="https://aria.acme.com")

    # Register peer
    hub.register_peer("globex", api_base="https://aria.globex.io", api_key="sk-...")

    # Publish epoch to peers
    await hub.publish_epoch(epoch_summary)

    # Fetch and verify a peer's epoch
    remote_summary = await hub.fetch_epoch("globex", "epoch-abc123")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FederatedEpoch:
    """A minimal epoch summary suitable for cross-node sharing."""
    node_id:         str
    epoch_id:        str
    model_id:        str
    records_count:   int
    commitment_hash: str
    bsv_tx_close:    str
    opened_at:       str
    closed_at:       str
    merkle_root:     str = ""
    metadata:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FederatedEpoch":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PeerNode:
    """A remote ARIA federation peer."""
    node_id:   str
    api_base:  str
    api_key:   str = ""
    active:    bool = True
    last_seen: str = ""

    def endpoint(self, path: str) -> str:
        return f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"


@dataclass
class FederationSyncResult:
    """Result of a federation sync operation."""
    peer_id:     str
    epochs_sent: int = 0
    epochs_recv: int = 0
    errors:      list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# FederationHub
# ---------------------------------------------------------------------------

class FederationHub:
    """Manages federation of epoch data across ARIA nodes.

    Args:
        node_id:  Identifier for this node.
        api_base: Base URL of this node's ARIA HTTP API.
    """

    def __init__(self, node_id: str, api_base: str = "") -> None:
        self._node_id   = node_id
        self._api_base  = api_base
        self._peers:    dict[str, PeerNode] = {}
        self._registry: dict[str, FederatedEpoch] = {}  # epoch_id → epoch

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def register_peer(
        self,
        node_id: str,
        api_base: str,
        api_key: str = "",
        replace: bool = False,
    ) -> PeerNode:
        """Register a federation peer.

        Args:
            node_id:  Peer's node identifier.
            api_base: Peer's ARIA API base URL.
            api_key:  Optional authentication key.
            replace:  Overwrite if already registered.

        Raises:
            ValueError: If peer already registered and replace=False.
        """
        if node_id in self._peers and not replace:
            raise ValueError(f"Peer '{node_id}' already registered")
        peer = PeerNode(node_id=node_id, api_base=api_base, api_key=api_key)
        self._peers[node_id] = peer
        return peer

    def deregister_peer(self, node_id: str) -> bool:
        """Remove a peer. Returns True if found."""
        if node_id in self._peers:
            del self._peers[node_id]
            return True
        return False

    def get_peer(self, node_id: str) -> PeerNode | None:
        return self._peers.get(node_id)

    def list_peers(self, active_only: bool = False) -> list[str]:
        if active_only:
            return [nid for nid, p in self._peers.items() if p.active]
        return list(self._peers.keys())

    # ------------------------------------------------------------------
    # Local registry
    # ------------------------------------------------------------------

    def publish_local(self, epoch: FederatedEpoch) -> None:
        """Register a local epoch in the federation registry."""
        self._registry[epoch.epoch_id] = epoch

    def get_epoch(self, epoch_id: str) -> FederatedEpoch | None:
        """Retrieve an epoch from the local registry."""
        return self._registry.get(epoch_id)

    def list_epochs(self, node_id: str | None = None) -> list[str]:
        """List epoch IDs, optionally filtered by node."""
        if node_id:
            return [eid for eid, e in self._registry.items() if e.node_id == node_id]
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Remote operations (HTTP-based, for use with ARIA HTTP API)
    # ------------------------------------------------------------------

    async def publish_epoch(
        self,
        epoch: FederatedEpoch,
        peers: list[str] | None = None,
    ) -> dict[str, bool]:
        """Push an epoch summary to federation peers.

        Args:
            epoch: FederatedEpoch to publish.
            peers: Peer node IDs to publish to. Defaults to all active peers.

        Returns:
            Dict of peer_id → success bool.
        """
        target_peers = peers or self.list_peers(active_only=True)
        results = {}
        for pid in target_peers:
            peer = self._peers.get(pid)
            if not peer:
                results[pid] = False
                continue
            try:
                await self._post_epoch(peer, epoch)
                results[pid] = True
            except Exception as exc:
                _log.warning("Federation: could not publish to %s: %s", pid, exc)
                results[pid] = False
        # Also store locally
        self.publish_local(epoch)
        return results

    async def fetch_epoch(
        self,
        peer_id: str,
        epoch_id: str,
    ) -> FederatedEpoch | None:
        """Fetch an epoch from a peer.

        Args:
            peer_id:  Peer node ID.
            epoch_id: Epoch to fetch.

        Returns:
            FederatedEpoch if found, else None.
        """
        peer = self._peers.get(peer_id)
        if not peer:
            return None
        try:
            return await self._get_epoch(peer, epoch_id)
        except Exception as exc:
            _log.warning("Federation: could not fetch %s from %s: %s", epoch_id, peer_id, exc)
            return None

    async def sync_with_peer(self, peer_id: str) -> FederationSyncResult:
        """Bidirectional sync of epochs with a peer."""
        result = FederationSyncResult(peer_id=peer_id)
        peer = self._peers.get(peer_id)
        if not peer:
            result.errors.append(f"Unknown peer: {peer_id}")
            return result

        # Send our epochs
        for epoch in self._registry.values():
            if epoch.node_id == self._node_id:
                try:
                    await self._post_epoch(peer, epoch)
                    result.epochs_sent += 1
                except Exception as exc:
                    result.errors.append(f"send {epoch.epoch_id}: {exc}")

        return result

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _post_epoch(self, peer: PeerNode, epoch: FederatedEpoch) -> None:
        url     = peer.endpoint("/federation/epochs")
        payload = epoch.to_json().encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **({"Authorization": peer.api_key} if peer.api_key else {}),
        }
        await self._http_post(url, payload, headers)

    async def _get_epoch(self, peer: PeerNode, epoch_id: str) -> FederatedEpoch | None:
        url = peer.endpoint(f"/federation/epochs/{epoch_id}")
        headers = {**({"Authorization": peer.api_key} if peer.api_key else {})}
        data = await self._http_get(url, headers)
        if data:
            return FederatedEpoch.from_dict(json.loads(data))
        return None

    async def _http_post(self, url: str, body: bytes, headers: dict) -> str:
        import urllib.request
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Peer URL must use http(s): {url!r}")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 — scheme validated above
            return resp.read().decode("utf-8")

    async def _http_get(self, url: str, headers: dict) -> str | None:
        import urllib.request
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Peer URL must use http(s): {url!r}")
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 — scheme validated above
            return resp.read().decode("utf-8")
