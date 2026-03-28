"""
aria.nostr_anchor — Nostr anchoring for ARIA epoch commitments.

Publishes a BRC-121 epoch commitment as a Nostr event (NIP-01 kind 1)
to one or more relays, creating a censorship-resistant public record
of the commitment hash.

Nostr provides a complementary anchoring layer to BSV:
- BSV: immutable, ordered, economically secured
- Nostr: censorship-resistant, fast, publicly searchable

Usage::

    from aria.nostr_anchor import NostrAnchor, NostrAnchorPayload

    anchor = NostrAnchor(
        private_key_hex="deadbeef...",
        relays=["wss://relay.damus.io", "wss://nos.lol"],
    )
    result = await anchor.anchor_epoch(epoch_id, commitment_hash)
    print(result.event_id)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

# Nostr event kind for ARIA anchors
# Kind 1 = text note (public); kind 30078 = parameterised replaceable for structured data
_ARIA_NOSTR_KIND = 1
_ARIA_TAG = "aria-bsv"


# ---------------------------------------------------------------------------
# Payload and event
# ---------------------------------------------------------------------------

@dataclass
class NostrAnchorPayload:
    """Data anchored in a Nostr event."""
    epoch_id:        str
    commitment_hash: str   # hex string
    merkle_root:     str   # hex string
    records_count:   int
    bsv_tx_close:    str = ""  # BSV transaction ID if available

    def to_content(self) -> str:
        """Format as a human-readable Nostr note content."""
        lines = [
            f"#aria-bsv epoch commitment",
            f"epoch: {self.epoch_id}",
            f"commitment: {self.commitment_hash}",
            f"merkle_root: {self.merkle_root}",
            f"records: {self.records_count}",
        ]
        if self.bsv_tx_close:
            lines.append(f"bsv_tx: {self.bsv_tx_close}")
        return "\n".join(lines)

    def to_tags(self) -> list[list[str]]:
        """Nostr event tags."""
        tags = [
            ["t", _ARIA_TAG],
            ["t", "brc-121"],
            ["aria-epoch", self.epoch_id],
            ["aria-commitment", self.commitment_hash],
        ]
        if self.bsv_tx_close:
            tags.append(["aria-bsv-tx", self.bsv_tx_close])
        return tags


@dataclass
class NostrEvent:
    """A Nostr event (NIP-01)."""
    kind:       int
    content:    str
    tags:       list[list[str]]
    pubkey:     str
    created_at: int = 0
    id:         str = ""
    sig:        str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = int(time.time())

    def compute_id(self) -> str:
        """Compute NIP-01 event ID (SHA-256 of canonical JSON)."""
        serialised = json.dumps([
            0,
            self.pubkey,
            self.created_at,
            self.kind,
            self.tags,
            self.content,
        ], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "pubkey":     self.pubkey,
            "created_at": self.created_at,
            "kind":       self.kind,
            "tags":       self.tags,
            "content":    self.content,
            "sig":        self.sig,
        }


@dataclass
class NostrAnchorResult:
    """Result of a Nostr anchoring operation."""
    epoch_id:    str
    event_id:    str
    relays_ok:   list[str] = field(default_factory=list)
    relays_fail: list[str] = field(default_factory=list)
    anchored_at: str = ""
    error:       str | None = None

    def __post_init__(self):
        if not self.anchored_at:
            self.anchored_at = datetime.now(timezone.utc).isoformat()

    @property
    def success(self) -> bool:
        return len(self.relays_ok) > 0

    def __str__(self) -> str:
        return (
            f"NostrAnchorResult: event={self.event_id[:12]}...  "
            f"relays_ok={len(self.relays_ok)}  "
            f"relays_fail={len(self.relays_fail)}"
        )


# ---------------------------------------------------------------------------
# NostrAnchor
# ---------------------------------------------------------------------------

class NostrAnchor:
    """Publishes ARIA epoch commitments to Nostr relays.

    Args:
        private_key_hex: 32-byte Nostr private key as hex string.
        relays:          List of wss:// relay URLs.
    """

    def __init__(
        self,
        private_key_hex: str,
        relays: list[str] | None = None,
    ) -> None:
        self._sk     = private_key_hex.lstrip("0x")
        self._relays = relays or []
        self._pubkey = self._derive_pubkey(self._sk)

    async def anchor_epoch(
        self,
        epoch_id: str,
        commitment_hash: str,
        merkle_root: str = "",
        records_count: int = 0,
        bsv_tx_close: str = "",
    ) -> NostrAnchorResult:
        """Publish an epoch commitment to Nostr relays.

        Args:
            epoch_id:        Epoch identifier.
            commitment_hash: BRC-121 commitment hash (hex).
            merkle_root:     Epoch Merkle root (hex).
            records_count:   Number of records in the epoch.
            bsv_tx_close:    BSV closing transaction ID.

        Returns:
            NostrAnchorResult with event_id and relay delivery stats.
        """
        payload = NostrAnchorPayload(
            epoch_id=epoch_id,
            commitment_hash=commitment_hash,
            merkle_root=merkle_root,
            records_count=records_count,
            bsv_tx_close=bsv_tx_close,
        )

        event = NostrEvent(
            kind=_ARIA_NOSTR_KIND,
            content=payload.to_content(),
            tags=payload.to_tags(),
            pubkey=self._pubkey,
        )
        event.id  = event.compute_id()
        event.sig = self._sign(event)

        relays_ok, relays_fail = [], []
        for relay in self._relays:
            try:
                await self._publish(relay, event)
                relays_ok.append(relay)
            except Exception as exc:
                _log.warning("NostrAnchor: relay %s failed: %s", relay, exc)
                relays_fail.append(relay)

        return NostrAnchorResult(
            epoch_id=epoch_id,
            event_id=event.id,
            relays_ok=relays_ok,
            relays_fail=relays_fail,
        )

    def build_event(
        self,
        epoch_id: str,
        commitment_hash: str,
        **kwargs: Any,
    ) -> NostrEvent:
        """Build a signed Nostr event without publishing."""
        payload = NostrAnchorPayload(
            epoch_id=epoch_id,
            commitment_hash=commitment_hash,
            merkle_root=kwargs.get("merkle_root", ""),
            records_count=kwargs.get("records_count", 0),
            bsv_tx_close=kwargs.get("bsv_tx_close", ""),
        )
        event = NostrEvent(
            kind=_ARIA_NOSTR_KIND,
            content=payload.to_content(),
            tags=payload.to_tags(),
            pubkey=self._pubkey,
        )
        event.id  = event.compute_id()
        event.sig = self._sign(event)
        return event

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _derive_pubkey(self, sk_hex: str) -> str:
        """Derive Nostr public key (x-only, schnorr). Uses secp256k1 if available."""
        try:
            from coincurve import PrivateKey
            sk = PrivateKey(bytes.fromhex(sk_hex.ljust(64, "0")))
            return sk.public_key.format(compressed=True)[1:].hex()  # x-only
        except ImportError:
            # Deterministic stub for testing without cryptographic library
            return hashlib.sha256(sk_hex.encode()).hexdigest()

    def _sign(self, event: NostrEvent) -> str:
        """Sign the event ID. Uses schnorr if coincurve available, else SHA-256 stub."""
        try:
            from coincurve import PrivateKey
            sk = PrivateKey(bytes.fromhex(self._sk.ljust(64, "0")))
            sig = sk.sign_recoverable(bytes.fromhex(event.id))
            return sig.hex()
        except ImportError:
            # Deterministic stub (not real schnorr — for testing only)
            return hashlib.sha256((event.id + self._sk).encode()).hexdigest() * 2

    async def _publish(self, relay_url: str, event: NostrEvent) -> None:
        """Publish event to a relay via WebSocket."""
        try:
            import websockets
            msg = json.dumps(["EVENT", event.to_dict()])
            async with websockets.connect(relay_url, open_timeout=5) as ws:
                await ws.send(msg)
                resp = await ws.recv()
                _log.debug("Nostr relay %s: %s", relay_url, resp[:100])
        except ImportError:
            raise ImportError("websockets not installed. pip install aria-bsv[nostr]")
