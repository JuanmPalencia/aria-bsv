"""Tests for aria.nostr_anchor — Nostr anchoring."""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.nostr_anchor import (
    NostrAnchor,
    NostrAnchorPayload,
    NostrAnchorResult,
    NostrEvent,
)


# ---------------------------------------------------------------------------
# NostrAnchorPayload
# ---------------------------------------------------------------------------

class TestNostrAnchorPayload:
    def _payload(self):
        return NostrAnchorPayload(
            epoch_id="epoch-001",
            commitment_hash="0x" + "ab" * 32,
            merkle_root="cd" * 32,
            records_count=100,
            bsv_tx_close="abc123def456",
        )

    def test_to_content_has_epoch(self):
        content = self._payload().to_content()
        assert "epoch-001" in content

    def test_to_content_has_commitment(self):
        content = self._payload().to_content()
        assert "commitment" in content
        assert "abab" in content.lower()

    def test_to_content_has_bsv_tx(self):
        content = self._payload().to_content()
        assert "abc123def456" in content

    def test_to_content_no_bsv_tx(self):
        p = NostrAnchorPayload(
            epoch_id="ep",
            commitment_hash="0xab",
            merkle_root="cd",
            records_count=5,
        )
        content = p.to_content()
        assert "bsv_tx" not in content

    def test_to_tags_has_aria_tag(self):
        tags = self._payload().to_tags()
        t_tags = [t[1] for t in tags if t[0] == "t"]
        assert "aria-bsv" in t_tags
        assert "brc-121" in t_tags

    def test_to_tags_has_epoch(self):
        tags = self._payload().to_tags()
        epoch_tags = [t for t in tags if t[0] == "aria-epoch"]
        assert len(epoch_tags) == 1
        assert epoch_tags[0][1] == "epoch-001"

    def test_to_tags_has_bsv_tx_when_present(self):
        tags = self._payload().to_tags()
        bsv_tags = [t for t in tags if t[0] == "aria-bsv-tx"]
        assert len(bsv_tags) == 1


# ---------------------------------------------------------------------------
# NostrEvent
# ---------------------------------------------------------------------------

class TestNostrEvent:
    def _event(self):
        return NostrEvent(
            kind=1,
            content="test content",
            tags=[["t", "aria-bsv"]],
            pubkey="abc" * 21 + "ab",  # 64-char pubkey
        )

    def test_compute_id_deterministic(self):
        evt = self._event()
        id1 = evt.compute_id()
        id2 = evt.compute_id()
        assert id1 == id2

    def test_compute_id_changes_with_content(self):
        e1 = self._event()
        e2 = self._event()
        e2.content = "different"
        assert e1.compute_id() != e2.compute_id()

    def test_compute_id_is_hex(self):
        evt = self._event()
        evt_id = evt.compute_id()
        assert len(evt_id) == 64
        int(evt_id, 16)  # should not raise

    def test_created_at_auto_set(self):
        evt = self._event()
        assert evt.created_at > 0

    def test_to_dict(self):
        evt = self._event()
        evt.id = evt.compute_id()
        d = evt.to_dict()
        assert "id" in d
        assert "pubkey" in d
        assert "kind" in d
        assert "content" in d
        assert "tags" in d
        assert "sig" in d


# ---------------------------------------------------------------------------
# NostrAnchor
# ---------------------------------------------------------------------------

class TestNostrAnchor:
    def _anchor(self, relays=None):
        sk = "a" * 64  # fake key
        return NostrAnchor(private_key_hex=sk, relays=relays or [])

    def test_build_event_no_relays(self):
        anchor = self._anchor()
        evt = anchor.build_event("ep-1", "0x" + "ab" * 32)
        assert evt.kind == 1
        assert "ep-1" in evt.content
        assert evt.id != ""
        assert evt.sig != ""

    def test_event_id_valid_hex(self):
        anchor = self._anchor()
        evt = anchor.build_event("ep-1", "0x" + "cd" * 32)
        int(evt.id, 16)  # should not raise

    def test_pubkey_derived(self):
        anchor = self._anchor()
        assert len(anchor._pubkey) >= 32  # some hex string

    def test_anchor_epoch_no_relays(self):
        anchor = self._anchor(relays=[])

        async def _run():
            return await anchor.anchor_epoch("ep-1", "0xabc", merkle_root="def", records_count=5)

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.epoch_id == "ep-1"
        assert result.event_id != ""
        assert len(result.relays_ok) == 0

    def test_anchor_epoch_relay_success(self):
        anchor = self._anchor(relays=["wss://relay.test"])

        async def _run():
            with patch.object(anchor, "_publish", new=AsyncMock()):
                return await anchor.anchor_epoch("ep-1", "0xabc")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert len(result.relays_ok) == 1
        assert "wss://relay.test" in result.relays_ok

    def test_anchor_epoch_relay_failure(self):
        anchor = self._anchor(relays=["wss://bad-relay"])

        async def _run():
            with patch.object(anchor, "_publish", new=AsyncMock(side_effect=RuntimeError("conn"))):
                return await anchor.anchor_epoch("ep-1", "0xabc")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert len(result.relays_fail) == 1
        assert len(result.relays_ok) == 0

    def test_anchor_epoch_partial_success(self):
        anchor = self._anchor(relays=["wss://good", "wss://bad"])
        call_count = [0]

        async def maybe_fail(relay, event):
            call_count[0] += 1
            if "bad" in relay:
                raise RuntimeError("fail")

        async def _run():
            with patch.object(anchor, "_publish", side_effect=maybe_fail):
                return await anchor.anchor_epoch("ep-1", "0xabc")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert len(result.relays_ok) == 1
        assert len(result.relays_fail) == 1


# ---------------------------------------------------------------------------
# NostrAnchorResult
# ---------------------------------------------------------------------------

class TestNostrAnchorResult:
    def test_success_with_relays(self):
        r = NostrAnchorResult(
            epoch_id="ep", event_id="abc", relays_ok=["wss://r1"]
        )
        assert r.success is True

    def test_failure_no_relays(self):
        r = NostrAnchorResult(epoch_id="ep", event_id="abc", relays_ok=[])
        assert r.success is False

    def test_anchored_at_set(self):
        r = NostrAnchorResult(epoch_id="ep", event_id="id")
        assert r.anchored_at != ""

    def test_str_representation(self):
        r = NostrAnchorResult(epoch_id="ep", event_id="a" * 32,
                              relays_ok=["r1", "r2"], relays_fail=["r3"])
        s = str(r)
        assert "relays_ok=2" in s
        assert "relays_fail=1" in s
