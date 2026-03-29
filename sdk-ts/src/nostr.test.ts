/**
 * nostr.test.ts — NostrAnchorer and buildNostrContent unit tests.
 *
 * All WebSocket interactions are intercepted via _setNostrWebSocketImpl so no
 * real relay connection is needed.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  buildNostrContent,
  NostrAnchorer,
  _setNostrWebSocketImpl,
} from "./nostr.js";
import type {
  NostrEvent,
  NostrAnchorPayload,
  MockWebSocket,
} from "./nostr.js";

// ---------------------------------------------------------------------------
// MockWS — controllable WebSocket stub
// ---------------------------------------------------------------------------

class MockWS implements MockWebSocket {
  readonly url: string;
  sent: string[] = [];

  onopen:    ((ev: Event) => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose:   ((ev: Event) => void) | null = null;
  onerror:   ((ev: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    // Fire onopen on the next microtask so callers can set the handler first
    Promise.resolve().then(() => {
      if (this.onopen) this.onopen(new Event("open"));
    });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void { /* no-op */ }

  /** Simulate a message arriving from the relay. */
  receive(data: string): void {
    if (this.onmessage) this.onmessage({ data });
  }

  /** Simulate a WebSocket error. */
  triggerError(): void {
    if (this.onerror) this.onerror(new Event("error"));
  }

  /** Simulate the relay closing the connection. */
  triggerClose(): void {
    if (this.onclose) this.onclose(new Event("close"));
  }
}

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

const SAMPLE_EVENT: NostrEvent = {
  id:         "a".repeat(64),
  pubkey:     "b".repeat(64),
  created_at: 1700000000,
  kind:       1,
  tags:       [],
  content:    '{"epochId":"epoch-1"}',
  sig:        "c".repeat(128),
};

const SAMPLE_PAYLOAD: NostrAnchorPayload = {
  epochId:       "epoch-1",
  merkleRoot:    "d".repeat(64),
  recordsCount:  10,
  systemId:      "my-system",
  brc121Version: "1.0",
};

// ---------------------------------------------------------------------------
// buildNostrContent
// ---------------------------------------------------------------------------

describe("buildNostrContent", () => {
  it("serialises payload to JSON", () => {
    const content = buildNostrContent(SAMPLE_PAYLOAD);
    expect(JSON.parse(content)).toEqual(SAMPLE_PAYLOAD);
  });

  it("includes all required fields", () => {
    const parsed = JSON.parse(buildNostrContent(SAMPLE_PAYLOAD)) as NostrAnchorPayload;
    expect(parsed.epochId).toBe("epoch-1");
    expect(parsed.merkleRoot).toBe("d".repeat(64));
    expect(parsed.recordsCount).toBe(10);
    expect(parsed.systemId).toBe("my-system");
    expect(parsed.brc121Version).toBe("1.0");
  });

  it("returns a string", () => {
    expect(typeof buildNostrContent(SAMPLE_PAYLOAD)).toBe("string");
  });

  it("is valid JSON (parseable without throw)", () => {
    expect(() => JSON.parse(buildNostrContent(SAMPLE_PAYLOAD))).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// NostrAnchorer — constructor
// ---------------------------------------------------------------------------

describe("NostrAnchorer — constructor", () => {
  it("stores relayUrl and optional pubkey", () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test", pubkey: "mypub" });
    expect(anchorer.pubkey).toBe("mypub");
  });

  it("defaults pubkey to empty string when omitted", () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    expect(anchorer.pubkey).toBe("");
  });
});

// ---------------------------------------------------------------------------
// NostrAnchorer — publish
// ---------------------------------------------------------------------------

describe("NostrAnchorer — publish", () => {
  let ws: MockWS;

  beforeEach(() => {
    ws = new MockWS("wss://relay.test");
    _setNostrWebSocketImpl(() => ws);
  });

  it("returns ok:true when relay responds OK true", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve(); // let onopen fire
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, ""]));

    const result = await p;
    expect(result.ok).toBe(true);
  });

  it("returns ok:false when relay responds OK false", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, false, "duplicate event"]));

    const result = await p;
    expect(result.ok).toBe(false);
    expect(result.message).toBe("duplicate event");
  });

  it("sends an EVENT message after connect", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, ""]));
    await p;

    expect(ws.sent).toHaveLength(1);
    const msg = JSON.parse(ws.sent[0]) as [string, NostrEvent];
    expect(msg[0]).toBe("EVENT");
    expect(msg[1].id).toBe(SAMPLE_EVENT.id);
  });

  it("returns ok:false on WebSocket error", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    ws.triggerError();

    const result = await p;
    expect(result.ok).toBe(false);
    expect(result.message).toContain("error");
  });

  it("returns ok:false when connection closes before OK", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    ws.triggerClose();

    const result = await p;
    expect(result.ok).toBe(false);
    expect(result.message).toContain("closed");
  });

  it("ignores OK messages for different event IDs", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    // Wrong event ID — should be ignored
    ws.receive(JSON.stringify(["OK", "unrelated-id", true, ""]));
    // Correct event ID — should settle
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, "accepted"]));

    const result = await p;
    expect(result.ok).toBe(true);
    expect(result.message).toBe("accepted");
  });

  it("ignores non-JSON messages without throwing", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT);

    await Promise.resolve();
    ws.receive("not json {{{{");
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, ""]));

    const result = await p;
    expect(result.ok).toBe(true);
  });

  it("settles only once even if OK is received twice", async () => {
    let resolveCount = 0;
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.publish(SAMPLE_EVENT).then((r) => { resolveCount++; return r; });

    await Promise.resolve();
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, ""]));
    ws.receive(JSON.stringify(["OK", SAMPLE_EVENT.id, true, "again"]));
    await p;

    expect(resolveCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// NostrAnchorer — queryEpochs
// ---------------------------------------------------------------------------

describe("NostrAnchorer — queryEpochs", () => {
  let ws: MockWS;

  beforeEach(() => {
    ws = new MockWS("wss://relay.test");
    _setNostrWebSocketImpl(() => ws);
  });

  it("sends a REQ message after connect", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    const req = JSON.parse(ws.sent[0]) as [string, string, unknown];
    expect(req[0]).toBe("REQ");
    const subId = req[1];
    ws.receive(JSON.stringify(["EOSE", subId]));
    await p;
  });

  it("collects EVENT messages until EOSE", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    const [, subId] = JSON.parse(ws.sent[0]) as [string, string];

    ws.receive(JSON.stringify(["EVENT", subId, { ...SAMPLE_EVENT, id: "ev1" }]));
    ws.receive(JSON.stringify(["EVENT", subId, { ...SAMPLE_EVENT, id: "ev2" }]));
    ws.receive(JSON.stringify(["EOSE", subId]));

    const events = await p;
    expect(events).toHaveLength(2);
    expect(events[0].id).toBe("ev1");
    expect(events[1].id).toBe("ev2");
  });

  it("returns empty array when EOSE arrives immediately", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    const [, subId] = JSON.parse(ws.sent[0]) as [string, string];
    ws.receive(JSON.stringify(["EOSE", subId]));

    const events = await p;
    expect(events).toHaveLength(0);
  });

  it("ignores EVENT messages for other subscription IDs", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    const [, subId] = JSON.parse(ws.sent[0]) as [string, string];
    ws.receive(JSON.stringify(["EVENT", "other-sub", SAMPLE_EVENT]));
    ws.receive(JSON.stringify(["EOSE", subId]));

    const events = await p;
    expect(events).toHaveLength(0);
  });

  it("returns empty array on WebSocket error", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    ws.triggerError();

    const events = await p;
    expect(events).toHaveLength(0);
  });

  it("returns collected events when connection closes before EOSE", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({ kinds: [1] });

    await Promise.resolve();
    const [, subId] = JSON.parse(ws.sent[0]) as [string, string];
    ws.receive(JSON.stringify(["EVENT", subId, SAMPLE_EVENT]));
    ws.triggerClose();

    const events = await p;
    expect(events).toHaveLength(1);
  });

  it("includes filter in REQ message", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const filter = { kinds: [30078], since: 1700000000 };
    const p = anchorer.queryEpochs(filter);

    await Promise.resolve();
    const [, subId, sentFilter] = JSON.parse(ws.sent[0]) as [string, string, typeof filter];
    ws.receive(JSON.stringify(["EOSE", subId]));
    await p;

    expect(sentFilter.kinds).toEqual([30078]);
    expect(sentFilter.since).toBe(1700000000);
  });

  it("ignores non-JSON relay messages without throwing", async () => {
    const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
    const p = anchorer.queryEpochs({});

    await Promise.resolve();
    const [, subId] = JSON.parse(ws.sent[0]) as [string, string];
    ws.receive("garbage data !!!");
    ws.receive(JSON.stringify(["EOSE", subId]));

    const events = await p;
    expect(events).toHaveLength(0);
  });

  it("generates unique subscription IDs across calls", async () => {
    const subIds: string[] = [];

    for (let i = 0; i < 3; i++) {
      ws = new MockWS("wss://relay.test");
      _setNostrWebSocketImpl(() => ws);

      const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.test" });
      const p = anchorer.queryEpochs({});
      await Promise.resolve();
      const [, subId] = JSON.parse(ws.sent[0]) as [string, string];
      subIds.push(subId);
      ws.receive(JSON.stringify(["EOSE", subId]));
      await p;
    }

    // All three should be different (with overwhelming probability)
    expect(new Set(subIds).size).toBe(3);
  });
});
