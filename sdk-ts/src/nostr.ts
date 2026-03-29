/**
 * aria-bsv/nostr — Nostr relay anchor for ARIA epoch commitments.
 *
 * Protocol: NIP-01 WebSocket messages.
 * Publishes ARIA epoch data as Nostr events and queries relay history.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A Nostr event (NIP-01). */
export interface NostrEvent {
  id: string;
  pubkey: string;
  created_at: number;
  kind: number;
  tags: string[][];
  content: string;
  sig: string;
}

/** ARIA-specific data encoded in a Nostr event's content field. */
export interface NostrAnchorPayload {
  epochId: string;
  merkleRoot: string;
  recordsCount: number;
  systemId: string;
  brc121Version: string;
}

/**
 * Minimal WebSocket interface used by {@link NostrAnchorer}.
 * Compatible with the browser `WebSocket` API and the `ws` npm package.
 */
export interface MockWebSocket {
  send(data: string): void;
  close(): void;
  onopen: ((ev: Event) => void) | null;
  onmessage: ((ev: { data: string }) => void) | null;
  onclose: ((ev: Event) => void) | null;
  onerror: ((ev: Event) => void) | null;
}

/** Factory that creates a {@link MockWebSocket}-compatible connection. */
export type WebSocketFactory = (url: string) => MockWebSocket;

// ---------------------------------------------------------------------------
// buildNostrContent
// ---------------------------------------------------------------------------

/**
 * Serialise a {@link NostrAnchorPayload} to JSON suitable for embedding in a
 * Nostr event's `content` field.
 */
export function buildNostrContent(payload: NostrAnchorPayload): string {
  return JSON.stringify(payload);
}

// ---------------------------------------------------------------------------
// NostrAnchorer
// ---------------------------------------------------------------------------

/**
 * Anchor ARIA epoch commitments to Nostr relays via NIP-01 WebSocket messages.
 *
 * @example
 * ```ts
 * const anchorer = new NostrAnchorer({ relayUrl: "wss://relay.damus.io" });
 * const result = await anchorer.publish(event);
 * if (!result.ok) console.error(result.message);
 * ```
 */
export class NostrAnchorer {
  private readonly _relayUrl: string;
  readonly pubkey: string;

  constructor(options: { relayUrl: string; pubkey?: string }) {
    this._relayUrl = options.relayUrl;
    this.pubkey = options.pubkey ?? "";
  }

  /**
   * Publish a Nostr event to the relay.
   *
   * Sends `["EVENT", event]` and waits for `["OK", id, true/false, msg]`.
   *
   * @returns `{ ok, message }` reflecting the relay's acceptance decision.
   */
  async publish(event: NostrEvent): Promise<{ ok: boolean; message: string }> {
    return new Promise((resolve) => {
      const ws = _wsFactory(this._relayUrl);
      let settled = false;

      const settle = (result: { ok: boolean; message: string }): void => {
        if (!settled) {
          settled = true;
          try { ws.close(); } catch { /* ignore */ }
          resolve(result);
        }
      };

      ws.onopen = () => {
        try {
          ws.send(JSON.stringify(["EVENT", event]));
        } catch (err) {
          settle({
            ok: false,
            message: err instanceof Error ? err.message : "Send failed",
          });
        }
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as unknown[];
          if (Array.isArray(msg) && msg[0] === "OK" && msg[1] === event.id) {
            settle({ ok: msg[2] as boolean, message: (msg[3] as string) ?? "" });
          }
        } catch {
          // ignore non-JSON or unrelated messages
        }
      };

      ws.onerror = () => {
        settle({ ok: false, message: "WebSocket error" });
      };

      ws.onclose = () => {
        settle({ ok: false, message: "Connection closed without OK response" });
      };
    });
  }

  /**
   * Query stored events from the relay using a NIP-01 subscription.
   *
   * Sends `["REQ", subId, filter]`, collects `EVENT` messages until `EOSE`.
   *
   * @param filter NIP-01 filter object.
   * @returns Array of {@link NostrEvent} objects returned by the relay.
   */
  async queryEpochs(filter: {
    kinds?: number[];
    "#e"?: string[];
    since?: number;
  }): Promise<NostrEvent[]> {
    return new Promise((resolve) => {
      const ws = _wsFactory(this._relayUrl);
      const subId = _randomSubId();
      const events: NostrEvent[] = [];
      let settled = false;

      const settle = (result: NostrEvent[]): void => {
        if (!settled) {
          settled = true;
          try { ws.close(); } catch { /* ignore */ }
          resolve(result);
        }
      };

      ws.onopen = () => {
        try {
          ws.send(JSON.stringify(["REQ", subId, filter]));
        } catch {
          settle([]);
        }
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as unknown[];
          if (!Array.isArray(msg)) return;
          if (msg[0] === "EVENT" && msg[1] === subId) {
            events.push(msg[2] as NostrEvent);
          } else if (msg[0] === "EOSE" && msg[1] === subId) {
            settle(events);
          }
        } catch {
          // ignore
        }
      };

      ws.onerror = () => {
        settle([]);
      };

      ws.onclose = () => {
        settle(events);
      };
    });
  }
}

// ---------------------------------------------------------------------------
// WebSocket abstraction (replaceable in tests)
// ---------------------------------------------------------------------------

let _wsFactory: WebSocketFactory = (url: string): MockWebSocket => {
  // Browser / Node 22+ native WebSocket
  if (typeof WebSocket !== "undefined") {
    return new WebSocket(url) as unknown as MockWebSocket;
  }
  // Node.js — attempt dynamic require of the 'ws' package
  if (typeof require !== "undefined") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { WebSocket: WS } = require("ws") as {
        WebSocket: new (url: string) => MockWebSocket;
      };
      return new WS(url);
    } catch {
      // ws not installed
    }
  }
  throw new Error(
    "No WebSocket implementation available. Use Node 22+ or install the 'ws' package."
  );
};

/** Replace the WebSocket factory (useful for testing). */
export function _setNostrWebSocketImpl(fn: WebSocketFactory): void {
  _wsFactory = fn;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _randomSubId(): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  return Array.from(
    { length: 16 },
    () => chars[Math.floor(Math.random() * chars.length)]
  ).join("");
}
