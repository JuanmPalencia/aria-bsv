/**
 * broadcaster.test.ts — ARCBroadcaster unit tests.
 *
 * All HTTP calls are intercepted via _setFetchImpl so no network is needed.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { ARCBroadcaster, _setFetchImpl } from "./broadcaster.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type MockResponse = {
  ok: boolean;
  status: number;
  body?: unknown;
  text?: string;
};

function makeFetch(responses: MockResponse[]) {
  let call = 0;
  const requests: Array<{ url: string; init: unknown }> = [];

  const fn = vi.fn(async (url: string, init: unknown) => {
    requests.push({ url, init });
    const resp = responses[Math.min(call++, responses.length - 1)];
    return {
      ok: resp.ok,
      status: resp.status,
      json: async () => resp.body ?? {},
      text: async () => resp.text ?? "",
    };
  });

  return { fn, requests, callCount: () => call };
}

function okResponse(txid = "abc123", txStatus = "OK"): MockResponse {
  return { ok: true, status: 200, body: { txid, txStatus } };
}

function errResponse(status: number, text = "bad"): MockResponse {
  return { ok: false, status, text };
}

// ---------------------------------------------------------------------------
// Basic broadcast
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — successful broadcast", () => {
  beforeEach(() => {
    // Reset to a no-op so leaking is safe
    _setFetchImpl(async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => "" }));
  });

  it("returns propagated:true with txid on 200", async () => {
    const { fn } = makeFetch([okResponse("deadbeef", "SEEN_ON_NETWORK")]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 0 });
    const result = await bc.broadcast("rawtx_hex");

    expect(result.propagated).toBe(true);
    expect(result.txid).toBe("deadbeef");
    expect(result.message).toBe("SEEN_ON_NETWORK");
  });

  it("makes exactly one HTTP call on success", async () => {
    const { fn, callCount } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 3 });
    await bc.broadcast("00aabb");

    expect(callCount()).toBe(1);
  });

  it("sends rawTx in JSON body", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 0 });
    await bc.broadcast("mycustomtx");

    const body = JSON.parse((requests[0].init as { body: string }).body);
    expect(body).toEqual({ rawTx: "mycustomtx" });
  });

  it("uses POST method", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect((requests[0].init as { method: string }).method).toBe("POST");
  });
});

// ---------------------------------------------------------------------------
// URL configuration
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — URL configuration", () => {
  it("defaults to arc.taal.com/v1/tx", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect(requests[0].url).toBe("https://arc.taal.com/v1/tx");
  });

  it("uses custom apiUrl", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ apiUrl: "https://custom.arc.io", maxRetries: 0 }).broadcast("tx");

    expect(requests[0].url).toBe("https://custom.arc.io/v1/tx");
  });

  it("strips trailing slash from apiUrl", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ apiUrl: "https://custom.arc.io/", maxRetries: 0 }).broadcast("tx");

    expect(requests[0].url).toBe("https://custom.arc.io/v1/tx");
  });
});

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — headers", () => {
  it("sets Content-Type: application/json", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    const headers = (requests[0].init as { headers: Record<string, string> }).headers;
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("sets Accept: application/json", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    const headers = (requests[0].init as { headers: Record<string, string> }).headers;
    expect(headers["Accept"]).toBe("application/json");
  });

  it("includes Authorization Bearer header when apiKey provided", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ apiKey: "mainnet_mytoken", maxRetries: 0 }).broadcast("tx");

    const headers = (requests[0].init as { headers: Record<string, string> }).headers;
    expect(headers["Authorization"]).toBe("Bearer mainnet_mytoken");
  });

  it("omits Authorization header when no apiKey", async () => {
    const { fn, requests } = makeFetch([okResponse()]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    const headers = (requests[0].init as { headers: Record<string, string> }).headers;
    expect(headers["Authorization"]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// 4xx — non-retryable
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — 4xx errors (non-retryable)", () => {
  it("returns propagated:false on 400 without retrying", async () => {
    const { fn, callCount } = makeFetch([errResponse(400, "Invalid tx")]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 3, baseDelayMs: 0 });
    const result = await bc.broadcast("badtx");

    expect(result.propagated).toBe(false);
    expect(callCount()).toBe(1);
    expect(result.message).toContain("400");
  });

  it("returns propagated:false on 401", async () => {
    const { fn, callCount } = makeFetch([errResponse(401, "Unauthorized")]);
    _setFetchImpl(fn);

    const result = await new ARCBroadcaster({ maxRetries: 3, baseDelayMs: 0 }).broadcast("tx");

    expect(result.propagated).toBe(false);
    expect(callCount()).toBe(1);
  });

  it("returns propagated:false on 422", async () => {
    const { fn, callCount } = makeFetch([errResponse(422, "Unprocessable")]);
    _setFetchImpl(fn);

    const result = await new ARCBroadcaster({ maxRetries: 2, baseDelayMs: 0 }).broadcast("tx");

    expect(result.propagated).toBe(false);
    expect(callCount()).toBe(1);
  });

  it("includes server error text in message on 4xx", async () => {
    const { fn } = makeFetch([errResponse(400, "double spend")]);
    _setFetchImpl(fn);

    const result = await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect(result.message).toContain("double spend");
  });
});

// ---------------------------------------------------------------------------
// 5xx — retryable
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — 5xx errors (retryable)", () => {
  it("retries on 503 and succeeds on second attempt", async () => {
    const { fn, callCount } = makeFetch([errResponse(503), okResponse("retry_txid")]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 2, baseDelayMs: 0 });
    const result = await bc.broadcast("tx");

    expect(result.propagated).toBe(true);
    expect(result.txid).toBe("retry_txid");
    expect(callCount()).toBe(2);
  });

  it("returns propagated:false when all retries exhausted on 5xx", async () => {
    const { fn, callCount } = makeFetch([
      errResponse(500),
      errResponse(500),
      errResponse(500),
    ]);
    _setFetchImpl(fn);

    const bc = new ARCBroadcaster({ maxRetries: 2, baseDelayMs: 0 });
    const result = await bc.broadcast("tx");

    expect(result.propagated).toBe(false);
    expect(callCount()).toBe(3); // 1 initial + 2 retries
  });

  it("with maxRetries:0 makes exactly 1 attempt on 5xx", async () => {
    const { fn, callCount } = makeFetch([errResponse(500)]);
    _setFetchImpl(fn);

    await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect(callCount()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Network errors (fetch throws)
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — network errors", () => {
  it("returns propagated:false when fetch throws", async () => {
    _setFetchImpl(async () => { throw new Error("ECONNREFUSED"); });

    const bc = new ARCBroadcaster({ maxRetries: 0 });
    const result = await bc.broadcast("tx");

    expect(result.propagated).toBe(false);
    expect(result.txid).toBe("");
    expect(result.message).toContain("ECONNREFUSED");
  });

  it("retries on network error and returns propagated:true on recovery", async () => {
    let calls = 0;
    _setFetchImpl(async () => {
      if (++calls === 1) throw new Error("timeout");
      return { ok: true, status: 200, json: async () => ({ txid: "recovered", txStatus: "OK" }), text: async () => "" };
    });

    const bc = new ARCBroadcaster({ maxRetries: 1, baseDelayMs: 0 });
    const result = await bc.broadcast("tx");

    expect(result.propagated).toBe(true);
    expect(result.txid).toBe("recovered");
  });

  it("returns last error message after max retries on network error", async () => {
    _setFetchImpl(async () => { throw new Error("DNS lookup failed"); });

    const bc = new ARCBroadcaster({ maxRetries: 1, baseDelayMs: 0 });
    const result = await bc.broadcast("tx");

    expect(result.propagated).toBe(false);
    expect(result.message).toContain("DNS lookup failed");
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("ARCBroadcaster — edge cases", () => {
  it("handles missing txid in response (uses empty string)", async () => {
    _setFetchImpl(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ txStatus: "STORED" }), // no txid
      text: async () => "",
    }));

    const result = await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect(result.propagated).toBe(true);
    expect(result.txid).toBe("");
  });

  it("handles missing txStatus in response (uses OK)", async () => {
    _setFetchImpl(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ txid: "abc" }), // no txStatus
      text: async () => "",
    }));

    const result = await new ARCBroadcaster({ maxRetries: 0 }).broadcast("tx");

    expect(result.message).toBe("OK");
  });

  it("can broadcast an empty string (edge case)", async () => {
    const { fn, requests } = makeFetch([okResponse("empty_tx_id")]);
    _setFetchImpl(fn);

    const result = await new ARCBroadcaster({ maxRetries: 0 }).broadcast("");

    expect(result.propagated).toBe(true);
    const body = JSON.parse((requests[0].init as { body: string }).body);
    expect(body.rawTx).toBe("");
  });
});
