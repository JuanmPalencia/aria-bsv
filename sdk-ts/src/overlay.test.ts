/**
 * Tests for aria-bsv/overlay — TopicManager, LookupService, OverlayClient.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  TopicManager,
  LookupService,
  OverlayClient,
  _setOverlayFetchImpl,
} from "./overlay.js";
import type { AdmittanceResult, LookupResult } from "./overlay.js";

// ---------------------------------------------------------------------------
// Setup: mock fetch
// ---------------------------------------------------------------------------

type FakeResponse = {
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
  text: () => Promise<string>;
};

function setFakeResponse(body: unknown, status = 200): void {
  _setOverlayFetchImpl(async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  }));
}

function setFakeError(status: number, message = "error"): void {
  _setOverlayFetchImpl(async () => ({
    ok: false,
    status,
    json: async () => ({ error: message }),
    text: async () => message,
  }));
}

const _BASE = "https://overlay.example.com";
const _RAW_TX = "01000000" + "00".repeat(100);

// ---------------------------------------------------------------------------
// TopicManager
// ---------------------------------------------------------------------------

describe("TopicManager", () => {
  it("default topic is tm_aria_epochs", () => {
    const tm = new TopicManager({ baseUrl: _BASE });
    expect(tm.topic).toBe("tm_aria_epochs");
  });

  it("custom topic is stored", () => {
    const tm = new TopicManager({ baseUrl: _BASE, topic: "tm_custom" });
    expect(tm.topic).toBe("tm_custom");
  });

  it("submit returns AdmittanceResult with admitted outputs", async () => {
    setFakeResponse({
      txid: "ab".repeat(32),
      topics: { tm_aria_epochs: { outputsToAdmit: [0] } },
    });
    const tm = new TopicManager({ baseUrl: _BASE });
    const result = await tm.submit(_RAW_TX);
    expect(result.txid).toBe("ab".repeat(32));
    expect(result.admitted).toBe(true);
    expect(result.admittedOutputs).toEqual([0]);
    expect(result.topic).toBe("tm_aria_epochs");
  });

  it("submit with no admitted outputs sets admitted=false", async () => {
    setFakeResponse({
      txid: "cd".repeat(32),
      topics: { tm_aria_epochs: { outputsToAdmit: [] } },
    });
    const tm = new TopicManager({ baseUrl: _BASE });
    const result = await tm.submit(_RAW_TX);
    expect(result.admitted).toBe(false);
    expect(result.admittedOutputs).toEqual([]);
    expect(result.message).toContain("No outputs admitted");
  });

  it("submit with missing topics key still returns result", async () => {
    setFakeResponse({ txid: "ef".repeat(32) });
    const tm = new TopicManager({ baseUrl: _BASE });
    const result = await tm.submit(_RAW_TX);
    expect(result.txid).toBe("ef".repeat(32));
    expect(result.admitted).toBe(false);
  });

  it("trailing slash in baseUrl is stripped", async () => {
    let capturedUrl = "";
    _setOverlayFetchImpl(async (url) => {
      capturedUrl = url;
      return { ok: true, status: 200, json: async () => ({ txid: "" }), text: async () => "" };
    });
    const tm = new TopicManager({ baseUrl: _BASE + "/" });
    await tm.submit(_RAW_TX);
    expect(capturedUrl).toBe(`${_BASE}/v1/submit`);
  });

  it("HTTP error throws", async () => {
    setFakeError(500, "Internal Server Error");
    const tm = new TopicManager({ baseUrl: _BASE });
    await expect(tm.submit(_RAW_TX)).rejects.toThrow("500");
  });

  it("apiKey is sent as Authorization Bearer", async () => {
    let capturedHeaders: Record<string, string> = {};
    _setOverlayFetchImpl(async (_url, init) => {
      capturedHeaders = init.headers;
      return { ok: true, status: 200, json: async () => ({ txid: "" }), text: async () => "" };
    });
    const tm = new TopicManager({ baseUrl: _BASE, apiKey: "secret-key" });
    await tm.submit(_RAW_TX);
    expect(capturedHeaders["Authorization"]).toBe("Bearer secret-key");
  });

  it("no apiKey → no Authorization header", async () => {
    let capturedHeaders: Record<string, string> = {};
    _setOverlayFetchImpl(async (_url, init) => {
      capturedHeaders = init.headers;
      return { ok: true, status: 200, json: async () => ({ txid: "" }), text: async () => "" };
    });
    const tm = new TopicManager({ baseUrl: _BASE });
    await tm.submit(_RAW_TX);
    expect(capturedHeaders["Authorization"]).toBeUndefined();
  });

  it("multiple admitted outputs", async () => {
    setFakeResponse({
      txid: "aa".repeat(32),
      topics: { tm_aria_epochs: { outputsToAdmit: [0, 1, 2] } },
    });
    const tm = new TopicManager({ baseUrl: _BASE });
    const result = await tm.submit(_RAW_TX);
    expect(result.admittedOutputs).toEqual([0, 1, 2]);
    expect(result.admitted).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// LookupService
// ---------------------------------------------------------------------------

describe("LookupService", () => {
  it("default service name is ls_aria", () => {
    const ls = new LookupService({ baseUrl: _BASE });
    expect(ls.serviceName).toBe("ls_aria");
  });

  it("custom service name stored", () => {
    const ls = new LookupService({ baseUrl: _BASE, serviceName: "ls_custom" });
    expect(ls.serviceName).toBe("ls_custom");
  });

  it("lookup returns array of LookupResults", async () => {
    setFakeResponse({
      results: [
        { txid: "ab".repeat(32), outputIndex: 0, beef: null, data: { type: "EPOCH_OPEN" } },
        { txid: "cd".repeat(32), outputIndex: 1, beef: "deadbeef", data: { type: "EPOCH_CLOSE" } },
      ],
    });
    const ls = new LookupService({ baseUrl: _BASE });
    const results = await ls.lookup({ system_id: "test" });
    expect(results).toHaveLength(2);
    expect(results[0].txid).toBe("ab".repeat(32));
    expect(results[0].outputIndex).toBe(0);
    expect(results[0].beef).toBeNull();
    expect(results[1].beef).toBe("deadbeef");
  });

  it("lookup with empty results returns []", async () => {
    setFakeResponse({ results: [] });
    const ls = new LookupService({ baseUrl: _BASE });
    const results = await ls.lookup({ system_id: "test" });
    expect(results).toEqual([]);
  });

  it("lookup with no results key returns []", async () => {
    setFakeResponse({});
    const ls = new LookupService({ baseUrl: _BASE });
    const results = await ls.lookup({});
    expect(results).toEqual([]);
  });

  it("HTTP error throws", async () => {
    setFakeError(404, "Not Found");
    const ls = new LookupService({ baseUrl: _BASE });
    await expect(ls.lookup({})).rejects.toThrow("404");
  });

  it("apiKey sent as Authorization Bearer", async () => {
    let capturedHeaders: Record<string, string> = {};
    _setOverlayFetchImpl(async (_url, init) => {
      capturedHeaders = init.headers;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const ls = new LookupService({ baseUrl: _BASE, apiKey: "my-key" });
    await ls.lookup({});
    expect(capturedHeaders["Authorization"]).toBe("Bearer my-key");
  });

  it("data field defaults to {} if missing", async () => {
    setFakeResponse({ results: [{ txid: "aa".repeat(32), outputIndex: 0 }] });
    const ls = new LookupService({ baseUrl: _BASE });
    const results = await ls.lookup({});
    expect(results[0].data).toEqual({});
  });

  it("beef defaults to null if missing", async () => {
    setFakeResponse({ results: [{ txid: "bb".repeat(32), outputIndex: 0, data: {} }] });
    const ls = new LookupService({ baseUrl: _BASE });
    const results = await ls.lookup({});
    expect(results[0].beef).toBeNull();
  });

  it("request body includes service name and limit", async () => {
    let capturedBody = "";
    _setOverlayFetchImpl(async (_url, init) => {
      capturedBody = init.body;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const ls = new LookupService({ baseUrl: _BASE, serviceName: "ls_test" });
    await ls.lookup({ system_id: "sys1" }, 25);
    const parsed = JSON.parse(capturedBody);
    expect(parsed.service).toBe("ls_test");
    expect(parsed.limit).toBe(25);
    expect(parsed.query.system_id).toBe("sys1");
  });
});

// ---------------------------------------------------------------------------
// OverlayClient
// ---------------------------------------------------------------------------

describe("OverlayClient", () => {
  it("exposes topicManager and lookupService", () => {
    const client = new OverlayClient({ baseUrl: _BASE });
    expect(client.topicManager).toBeDefined();
    expect(client.lookupService).toBeDefined();
  });

  it("topicManager has default epoch topic", () => {
    const client = new OverlayClient({ baseUrl: _BASE });
    expect(client.topicManager.topic).toBe("tm_aria_epochs");
  });

  it("custom epochTopic option flows through", () => {
    const client = new OverlayClient({ baseUrl: _BASE, epochTopic: "tm_custom" });
    expect(client.topicManager.topic).toBe("tm_custom");
  });

  it("lookupService has default service name", () => {
    const client = new OverlayClient({ baseUrl: _BASE });
    expect(client.lookupService.serviceName).toBe("ls_aria");
  });

  it("custom lookupService option flows through", () => {
    const client = new OverlayClient({ baseUrl: _BASE, lookupService: "ls_custom" });
    expect(client.lookupService.serviceName).toBe("ls_custom");
  });

  it("submitEpoch delegates to topicManager.submit", async () => {
    setFakeResponse({
      txid: "11".repeat(32),
      topics: { tm_aria_epochs: { outputsToAdmit: [0] } },
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    const result = await client.submitEpoch(_RAW_TX);
    expect(result.txid).toBe("11".repeat(32));
    expect(result.admitted).toBe(true);
  });

  it("findEpochs returns lookup results", async () => {
    setFakeResponse({
      results: [{ txid: "22".repeat(32), outputIndex: 0, data: { type: "EPOCH_OPEN" } }],
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    const results = await client.findEpochs("my-system");
    expect(results).toHaveLength(1);
    expect(results[0].data["type"]).toBe("EPOCH_OPEN");
  });

  it("findEpochs with epochId includes it in query", async () => {
    let capturedBody = "";
    _setOverlayFetchImpl(async (_url, init) => {
      capturedBody = init.body;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    await client.findEpochs("sys", "epoch-42");
    const parsed = JSON.parse(capturedBody);
    expect(parsed.query.epoch_id).toBe("epoch-42");
    expect(parsed.query.system_id).toBe("sys");
  });

  it("findEpochs without epochId omits epoch_id from query", async () => {
    let capturedBody = "";
    _setOverlayFetchImpl(async (_url, init) => {
      capturedBody = init.body;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    await client.findEpochs("sys");
    const parsed = JSON.parse(capturedBody);
    expect(parsed.query["epoch_id"]).toBeUndefined();
  });

  it("findRecords passes AUDIT_RECORD type", async () => {
    let capturedBody = "";
    _setOverlayFetchImpl(async (_url, init) => {
      capturedBody = init.body;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    await client.findRecords("sys", "epoch-1");
    const parsed = JSON.parse(capturedBody);
    expect(parsed.query.type).toBe("AUDIT_RECORD");
    expect(parsed.query.system_id).toBe("sys");
    expect(parsed.query.epoch_id).toBe("epoch-1");
  });

  it("findRecords default limit is 100", async () => {
    let capturedBody = "";
    _setOverlayFetchImpl(async (_url, init) => {
      capturedBody = init.body;
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    const client = new OverlayClient({ baseUrl: _BASE });
    await client.findRecords("sys", "epoch-1");
    expect(JSON.parse(capturedBody).limit).toBe(100);
  });

  it("apiKey propagates to both topicManager and lookupService", async () => {
    const headers: string[] = [];
    _setOverlayFetchImpl(async (_url, init) => {
      if (init.headers["Authorization"]) headers.push(init.headers["Authorization"]);
      return { ok: true, status: 200, json: async () => ({ txid: "", results: [], topics: {} }), text: async () => "" };
    });
    const client = new OverlayClient({ baseUrl: _BASE, apiKey: "shared-key" });
    await client.submitEpoch(_RAW_TX);
    await client.findEpochs("sys");
    expect(headers.every((h) => h === "Bearer shared-key")).toBe(true);
    expect(headers).toHaveLength(2);
  });
});
