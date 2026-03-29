/**
 * Tests for aria-bsv/auditor — InferenceAuditor record / flush / close.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { InferenceAuditor } from "./auditor.js";
import { _setFetchImpl } from "./broadcaster.js";
import type { AuditConfig } from "./types.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function cfg(overrides: Partial<AuditConfig> = {}): AuditConfig {
  return {
    system_id: "test-system",
    batch_ms: 0,       // disable auto-flush timer
    batch_size: 50,
    max_retries: 0,    // no retry delays in tests
    base_delay_ms: 0,
    ...overrides,
  };
}

/** Make the broadcaster always succeed with a fake txid. */
function mockBroadcaster(txid = "cafecafe".repeat(8)): void {
  _setFetchImpl(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ txid, txStatus: "MINED" }),
    text: async () => "",
  }));
}

/** Make the broadcaster always fail (network error). */
function mockBroadcasterFail(): void {
  _setFetchImpl(async () => {
    throw new Error("network error");
  });
}

beforeEach(() => {
  mockBroadcaster();
});

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe("InferenceAuditor constructor", () => {
  it("creates an auditor with no pending records", () => {
    const a = new InferenceAuditor(cfg());
    expect(a.pendingRecords).toHaveLength(0);
  });

  it("epochId is empty before open()", () => {
    const a = new InferenceAuditor(cfg());
    expect(a.epochId).toBe("");
  });
});

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------

describe("InferenceAuditor.open()", () => {
  it("sets epochId", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    expect(a.epochId).not.toBe("");
  });

  it("returns the epoch ID", async () => {
    const a = new InferenceAuditor(cfg());
    const id = await a.open();
    expect(id).toBe(a.epochId);
  });

  it("epoch ID contains timestamp portion", async () => {
    const a = new InferenceAuditor(cfg());
    const id = await a.open();
    expect(id).toMatch(/^\d+_[0-9a-f]+$/);
  });
});

// ---------------------------------------------------------------------------
// record()
// ---------------------------------------------------------------------------

describe("InferenceAuditor.record()", () => {
  it("returns an AuditRecord with correct fields", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const rec = await a.record("gpt-4o", { prompt: "hi" }, { text: "hello" });

    expect(rec.model_id).toBe("gpt-4o");
    expect(rec.system_id).toBe("test-system");
    expect(rec.input_hash).toHaveLength(64);
    expect(rec.output_hash).toHaveLength(64);
    expect(rec.sequence).toBe(0);
    expect(rec.epoch_id).toBe(a.epochId);
    expect(rec.confidence).toBeNull();
    expect(rec.latency_ms).toBeGreaterThanOrEqual(0);
  });

  it("sequence increments per record", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const r1 = await a.record("m", "in1", "out1");
    const r2 = await a.record("m", "in2", "out2");
    const r3 = await a.record("m", "in3", "out3");
    expect(r1.sequence).toBe(0);
    expect(r2.sequence).toBe(1);
    expect(r3.sequence).toBe(2);
  });

  it("record_id matches expected format", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const rec = await a.record("m", "i", "o");
    expect(rec.record_id).toMatch(/^rec_\S+_\d{6}$/);
  });

  it("stores confidence when provided", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const rec = await a.record("m", "i", "o", { confidence: 0.95 });
    expect(rec.confidence).toBe(0.95);
  });

  it("stores metadata when provided", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const rec = await a.record("m", "i", "o", { metadata: { user: "alice" } });
    expect(rec.metadata).toEqual({ user: "alice" });
  });

  it("adds record to pendingRecords", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    expect(a.pendingRecords).toHaveLength(1);
  });

  it("auto-opens epoch if not open", async () => {
    const a = new InferenceAuditor(cfg());
    await a.record("m", "i", "o");
    expect(a.epochId).not.toBe("");
  });

  it("different input data → different input_hash", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const r1 = await a.record("m", { prompt: "A" }, "out");
    const r2 = await a.record("m", { prompt: "B" }, "out");
    expect(r1.input_hash).not.toBe(r2.input_hash);
  });

  it("same input data → same input_hash", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const r1 = await a.record("m", { prompt: "same" }, "out1");
    const r2 = await a.record("m", { prompt: "same" }, "out2");
    expect(r1.input_hash).toBe(r2.input_hash);
  });

  it("auto-flush when batch_size reached", async () => {
    const a = new InferenceAuditor(cfg({ batch_size: 2 }));
    await a.open();
    await a.record("m", "i1", "o1");
    await a.record("m", "i2", "o2"); // should trigger flush
    expect(a.pendingRecords).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------

describe("InferenceAuditor.flush()", () => {
  it("returns empty string when no records", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    const txid = await a.flush();
    expect(txid).toBe("");
  });

  it("returns a txid string after recording", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    const txid = await a.flush();
    expect(txid).toBeTruthy();
    expect(typeof txid).toBe("string");
  });

  it("clears pending records after flush", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    await a.flush();
    expect(a.pendingRecords).toHaveLength(0);
  });

  it("resets epochId after flush", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    await a.flush();
    expect(a.epochId).toBe("");
  });

  it("is idempotent — second flush with no records returns ''", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    await a.flush();
    const second = await a.flush();
    expect(second).toBe("");
  });
});

// ---------------------------------------------------------------------------
// Broadcaster failure handling
// ---------------------------------------------------------------------------

describe("InferenceAuditor with broadcaster failure", () => {
  it("still returns a stub txid on network error", async () => {
    mockBroadcasterFail();
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    const txid = await a.flush();
    // A non-empty stub txid is produced so the protocol chain is maintained
    expect(txid).toBeTruthy();
    expect(typeof txid).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// close()
// ---------------------------------------------------------------------------

describe("InferenceAuditor.close()", () => {
  it("flushes pending records on close", async () => {
    const a = new InferenceAuditor(cfg());
    await a.open();
    await a.record("m", "i", "o");
    await a.close();
    expect(a.pendingRecords).toHaveLength(0);
  });
});
