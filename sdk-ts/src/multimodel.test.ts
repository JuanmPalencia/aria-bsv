/**
 * multimodel.test.ts — MultiModelAuditor unit tests.
 */

import { describe, it, expect } from "vitest";
import { MultiModelAuditor } from "./multimodel.js";
import type { ModelEntry, MultiModelEpochOpen } from "./multimodel.js";
import { EMPTY_ROOT } from "./merkle.js";

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

const MODEL_A: ModelEntry = {
  modelId:   "gpt-4o",
  modelHash: "a".repeat(64),
};
const MODEL_B: ModelEntry = {
  modelId:   "claude-3",
  modelHash: "b".repeat(64),
};
const MODEL_C: ModelEntry = {
  modelId:   "gemini-pro",
  modelHash: "c".repeat(64),
};

function makeAuditor(models = [MODEL_A, MODEL_B]) {
  return new MultiModelAuditor({ systemId: "test-system", models });
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe("MultiModelAuditor — constructor", () => {
  it("stores all provided models in stats().models", () => {
    const a = makeAuditor([MODEL_A, MODEL_B, MODEL_C]);
    const { models } = a.stats();
    expect(models).toContain("gpt-4o");
    expect(models).toContain("claude-3");
    expect(models).toContain("gemini-pro");
    expect(models).toHaveLength(3);
  });

  it("starts with no buffered records", () => {
    expect(makeAuditor().stats().buffered).toBe(0);
  });

  it("starts with an empty epochId", () => {
    expect(makeAuditor().stats().epochId).toBe("");
  });
});

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------

describe("MultiModelAuditor — open()", () => {
  it("returns a MultiModelEpochOpen with type EPOCH_OPEN", async () => {
    const payload = await makeAuditor().open();
    expect(payload.type).toBe("EPOCH_OPEN");
  });

  it("sets system_id correctly", async () => {
    const payload = await makeAuditor().open();
    expect(payload.system_id).toBe("test-system");
  });

  it("generates a non-empty epoch_id", async () => {
    const payload = await makeAuditor().open();
    expect(payload.epoch_id.length).toBeGreaterThan(0);
  });

  it("sets brc121_version to 1.0", async () => {
    const payload = await makeAuditor().open();
    expect(payload.brc121_version).toBe("1.0");
  });

  it("includes model_hashes map with all models", async () => {
    const payload = await makeAuditor().open();
    expect(payload.model_hashes["gpt-4o"]).toBe(MODEL_A.modelHash);
    expect(payload.model_hashes["claude-3"]).toBe(MODEL_B.modelHash);
  });

  it("includes models array", async () => {
    const payload = await makeAuditor().open();
    expect(payload.models).toHaveLength(2);
    expect(payload.models[0].modelId).toBe("gpt-4o");
  });

  it("computes a multiModelRoot string of 64 hex chars", async () => {
    const payload = await makeAuditor().open();
    expect(payload.multiModelRoot).toMatch(/^[0-9a-f]{64}$/);
  });

  it("multiModelRoot is deterministic for the same models", async () => {
    const a = makeAuditor();
    const b = makeAuditor();
    const pa = await a.open();
    const pb = await b.open();
    expect(pa.multiModelRoot).toBe(pb.multiModelRoot);
  });

  it("multiModelRoot differs when models differ", async () => {
    const a = new MultiModelAuditor({ systemId: "s", models: [MODEL_A] });
    const b = new MultiModelAuditor({ systemId: "s", models: [MODEL_B] });
    const pa = await a.open();
    const pb = await b.open();
    expect(pa.multiModelRoot).not.toBe(pb.multiModelRoot);
  });

  it("sets nonce as 32-char hex string", async () => {
    const payload = await makeAuditor().open();
    expect(payload.nonce).toMatch(/^[0-9a-f]{32}$/);
  });

  it("sets opened_at to a recent ISO-8601 timestamp", async () => {
    const before = new Date().toISOString();
    const payload = await makeAuditor().open();
    const after  = new Date().toISOString();
    expect(payload.opened_at >= before).toBe(true);
    expect(payload.opened_at <= after).toBe(true);
  });

  it("calling open() again resets epoch state", async () => {
    const a = makeAuditor();
    const p1 = await a.open();
    await a.record("gpt-4o", "in", "out");
    const p2 = await a.open();
    expect(p2.epoch_id).not.toBe(p1.epoch_id);
    expect(a.stats().buffered).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// record()
// ---------------------------------------------------------------------------

describe("MultiModelAuditor — record()", () => {
  it("returns an AuditRecord with correct structure", async () => {
    const a = makeAuditor();
    await a.open();
    const rec = await a.record("gpt-4o", { q: "hello" }, { a: "world" });

    expect(rec.model_id).toBe("gpt-4o");
    expect(rec.system_id).toBe("test-system");
    expect(rec.input_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(rec.output_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(rec.sequence).toBe(0);
  });

  it("increments sequence for consecutive records", async () => {
    const a = makeAuditor();
    await a.open();
    const r1 = await a.record("gpt-4o", "a", "b");
    const r2 = await a.record("claude-3", "c", "d");
    expect(r1.sequence).toBe(0);
    expect(r2.sequence).toBe(1);
  });

  it("auto-opens an epoch if none is open", async () => {
    const a = makeAuditor();
    // Do NOT call open() first
    const rec = await a.record("gpt-4o", "in", "out");
    expect(rec.epoch_id.length).toBeGreaterThan(0);
  });

  it("throws for an unknown modelId", async () => {
    const a = makeAuditor();
    await a.open();
    await expect(a.record("unknown-model", "x", "y")).rejects.toThrow(
      /unknown modelid/i
    );
  });

  it("error message lists registered models", async () => {
    const a = makeAuditor();
    await a.open();
    await expect(a.record("badmodel", "x", "y")).rejects.toThrow(/gpt-4o/);
  });

  it("stores confidence when provided", async () => {
    const a = makeAuditor();
    await a.open();
    const rec = await a.record("gpt-4o", "in", "out", { confidence: 0.95 });
    expect(rec.confidence).toBe(0.95);
  });

  it("sets confidence to null when not provided", async () => {
    const a = makeAuditor();
    await a.open();
    const rec = await a.record("gpt-4o", "in", "out");
    expect(rec.confidence).toBeNull();
  });

  it("stores metadata when provided", async () => {
    const a = makeAuditor();
    await a.open();
    const rec = await a.record("gpt-4o", "in", "out", { metadata: { env: "prod" } });
    expect(rec.metadata["env"]).toBe("prod");
  });

  it("increments buffered count via stats()", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o", "i1", "o1");
    await a.record("claude-3", "i2", "o2");
    expect(a.stats().buffered).toBe(2);
  });

  it("record_id follows rec_{epochId}_{seq:06d} pattern", async () => {
    const a = makeAuditor();
    const open = await a.open();
    const rec = await a.record("gpt-4o", "in", "out");
    expect(rec.record_id).toBe(`rec_${open.epoch_id}_000000`);
  });
});

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------

describe("MultiModelAuditor — flush()", () => {
  it("returns epochClose with type EPOCH_CLOSE", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o", "i", "o");
    const { epochClose } = await a.flush();
    expect(epochClose.type).toBe("EPOCH_CLOSE");
  });

  it("returns all accumulated records", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o",  "i1", "o1");
    await a.record("claude-3", "i2", "o2");
    const { records } = await a.flush();
    expect(records).toHaveLength(2);
  });

  it("epochClose.record_count matches number of records", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o",  "i1", "o1");
    await a.record("claude-3", "i2", "o2");
    const { epochClose, records } = await a.flush();
    expect(epochClose.record_count).toBe(records.length);
  });

  it("epochClose.merkle_root is a 64-char hex string", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o", "in", "out");
    const { epochClose } = await a.flush();
    expect(epochClose.merkle_root).toMatch(/^[0-9a-f]{64}$/);
  });

  it("resets buffered count to 0 after flush", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o", "i", "o");
    await a.flush();
    expect(a.stats().buffered).toBe(0);
  });

  it("resets epochId to empty string after flush", async () => {
    const a = makeAuditor();
    await a.open();
    await a.record("gpt-4o", "i", "o");
    await a.flush();
    expect(a.stats().epochId).toBe("");
  });

  it("flush on empty buffer still returns empty records", async () => {
    const a = makeAuditor();
    await a.open();
    const { records, epochClose } = await a.flush();
    expect(records).toHaveLength(0);
    expect(epochClose.record_count).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// stats()
// ---------------------------------------------------------------------------

describe("MultiModelAuditor — stats()", () => {
  it("models list is stable across open/record/flush cycles", async () => {
    const a = makeAuditor([MODEL_A, MODEL_B, MODEL_C]);
    const { models: before } = a.stats();
    await a.open();
    await a.record("gpt-4o", "i", "o");
    await a.flush();
    const { models: after } = a.stats();
    expect(new Set(after)).toEqual(new Set(before));
  });

  it("epochId is set after open()", async () => {
    const a = makeAuditor();
    await a.open();
    expect(a.stats().epochId.length).toBeGreaterThan(0);
  });
});
