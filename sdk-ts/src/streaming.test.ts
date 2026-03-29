/**
 * Tests for aria-bsv/streaming — StreamingSession, ARIAStreamingAuditor.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { StreamingSession, ARIAStreamingAuditor, auditOpenAIStream, auditAnthropicStream } from "./streaming.js";
import { _setFetchImpl } from "./broadcaster.js";

// ---------------------------------------------------------------------------
// Setup: mock the broadcaster so no real HTTP calls happen
// ---------------------------------------------------------------------------

beforeEach(() => {
  _setFetchImpl(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ txid: "cafecafe".repeat(8), txStatus: "MINED" }),
    text: async () => "",
  }));
});

const _CFG = {
  system_id: "test-system",
  batch_ms: 0,
  batch_size: 100,
  max_retries: 0,
  base_delay_ms: 0,
};

// ---------------------------------------------------------------------------
// StreamingSession
// ---------------------------------------------------------------------------

describe("StreamingSession", () => {
  function makeSession(model = "gpt-4o", input: unknown = "hello") {
    const auditor = new ARIAStreamingAuditor(_CFG);
    return auditor.startStream(model, input);
  }

  it("starts with empty accumulated text", () => {
    const s = makeSession();
    expect(s.accumulated).toBe("");
  });

  it("starts with 0 chunks", () => {
    expect(makeSession().chunkCount).toBe(0);
  });

  it("isFinished is false before finish()", () => {
    expect(makeSession().isFinished).toBe(false);
  });

  it("record is null before finish()", () => {
    expect(makeSession().record).toBeNull();
  });

  it("addChunk accumulates text", () => {
    const s = makeSession();
    s.addChunk("Hello");
    s.addChunk(", ");
    s.addChunk("world");
    expect(s.accumulated).toBe("Hello, world");
  });

  it("addChunk ignores empty strings", () => {
    const s = makeSession();
    s.addChunk("");
    s.addChunk("  ");
    s.addChunk("");
    expect(s.chunkCount).toBe(1); // only "  " counts
  });

  it("chunkCount reflects non-empty chunks only", () => {
    const s = makeSession();
    s.addChunk("a");
    s.addChunk("b");
    s.addChunk("");
    expect(s.chunkCount).toBe(2);
  });

  it("finish() sets isFinished to true", async () => {
    const s = makeSession();
    await s.finish();
    expect(s.isFinished).toBe(true);
  });

  it("finish() returns an AuditRecord", async () => {
    const s = makeSession();
    s.addChunk("response");
    const rec = await s.finish();
    expect(rec.model_id).toBe("gpt-4o");
    expect(rec.system_id).toBe("test-system");
  });

  it("finish() stores accumulated text as output", async () => {
    const s = makeSession();
    s.addChunk("chunk1");
    s.addChunk("chunk2");
    const rec = await s.finish();
    expect(rec.output_hash).toHaveLength(64);
    // The output hash must correspond to "chunk1chunk2"
    const { hashObjectSync } = await import("./hasher.js");
    expect(rec.output_hash).toBe(hashObjectSync("chunk1chunk2"));
  });

  it("finish() is idempotent — second call returns same record", async () => {
    const s = makeSession();
    const r1 = await s.finish();
    const r2 = await s.finish();
    expect(r1.record_id).toBe(r2.record_id);
  });

  it("finish() with confidence override", async () => {
    const s = makeSession();
    const rec = await s.finish(0.95);
    expect(rec.confidence).toBe(0.95);
  });

  it("record is accessible after finish()", async () => {
    const s = makeSession();
    await s.finish();
    expect(s.record).not.toBeNull();
    expect(s.record?.model_id).toBe("gpt-4o");
  });
});

// ---------------------------------------------------------------------------
// ARIAStreamingAuditor
// ---------------------------------------------------------------------------

describe("ARIAStreamingAuditor", () => {
  it("startStream returns a StreamingSession", () => {
    const a = new ARIAStreamingAuditor(_CFG);
    const s = a.startStream("gpt-4o", "input");
    expect(s).toBeInstanceOf(StreamingSession);
  });

  it("exposes underlying auditor", () => {
    const a = new ARIAStreamingAuditor(_CFG);
    expect(a.auditor).toBeDefined();
  });

  it("multiple sessions use the same auditor epoch", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    const s1 = a.startStream("m1", "in1");
    const s2 = a.startStream("m2", "in2");
    s1.addChunk("out1");
    s2.addChunk("out2");
    const r1 = await s1.finish();
    const r2 = await s2.finish();
    // Both records belong to the same auditor
    expect(r1.system_id).toBe("test-system");
    expect(r2.system_id).toBe("test-system");
  });

  it("withStream calls fn and auto-finishes", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    const rec = await a.withStream("gpt-4o", "prompt", async (s) => {
      s.addChunk("Hello");
      s.addChunk(" world");
    });
    expect(rec.model_id).toBe("gpt-4o");
    const { hashObjectSync } = await import("./hasher.js");
    expect(rec.output_hash).toBe(hashObjectSync("Hello world"));
  });

  it("withStream finishes even if fn throws", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    let record = null;
    try {
      await a.withStream("m", "in", async (s) => {
        s.addChunk("partial");
        throw new Error("stream error");
      });
    } catch {
      // expected
    }
    // The session should have been finished — auditor has 1 pending record
    expect(a.auditor.pendingRecords.length).toBe(1);
  });

  it("startStream passes metadata to record", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    const s = a.startStream("m", "in", { metadata: { user: "alice" } });
    const rec = await s.finish();
    expect(rec.metadata).toEqual({ user: "alice" });
  });

  it("flush() delegates to underlying auditor", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    const s = a.startStream("m", "in");
    await s.finish();
    const txid = await a.flush();
    expect(typeof txid).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// auditOpenAIStream helper
// ---------------------------------------------------------------------------

describe("auditOpenAIStream", () => {
  it("accumulates content from delta chunks", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    async function* fakeStream() {
      yield { choices: [{ delta: { content: "Hello" } }] };
      yield { choices: [{ delta: { content: " world" } }] };
      yield { choices: [{ delta: { content: null } }] };
    }
    const { record } = await auditOpenAIStream(a, "gpt-4o", "prompt", fakeStream());
    const { hashObjectSync } = await import("./hasher.js");
    expect(record.output_hash).toBe(hashObjectSync("Hello world"));
  });

  it("handles empty stream", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    async function* empty() { /* no chunks */ }
    const { record } = await auditOpenAIStream(a, "m", "in", empty());
    expect(record.model_id).toBe("m");
  });

  it("handles missing choices gracefully", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    async function* noChoices() {
      yield {};
      yield { choices: [] };
    }
    const { record } = await auditOpenAIStream(a, "m", "in", noChoices());
    expect(record).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// auditAnthropicStream helper
// ---------------------------------------------------------------------------

describe("auditAnthropicStream", () => {
  it("accumulates text_delta events", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    async function* fakeStream() {
      yield { type: "content_block_delta", delta: { type: "text_delta", text: "Hi" } };
      yield { type: "content_block_delta", delta: { type: "text_delta", text: " there" } };
      yield { type: "message_stop" };
    }
    const { record, text } = await auditAnthropicStream(a, "claude-opus-4-6", "q", fakeStream());
    expect(text).toBe("Hi there");
    const { hashObjectSync } = await import("./hasher.js");
    expect(record.output_hash).toBe(hashObjectSync("Hi there"));
  });

  it("ignores non-text_delta events", async () => {
    const a = new ARIAStreamingAuditor(_CFG);
    async function* fakeStream() {
      yield { type: "message_start" };
      yield { type: "content_block_start", delta: { type: "text", text: "ignored" } };
      yield { type: "content_block_delta", delta: { type: "text_delta", text: "real" } };
    }
    const { text } = await auditAnthropicStream(a, "m", "q", fakeStream());
    expect(text).toBe("real");
  });
});
