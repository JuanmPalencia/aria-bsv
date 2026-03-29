/**
 * Tests for aria-bsv/dataset — hashBytes, hashColumns, DatasetAnchorer, verifyDatasetAnchor.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  hashBytes,
  hashColumns,
  DatasetAnchorer,
  verifyDatasetAnchor,
} from "./dataset.js";
import { _setFetchImpl } from "./broadcaster.js";
import { sha256HexSyncFromBytes } from "./hasher.js";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const _FAKE_TXID = "ab".repeat(32);

beforeEach(() => {
  _setFetchImpl(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ txid: _FAKE_TXID, txStatus: "MINED" }),
    text: async () => "",
  }));
});

function makeAnchorer(withBroadcaster = true): DatasetAnchorer {
  return new DatasetAnchorer({
    system_id: "test-system",
    broadcaster: withBroadcaster ? { maxRetries: 0, baseDelayMs: 0 } : undefined,
  });
}

// ---------------------------------------------------------------------------
// hashBytes
// ---------------------------------------------------------------------------

describe("hashBytes", () => {
  it("returns sha256: prefixed string", () => {
    expect(hashBytes(new Uint8Array([1, 2, 3]))).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("is deterministic", () => {
    const d = new Uint8Array([10, 20, 30]);
    expect(hashBytes(d)).toBe(hashBytes(d));
  });

  it("different data → different hash", () => {
    expect(hashBytes(new Uint8Array([1]))).not.toBe(hashBytes(new Uint8Array([2])));
  });

  it("empty bytes has known SHA-256 prefix", () => {
    // SHA-256(b"") = e3b0c44298fc1c14...
    expect(hashBytes(new Uint8Array(0)).startsWith("sha256:e3b0c44")).toBe(true);
  });

  it("matches manual sha256HexSyncFromBytes", () => {
    const d = new TextEncoder().encode("hello dataset");
    expect(hashBytes(d)).toBe("sha256:" + sha256HexSyncFromBytes(d));
  });
});

// ---------------------------------------------------------------------------
// hashColumns
// ---------------------------------------------------------------------------

describe("hashColumns", () => {
  it("returns sha256: prefixed string", () => {
    expect(hashColumns(["a", "b"])).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it("is order-independent", () => {
    expect(hashColumns(["b", "a"])).toBe(hashColumns(["a", "b"]));
  });

  it("different columns → different hash", () => {
    expect(hashColumns(["a", "b"])).not.toBe(hashColumns(["a", "c"]));
  });

  it("single column", () => {
    expect(hashColumns(["id"])).toMatch(/^sha256:/);
  });

  it("empty list", () => {
    expect(hashColumns([])).toMatch(/^sha256:/);
  });
});

// ---------------------------------------------------------------------------
// verifyDatasetAnchor
// ---------------------------------------------------------------------------

describe("verifyDatasetAnchor", () => {
  function makeAnchor(data: Uint8Array) {
    return {
      dataset_id: "id",
      system_id: "sys",
      content_hash: hashBytes(data),
      schema_hash: null,
      row_count: null,
      column_names: null,
      media_type: "application/octet-stream",
      anchored_at: "2025-01-01T00:00:00.000Z",
      txid: _FAKE_TXID,
      payload: {} as never,
    };
  }

  it("matching data returns true", () => {
    const d = new TextEncoder().encode("original");
    expect(verifyDatasetAnchor(d, makeAnchor(d))).toBe(true);
  });

  it("tampered data returns false", () => {
    const original = new TextEncoder().encode("original");
    const tampered = new TextEncoder().encode("tampered");
    expect(verifyDatasetAnchor(tampered, makeAnchor(original))).toBe(false);
  });

  it("empty bytes match", () => {
    const d = new Uint8Array(0);
    expect(verifyDatasetAnchor(d, makeAnchor(d))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// DatasetAnchorer.anchorBytes
// ---------------------------------------------------------------------------

describe("DatasetAnchorer.anchorBytes", () => {
  it("returns a DatasetAnchor", async () => {
    const a = makeAnchorer();
    const result = await a.anchorBytes(new TextEncoder().encode("data"));
    expect(result.system_id).toBe("test-system");
    expect(result.content_hash).toMatch(/^sha256:/);
  });

  it("content_hash matches hashBytes", async () => {
    const data = new TextEncoder().encode("my dataset");
    const a = makeAnchorer();
    const result = await a.anchorBytes(data);
    expect(result.content_hash).toBe(hashBytes(data));
  });

  it("txid set when broadcaster configured", async () => {
    const result = await makeAnchorer(true).anchorBytes(new TextEncoder().encode("x"));
    expect(result.txid).toBe(_FAKE_TXID);
  });

  it("txid empty without broadcaster", async () => {
    const result = await makeAnchorer(false).anchorBytes(new TextEncoder().encode("x"));
    expect(result.txid).toBe("");
  });

  it("default media_type is application/octet-stream", async () => {
    const result = await makeAnchorer().anchorBytes(new TextEncoder().encode("x"));
    expect(result.media_type).toBe("application/octet-stream");
  });

  it("custom media_type", async () => {
    const result = await makeAnchorer().anchorBytes(
      new TextEncoder().encode("a,b"),
      { media_type: "text/csv" }
    );
    expect(result.media_type).toBe("text/csv");
  });

  it("row_count stored", async () => {
    const result = await makeAnchorer().anchorBytes(
      new TextEncoder().encode("x"),
      { row_count: 1000 }
    );
    expect(result.row_count).toBe(1000);
  });

  it("column_names stored", async () => {
    const result = await makeAnchorer().anchorBytes(
      new TextEncoder().encode("x"),
      { column_names: ["a", "b"] }
    );
    expect(result.column_names).toEqual(["a", "b"]);
  });

  it("schema_hash computed when columns provided", async () => {
    const result = await makeAnchorer().anchorBytes(
      new TextEncoder().encode("x"),
      { column_names: ["a", "b"] }
    );
    expect(result.schema_hash).toBe(hashColumns(["a", "b"]));
  });

  it("schema_hash null without columns", async () => {
    const result = await makeAnchorer().anchorBytes(new TextEncoder().encode("x"));
    expect(result.schema_hash).toBeNull();
  });

  it("dataset_id is unique each call", async () => {
    const a = makeAnchorer();
    const r1 = await a.anchorBytes(new TextEncoder().encode("x"));
    const r2 = await a.anchorBytes(new TextEncoder().encode("x"));
    expect(r1.dataset_id).not.toBe(r2.dataset_id);
  });

  it("payload contains required fields", async () => {
    const result = await makeAnchorer().anchorBytes(new TextEncoder().encode("x"));
    expect(result.payload.type).toBe("DATASET_ANCHOR");
    expect(result.payload.brc121_version).toBe("1.0");
    expect(result.payload.nonce).toHaveLength(32);
  });
});

// ---------------------------------------------------------------------------
// DatasetAnchorer.anchorText
// ---------------------------------------------------------------------------

describe("DatasetAnchorer.anchorText", () => {
  it("content_hash matches UTF-8 bytes", async () => {
    const text = "hello world";
    const result = await makeAnchorer().anchorText(text);
    expect(result.content_hash).toBe(hashBytes(new TextEncoder().encode(text)));
  });

  it("default media_type is text/plain", async () => {
    const result = await makeAnchorer().anchorText("x");
    expect(result.media_type).toBe("text/plain");
  });

  it("custom media_type", async () => {
    const result = await makeAnchorer().anchorText("a,b", { media_type: "text/csv" });
    expect(result.media_type).toBe("text/csv");
  });

  it("verify roundtrip", async () => {
    const text = "col1,col2\n1,2\n";
    const anchor = await makeAnchorer().anchorText(text);
    expect(verifyDatasetAnchor(new TextEncoder().encode(text), anchor)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// DatasetAnchorer.anchorJson
// ---------------------------------------------------------------------------

describe("DatasetAnchorer.anchorJson", () => {
  it("media_type is application/json", async () => {
    const result = await makeAnchorer().anchorJson({ key: "value" });
    expect(result.media_type).toBe("application/json");
  });

  it("key order independent", async () => {
    const a = makeAnchorer(false);
    const r1 = await a.anchorJson({ b: 2, a: 1 });
    const r2 = await a.anchorJson({ a: 1, b: 2 });
    expect(r1.content_hash).toBe(r2.content_hash);
  });

  it("different objects different hash", async () => {
    const a = makeAnchorer(false);
    const r1 = await a.anchorJson({ x: 1 });
    const r2 = await a.anchorJson({ x: 2 });
    expect(r1.content_hash).not.toBe(r2.content_hash);
  });

  it("array preserves order", async () => {
    const a = makeAnchorer(false);
    const r1 = await a.anchorJson([1, 2, 3]);
    const r2 = await a.anchorJson([3, 2, 1]);
    expect(r1.content_hash).not.toBe(r2.content_hash);
  });
});
