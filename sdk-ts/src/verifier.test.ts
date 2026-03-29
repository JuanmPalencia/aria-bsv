/**
 * Tests for aria-bsv/verifier — verifyEpoch (off-chain).
 */

import { describe, it, expect } from "vitest";
import { verifyEpoch } from "./verifier.js";
import { hashObject } from "./hasher.js";
import { computeMerkleRoot } from "./merkle.js";
import type {
  AuditRecord,
  EpochOpenPayload,
  EpochClosePayload,
} from "./types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SYSTEM_ID = "test-system";
const EPOCH_ID = "1234567890_abcdef";
const OPEN_TXID = "aa".repeat(32);
const CLOSE_TXID = "bb".repeat(32);

function makeRecord(seq: number, overrides: Partial<AuditRecord> = {}): AuditRecord {
  return {
    record_id: `rec_${EPOCH_ID}_${String(seq).padStart(6, "0")}`,
    system_id: SYSTEM_ID,
    model_id: "test-model",
    input_hash: "0".repeat(64),
    output_hash: "1".repeat(64),
    confidence: null,
    latency_ms: 10,
    timestamp: "2025-01-01T00:00:00.000Z",
    sequence: seq,
    epoch_id: EPOCH_ID,
    metadata: {},
    ...overrides,
  };
}

async function makeValidEpoch(records: AuditRecord[]) {
  const leafHashes = await Promise.all(records.map((r) => hashObject(r)));
  const merkleRoot = await computeMerkleRoot(leafHashes);

  const openPayload: EpochOpenPayload = {
    type: "EPOCH_OPEN",
    brc121_version: "1.0",
    system_id: SYSTEM_ID,
    epoch_id: EPOCH_ID,
    opened_at: "2025-01-01T00:00:00.000Z",
    model_hashes: {},
    state_hash: "0".repeat(64),
    nonce: "0".repeat(32),
  };

  const closePayload: EpochClosePayload = {
    type: "EPOCH_CLOSE",
    brc121_version: "1.0",
    system_id: SYSTEM_ID,
    epoch_id: EPOCH_ID,
    prev_txid: OPEN_TXID,
    merkle_root: merkleRoot,
    record_count: records.length,
    closed_at: "2025-01-01T01:00:00.000Z",
  };

  return { openPayload, closePayload, merkleRoot };
}

// ---------------------------------------------------------------------------
// Valid epoch
// ---------------------------------------------------------------------------

describe("verifyEpoch — valid", () => {
  it("returns valid=true for a correct single-record epoch", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it("returns valid=true for a multi-record epoch", async () => {
    const records = [0, 1, 2, 3].map((i) => makeRecord(i));
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(true);
  });

  it("returns correct system_id", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.system_id).toBe(SYSTEM_ID);
  });

  it("returns correct epoch_id", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.epoch_id).toBe(EPOCH_ID);
  });

  it("returns correct record_count", async () => {
    const records = [0, 1, 2].map((i) => makeRecord(i));
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.record_count).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Payload type errors
// ---------------------------------------------------------------------------

describe("verifyEpoch — payload type errors", () => {
  it("fails if openPayload.type is wrong", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    // @ts-expect-error — intentional type mutation for test
    openPayload.type = "WRONG";
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("EPOCH_OPEN"))).toBe(true);
  });

  it("fails if closePayload.type is wrong", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    // @ts-expect-error
    closePayload.type = "WRONG";
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Linkage errors
// ---------------------------------------------------------------------------

describe("verifyEpoch — linkage errors", () => {
  it("fails if prev_txid does not match openTxid", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    closePayload.prev_txid = "cc".repeat(32);  // wrong
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("prev_txid"))).toBe(true);
  });

  it("fails if epoch IDs do not match", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    closePayload.epoch_id = "different_epoch";
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });

  it("fails if system IDs differ between payloads", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    closePayload.system_id = "other-system";
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Record integrity errors
// ---------------------------------------------------------------------------

describe("verifyEpoch — record integrity errors", () => {
  it("fails if record_count mismatches", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    closePayload.record_count = 99;
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("record_count"))).toBe(true);
  });

  it("fails if a record has wrong epoch_id", async () => {
    const records = [makeRecord(0, { epoch_id: "wrong-epoch" })];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });

  it("fails if a record has wrong system_id", async () => {
    const records = [makeRecord(0, { system_id: "impostor" })];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });

  it("fails on non-contiguous sequence numbers", async () => {
    // Sequences [0, 2] — gap at 1
    const records = [makeRecord(0), makeRecord(2)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
  });

  it("fails if Merkle root is tampered", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    closePayload.merkle_root = "d".repeat(64);
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("Merkle root"))).toBe(true);
  });

  it("fails if a record is modified after epoch was created", async () => {
    const records = [makeRecord(0)];
    const { openPayload, closePayload } = await makeValidEpoch(records);
    // Tamper with the record after computing the Merkle root
    records[0].model_id = "tampered-model";
    const result = await verifyEpoch(
      records, openPayload, closePayload, OPEN_TXID, CLOSE_TXID
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("Merkle"))).toBe(true);
  });
});
