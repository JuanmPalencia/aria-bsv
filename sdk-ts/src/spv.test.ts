/**
 * Tests for aria-bsv/spv — BlockHeader, MerkleBranch, SPV verification.
 */

import { describe, it, expect } from "vitest";
import {
  BlockHeader,
  SPVError,
  sha256d,
  reverseHex,
  verifyMerkleBranch,
  verifyHeaderChain,
  verifySpvProof,
} from "./spv.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  return out;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a); out.set(b, a.length);
  return out;
}

/** Build a raw 80-byte header (all fields LE). */
function makeHeaderBytes(opts: {
  version?: number;
  prevBlock?: Uint8Array;
  merkleRoot?: Uint8Array;
  time?: number;
  bits?: number;
  nonce?: number;
} = {}): Uint8Array {
  const buf = new Uint8Array(80);
  const view = new DataView(buf.buffer);
  view.setUint32(0, opts.version ?? 1, true);
  buf.set(opts.prevBlock ?? new Uint8Array(32), 4);
  buf.set(opts.merkleRoot ?? new Uint8Array(32), 36);
  view.setUint32(68, opts.time ?? 1_600_000_000, true);
  view.setUint32(72, opts.bits ?? 0x207fffff, true);
  view.setUint32(76, opts.nonce ?? 0, true);
  return buf;
}

function bytesToHexStr(b: Uint8Array) { return bytesToHex(b); }

/** Mine a header (find nonce that meets bits=0x207fffff). */
function mineHeader(overrides: {
  prevBlock?: string;  // display hex
  merkleRoot?: string; // display hex
} = {}): BlockHeader {
  const bits = 0x207fffff;
  const prevInternal = overrides.prevBlock
    ? hexToBytes(overrides.prevBlock).reverse()
    : new Uint8Array(32);
  const rootInternal = overrides.merkleRoot
    ? hexToBytes(overrides.merkleRoot).reverse()
    : new Uint8Array(32);

  for (let nonce = 0; nonce < 2 ** 32; nonce++) {
    const raw = makeHeaderBytes({
      prevBlock: prevInternal,
      merkleRoot: rootInternal,
      bits,
      nonce,
    });
    const header = BlockHeader.fromHex(bytesToHex(raw));
    if (header.meetsTarget()) return header;
  }
  throw new Error("Could not mine header");
}

/** Compute the standard Bitcoin Merkle root (SHA-256d, duplicate last if odd). */
function buildMerkleRoot(txidsInternal: Uint8Array[]): Uint8Array {
  let layer = [...txidsInternal];
  while (layer.length > 1) {
    if (layer.length % 2 === 1) layer.push(layer[layer.length - 1]);
    const next: Uint8Array[] = [];
    for (let i = 0; i < layer.length; i += 2) {
      next.push(sha256d(concatBytes(layer[i], layer[i + 1])));
    }
    layer = next;
  }
  return layer[0];
}

/** Build the Merkle branch (sibling hashes in internal order) for index. */
function buildBranch(txidsInternal: Uint8Array[], index: number): string[] {
  let layer = [...txidsInternal];
  const hashes: string[] = [];
  let idx = index;
  while (layer.length > 1) {
    if (layer.length % 2 === 1) layer.push(layer[layer.length - 1]);
    const siblingIdx = idx ^ 1;
    hashes.push(bytesToHex(layer[siblingIdx]));
    const next: Uint8Array[] = [];
    for (let i = 0; i < layer.length; i += 2) {
      next.push(sha256d(concatBytes(layer[i], layer[i + 1])));
    }
    layer = next;
    idx = Math.floor(idx / 2);
  }
  return hashes;
}

// ---------------------------------------------------------------------------
// reverseHex
// ---------------------------------------------------------------------------

describe("reverseHex", () => {
  it("reverses byte order", () => {
    expect(reverseHex("0102")).toBe("0201");
  });

  it("roundtrip", () => {
    const h = "deadbeef";
    expect(reverseHex(reverseHex(h))).toBe(h);
  });

  it("empty string", () => {
    expect(reverseHex("")).toBe("");
  });
});

// ---------------------------------------------------------------------------
// BlockHeader
// ---------------------------------------------------------------------------

describe("BlockHeader", () => {
  it("fromHex round-trips to toBytes", () => {
    const raw = makeHeaderBytes({ version: 2, nonce: 42 });
    const hex = bytesToHex(raw);
    const h = BlockHeader.fromHex(hex);
    expect(bytesToHex(h.toBytes())).toBe(hex);
  });

  it("wrong length throws SPVError", () => {
    expect(() => BlockHeader.fromHex("00".repeat(79))).toThrow(SPVError);
    expect(() => BlockHeader.fromHex("00".repeat(81))).toThrow(SPVError);
  });

  it("version parsed correctly", () => {
    const raw = makeHeaderBytes({ version: 3 });
    expect(BlockHeader.fromHex(bytesToHex(raw)).version).toBe(3);
  });

  it("time parsed correctly", () => {
    const raw = makeHeaderBytes({ time: 1_700_000_000 });
    expect(BlockHeader.fromHex(bytesToHex(raw)).time).toBe(1_700_000_000);
  });

  it("nonce parsed correctly", () => {
    const raw = makeHeaderBytes({ nonce: 12345 });
    expect(BlockHeader.fromHex(bytesToHex(raw)).nonce).toBe(12345);
  });

  it("blockHash is 64 hex chars", () => {
    const h = BlockHeader.fromHex(bytesToHex(makeHeaderBytes()));
    expect(h.blockHash()).toHaveLength(64);
    expect(h.blockHash()).toMatch(/^[0-9a-f]{64}$/);
  });

  it("different nonces produce different hashes", () => {
    const h1 = BlockHeader.fromHex(bytesToHex(makeHeaderBytes({ nonce: 0 })));
    const h2 = BlockHeader.fromHex(bytesToHex(makeHeaderBytes({ nonce: 1 })));
    expect(h1.blockHash()).not.toBe(h2.blockHash());
  });

  it("meetsTarget true for easy bits (mined)", () => {
    const h = mineHeader();
    expect(h.meetsTarget()).toBe(true);
  });

  it("meetsTarget false for impossible bits", () => {
    // bits = 0x01000000 → target = 0
    const raw = makeHeaderBytes({ bits: 0x01000000 });
    const h = BlockHeader.fromHex(bytesToHex(raw));
    expect(h.meetsTarget()).toBe(false);
  });

  it("merkleRoot round-trips in display order", () => {
    const mr = "ab".repeat(32);
    const raw = makeHeaderBytes({ merkleRoot: hexToBytes(mr).reverse() });
    const h = BlockHeader.fromHex(bytesToHex(raw));
    expect(h.merkleRoot).toBe(mr);
  });
});

// ---------------------------------------------------------------------------
// verifyMerkleBranch
// ---------------------------------------------------------------------------

describe("verifyMerkleBranch", () => {
  function setup(txidsDisplay: string[], targetIdx: number) {
    const internal = txidsDisplay.map((t) => hexToBytes(reverseHex(t)));
    const root = buildMerkleRoot(internal);
    const rootDisplay = bytesToHex(root.slice().reverse());
    const branchHashes = buildBranch(internal, targetIdx);
    return { txid: txidsDisplay[targetIdx], branchHashes, rootDisplay };
  }

  it("single tx verifies", () => {
    const { txid, branchHashes, rootDisplay } = setup(["aa".repeat(32)], 0);
    expect(verifyMerkleBranch(txid, { txIndex: 0, hashes: branchHashes }, rootDisplay)).toBe(true);
  });

  it("two txs — index 0 verifies", () => {
    const { txid, branchHashes, rootDisplay } = setup(["aa".repeat(32), "bb".repeat(32)], 0);
    expect(verifyMerkleBranch(txid, { txIndex: 0, hashes: branchHashes }, rootDisplay)).toBe(true);
  });

  it("two txs — index 1 verifies", () => {
    const { txid, branchHashes, rootDisplay } = setup(["aa".repeat(32), "bb".repeat(32)], 1);
    expect(verifyMerkleBranch(txid, { txIndex: 1, hashes: branchHashes }, rootDisplay)).toBe(true);
  });

  it("four txs — all indices verify", () => {
    const txids = ["aa".repeat(32), "bb".repeat(32), "cc".repeat(32), "dd".repeat(32)];
    for (let i = 0; i < 4; i++) {
      const { txid, branchHashes, rootDisplay } = setup(txids, i);
      expect(verifyMerkleBranch(txid, { txIndex: i, hashes: branchHashes }, rootDisplay)).toBe(true);
    }
  });

  it("tampered txid fails", () => {
    const txids = ["aa".repeat(32), "bb".repeat(32)];
    const { branchHashes, rootDisplay } = setup(txids, 0);
    expect(verifyMerkleBranch("cc".repeat(32), { txIndex: 0, hashes: branchHashes }, rootDisplay)).toBe(false);
  });

  it("wrong root fails", () => {
    const { txid, branchHashes } = setup(["aa".repeat(32), "bb".repeat(32)], 0);
    expect(verifyMerkleBranch(txid, { txIndex: 0, hashes: branchHashes }, "ff".repeat(32))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// verifyHeaderChain
// ---------------------------------------------------------------------------

describe("verifyHeaderChain", () => {
  it("empty chain is valid", () => {
    const r = verifyHeaderChain([]);
    expect(r.valid).toBe(true);
    expect(r.chainLength).toBe(0);
  });

  it("single header without PoW check is valid", () => {
    const h = BlockHeader.fromHex(bytesToHex(makeHeaderBytes()));
    const r = verifyHeaderChain([h], { checkPow: false });
    expect(r.valid).toBe(true);
  });

  it("two correctly linked mined headers pass", () => {
    const h1 = mineHeader();
    const h2 = mineHeader({ prevBlock: h1.blockHash() });
    const r = verifyHeaderChain([h1, h2], { checkPow: true });
    expect(r.valid).toBe(true);
  });

  it("broken linkage detected", () => {
    const h1 = mineHeader();
    const h2 = mineHeader({ prevBlock: "ff".repeat(32) });
    const r = verifyHeaderChain([h1, h2], { checkPow: false });
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => e.includes("prevBlock"))).toBe(true);
  });

  it("PoW failure detected", () => {
    const raw = makeHeaderBytes({ bits: 0x01000000 });
    const h = BlockHeader.fromHex(bytesToHex(raw));
    const r = verifyHeaderChain([h], { checkPow: true });
    expect(r.valid).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// verifySpvProof
// ---------------------------------------------------------------------------

describe("verifySpvProof", () => {
  function makeProof(txidsDisplay: string[], targetIdx: number) {
    const internal = txidsDisplay.map((t) => hexToBytes(reverseHex(t)));
    const root = buildMerkleRoot(internal);
    const merkleRootDisplay = bytesToHex(root.slice().reverse());
    const branchHashes = buildBranch(internal, targetIdx);
    const header = mineHeader({ merkleRoot: merkleRootDisplay });
    return {
      txid: txidsDisplay[targetIdx],
      branch: { txIndex: targetIdx, hashes: branchHashes },
      header,
    };
  }

  it("valid proof returns true", () => {
    const proof = makeProof(["aa".repeat(32), "bb".repeat(32)], 0);
    expect(verifySpvProof(proof, { checkPow: true })).toBe(true);
  });

  it("all txs in 4-tx block verify", () => {
    const txids = ["aa".repeat(32), "bb".repeat(32), "cc".repeat(32), "dd".repeat(32)];
    for (let i = 0; i < 4; i++) {
      const proof = makeProof(txids, i);
      expect(verifySpvProof(proof, { checkPow: true })).toBe(true);
    }
  });

  it("tampered txid fails", () => {
    const proof = makeProof(["aa".repeat(32), "bb".repeat(32)], 0);
    expect(verifySpvProof({ ...proof, txid: "cc".repeat(32) }, { checkPow: false })).toBe(false);
  });
});
