/**
 * aria-bsv/merkle — Merkle tree with second-preimage protection.
 *
 * Identical algorithm to the Python SDK's ``aria.core.merkle`` module so that
 * Merkle roots computed by either SDK are interchangeable.
 *
 * RFC 6962 / BRC-121 construction:
 *   - Leaf nodes:     SHA-256(0x00 || leaf_hash_bytes)
 *   - Internal nodes: SHA-256(0x01 || left_bytes || right_bytes)
 *   - Odd layer: the last node is promoted unchanged (not duplicated)
 *   - Empty tree: returns the all-zeros 64-char hex string
 */

import { sha256HexSyncFromBytes } from "./hasher.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Hex string → Uint8Array */
function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

/** Concatenate two Uint8Arrays */
function concat(...arrays: Uint8Array[]): Uint8Array {
  const total = arrays.reduce((n, a) => n + a.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}

const LEAF_PREFIX = new Uint8Array([0x00]);
const NODE_PREFIX = new Uint8Array([0x01]);

// ---------------------------------------------------------------------------
// Core async API
// ---------------------------------------------------------------------------

/**
 * Compute the leaf digest for a raw leaf hash string.
 *
 * ``leaf_hash`` must be a 64-character lowercase hex string (SHA-256 output).
 * Returns SHA-256(0x00 || leaf_bytes).
 */
async function _leafDigest(leafHash: string): Promise<string> {
  const payload = concat(LEAF_PREFIX, hexToBytes(leafHash));
  return _sha256Bytes(payload);
}

/**
 * Compute the internal node digest.
 * Returns SHA-256(0x01 || left_bytes || right_bytes).
 */
async function _nodeDigest(left: string, right: string): Promise<string> {
  const payload = concat(NODE_PREFIX, hexToBytes(left), hexToBytes(right));
  return _sha256Bytes(payload);
}

async function _sha256Bytes(data: Uint8Array): Promise<string> {
  if (
    typeof globalThis !== "undefined" &&
    globalThis.crypto?.subtle
  ) {
    const hashBuffer = await globalThis.crypto.subtle.digest("SHA-256", data);
    return _bufferToHex(new Uint8Array(hashBuffer));
  }
  if (typeof require !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { createHash } = require("crypto") as typeof import("crypto");
    return createHash("sha256").update(data).digest("hex");
  }
  // Pure-JS fallback — hash raw bytes directly, no string encoding
  return sha256HexSyncFromBytes(data);
}

// ---------------------------------------------------------------------------
// Core sync API (pure-JS SHA-256 only — no async needed)
// ---------------------------------------------------------------------------

function _leafDigestSync(leafHash: string): string {
  const payload = concat(LEAF_PREFIX, hexToBytes(leafHash));
  return sha256HexSyncFromBytes(payload);
}

function _nodeDigestSync(left: string, right: string): string {
  const payload = concat(NODE_PREFIX, hexToBytes(left), hexToBytes(right));
  return sha256HexSyncFromBytes(payload);
}

function _bufferToHex(buf: Uint8Array): string {
  return Array.from(buf)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ---------------------------------------------------------------------------
// EMPTY_ROOT constant (all-zeros, 64 hex chars)
// ---------------------------------------------------------------------------

export const EMPTY_ROOT = "0".repeat(64);

// ---------------------------------------------------------------------------
// MerkleTree — async variant
// ---------------------------------------------------------------------------

/**
 * Compute the Merkle root of *leafHashes* asynchronously.
 *
 * @param leafHashes Array of 64-char lowercase hex SHA-256 strings.
 * @returns 64-char lowercase hex SHA-256 string of the Merkle root.
 *   Returns {@link EMPTY_ROOT} for an empty array.
 */
export async function computeMerkleRoot(leafHashes: string[]): Promise<string> {
  if (leafHashes.length === 0) return EMPTY_ROOT;

  // Step 1: compute leaf digests
  let layer = await Promise.all(leafHashes.map(_leafDigest));

  // Step 2: reduce until one root remains
  while (layer.length > 1) {
    const next: string[] = [];
    for (let i = 0; i < layer.length; i += 2) {
      if (i + 1 < layer.length) {
        next.push(await _nodeDigest(layer[i], layer[i + 1]));
      } else {
        // Odd node — promote without hashing
        next.push(layer[i]);
      }
    }
    layer = next;
  }

  return layer[0];
}

// ---------------------------------------------------------------------------
// MerkleTree — sync variant
// ---------------------------------------------------------------------------

/**
 * Synchronous variant of {@link computeMerkleRoot}.
 * Uses the pure-JS SHA-256 fallback — identical output.
 */
export function computeMerkleRootSync(leafHashes: string[]): string {
  if (leafHashes.length === 0) return EMPTY_ROOT;

  let layer = leafHashes.map(_leafDigestSync);

  while (layer.length > 1) {
    const next: string[] = [];
    for (let i = 0; i < layer.length; i += 2) {
      if (i + 1 < layer.length) {
        next.push(_nodeDigestSync(layer[i], layer[i + 1]));
      } else {
        next.push(layer[i]);
      }
    }
    layer = next;
  }

  return layer[0];
}

// ---------------------------------------------------------------------------
// MerkleTree class — stateful convenience wrapper
// ---------------------------------------------------------------------------

/**
 * Stateful Merkle tree that accumulates leaf hashes and computes the root
 * on demand.
 *
 * @example
 * ```ts
 * const tree = new MerkleTree();
 * tree.addLeaf(hashA);
 * tree.addLeaf(hashB);
 * const root = await tree.root();
 * ```
 */
export class MerkleTree {
  private readonly _leaves: string[] = [];

  /** Append a leaf hash (64-char hex SHA-256). */
  addLeaf(leafHash: string): void {
    this._leaves.push(leafHash);
  }

  /** Number of leaves added so far. */
  get size(): number {
    return this._leaves.length;
  }

  /** Compute the Merkle root asynchronously. */
  async root(): Promise<string> {
    return computeMerkleRoot(this._leaves);
  }

  /** Compute the Merkle root synchronously (pure-JS SHA-256). */
  rootSync(): string {
    return computeMerkleRootSync(this._leaves);
  }
}

// ---------------------------------------------------------------------------
// Proof generation and verification
// ---------------------------------------------------------------------------

export interface MerkleProof {
  /** Leaf index (0-based). */
  leafIndex: number;
  /** Array of {hash, position} sibling nodes from leaf to root. */
  path: Array<{ hash: string; position: "left" | "right" }>;
}

/**
 * Generate a Merkle inclusion proof for the leaf at *leafIndex*.
 *
 * @returns A {@link MerkleProof} that can be verified with
 *   {@link verifyMerkleProof}.
 */
export async function generateMerkleProof(
  leafHashes: string[],
  leafIndex: number
): Promise<MerkleProof> {
  if (leafHashes.length === 0) {
    throw new Error("Cannot generate proof for empty tree");
  }
  if (leafIndex < 0 || leafIndex >= leafHashes.length) {
    throw new RangeError(
      `leafIndex ${leafIndex} out of range [0, ${leafHashes.length})`
    );
  }

  let layer = await Promise.all(leafHashes.map(_leafDigest));
  let idx = leafIndex;
  const path: MerkleProof["path"] = [];

  while (layer.length > 1) {
    if (idx % 2 === 0) {
      // Right sibling (or promote if no sibling)
      if (idx + 1 < layer.length) {
        path.push({ hash: layer[idx + 1], position: "right" });
      }
    } else {
      // Left sibling
      path.push({ hash: layer[idx - 1], position: "left" });
    }

    // Build next layer
    const next: string[] = [];
    for (let i = 0; i < layer.length; i += 2) {
      if (i + 1 < layer.length) {
        next.push(await _nodeDigest(layer[i], layer[i + 1]));
      } else {
        next.push(layer[i]);
      }
    }
    layer = next;
    idx = Math.floor(idx / 2);
  }

  return { leafIndex, path };
}

/**
 * Verify a Merkle inclusion proof.
 *
 * @param leafHash  The raw leaf hash (64-char hex) — NOT the leaf digest.
 * @param proof     Proof returned by {@link generateMerkleProof}.
 * @param root      Expected Merkle root (64-char hex).
 * @returns ``true`` if the proof is valid.
 */
export async function verifyMerkleProof(
  leafHash: string,
  proof: MerkleProof,
  root: string
): Promise<boolean> {
  let current = await _leafDigest(leafHash);

  for (const sibling of proof.path) {
    if (sibling.position === "right") {
      current = await _nodeDigest(current, sibling.hash);
    } else {
      current = await _nodeDigest(sibling.hash, current);
    }
  }

  return current === root;
}
