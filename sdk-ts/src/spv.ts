/**
 * aria-bsv/spv — Offline Simplified Payment Verification for BSV.
 *
 * Mirrors ``aria.spv`` from the Python SDK.
 * No external dependencies — uses only the pure-JS SHA-256 from hasher.ts.
 *
 * ## Block header layout (80 bytes, little-endian):
 * ```
 * version (4) | prev_block (32) | merkle_root (32) | time (4) | bits (4) | nonce (4)
 * ```
 *
 * ## Two distinct Merkle trees:
 * - **SPV / block Merkle tree**: uses double SHA-256 (SHA-256d), no prefixes.
 *   Used to verify txids in block headers.
 * - **ARIA audit Merkle tree**: uses single SHA-256 with 0x00/0x01 RFC 6962
 *   prefixes (see merkle.ts). Used to anchor audit records.
 */

import { sha256HexSyncFromBytes } from "./hasher.js";

// ---------------------------------------------------------------------------
// SHA-256d helper
// ---------------------------------------------------------------------------

function _hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function _bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * SHA-256(SHA-256(data)) — standard Bitcoin double hash.
 */
export function sha256d(data: Uint8Array): Uint8Array {
  const first = _hexToBytes(sha256HexSyncFromBytes(data));
  return _hexToBytes(sha256HexSyncFromBytes(first));
}

/**
 * Reverse the byte order of a hex string (convert between display and internal order).
 */
export function reverseHex(hex: string): string {
  return _bytesToHex(_hexToBytes(hex).reverse());
}

// ---------------------------------------------------------------------------
// BlockHeader
// ---------------------------------------------------------------------------

export class SPVError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SPVError";
  }
}

/**
 * Parsed BSV block header (80 bytes).
 *
 * All hex values are stored in **display (big-endian) byte order** for
 * readability, matching what explorers and wallets show.
 */
export class BlockHeader {
  readonly version: number;
  /** 64 hex chars, display (big-endian) byte order. */
  readonly prevBlock: string;
  /** 64 hex chars, display (big-endian) byte order. */
  readonly merkleRoot: string;
  readonly time: number;
  /** Compact target representation. */
  readonly bits: number;
  readonly nonce: number;

  constructor(fields: {
    version: number;
    prevBlock: string;
    merkleRoot: string;
    time: number;
    bits: number;
    nonce: number;
  }) {
    this.version = fields.version;
    this.prevBlock = fields.prevBlock;
    this.merkleRoot = fields.merkleRoot;
    this.time = fields.time;
    this.bits = fields.bits;
    this.nonce = fields.nonce;
  }

  /**
   * Parse a hex-encoded 80-byte block header.
   *
   * @throws {@link SPVError} if the input is not exactly 160 hex chars (80 bytes).
   */
  static fromHex(hexHeader: string): BlockHeader {
    const raw = _hexToBytes(hexHeader);
    if (raw.length !== 80) {
      throw new SPVError(`Block header must be 80 bytes, got ${raw.length}`);
    }
    const view = new DataView(raw.buffer);
    const version = view.getUint32(0, true);       // LE
    const prevBlock = _bytesToHex(raw.slice(4, 36).reverse());  // display order
    const merkleRoot = _bytesToHex(raw.slice(36, 68).reverse()); // display order
    const time = view.getUint32(68, true);
    const bits = view.getUint32(72, true);
    const nonce = view.getUint32(76, true);
    return new BlockHeader({ version, prevBlock, merkleRoot, time, bits, nonce });
  }

  /**
   * Serialise back to 80 raw bytes (wire format).
   */
  toBytes(): Uint8Array {
    const buf = new Uint8Array(80);
    const view = new DataView(buf.buffer);
    view.setUint32(0, this.version, true);
    // Store in internal (little-endian) order
    const prevBytes = _hexToBytes(this.prevBlock).reverse();
    const rootBytes = _hexToBytes(this.merkleRoot).reverse();
    buf.set(prevBytes, 4);
    buf.set(rootBytes, 36);
    view.setUint32(68, this.time, true);
    view.setUint32(72, this.bits, true);
    view.setUint32(76, this.nonce, true);
    return buf;
  }

  /**
   * Return the block hash as a 64-hex display-order (big-endian) string.
   *
   * block_hash = SHA-256d(header_bytes) reversed for display.
   */
  blockHash(): string {
    const raw = sha256d(this.toBytes());
    return _bytesToHex(raw.reverse());
  }

  /**
   * Decode the ``bits`` compact representation to the full 256-bit target integer.
   *
   * @returns The target as a BigInt.
   */
  target(): bigint {
    const exponent = (this.bits >>> 24) & 0xff;
    const mantissa = BigInt(this.bits & 0x007fffff);
    if (exponent >= 3) {
      return mantissa * (256n ** BigInt(exponent - 3));
    }
    // exponent < 3: right-shift (integer division, result may be 0)
    return mantissa / (256n ** BigInt(3 - exponent));
  }

  /**
   * Return true if the block hash is ≤ the declared target (valid PoW).
   */
  meetsTarget(): boolean {
    const hashInt = BigInt("0x" + this.blockHash());
    return hashInt <= this.target();
  }
}

// ---------------------------------------------------------------------------
// Merkle branch
// ---------------------------------------------------------------------------

export interface MerkleBranch {
  /** 0-based position of the transaction in the block. */
  txIndex: number;
  /**
   * Sibling hashes at each level (internal/little-endian byte order),
   * from leaf to root.
   */
  hashes: string[];
}

/**
 * Verify that *txid* is included in a block whose Merkle root equals
 * *expectedMerkleRoot*.
 *
 * Uses the standard Bitcoin Merkle tree (SHA-256d, no second-preimage
 * prefix — different from the ARIA audit Merkle tree).
 *
 * @param txid                64-char hex txid (display/big-endian order).
 * @param branch              {@link MerkleBranch} from the block.
 * @param expectedMerkleRoot  Merkle root from the block header (display order).
 * @returns ``true`` if the proof is valid.
 */
export function verifyMerkleBranch(
  txid: string,
  branch: MerkleBranch,
  expectedMerkleRoot: string
): boolean {
  // Work in internal (little-endian) byte order
  let current = _hexToBytes(reverseHex(txid));
  let index = branch.txIndex;

  for (const siblingHex of branch.hashes) {
    const sibling = _hexToBytes(siblingHex);
    const combined = new Uint8Array(64);
    if (index % 2 === 0) {
      combined.set(current, 0);
      combined.set(sibling, 32);
    } else {
      combined.set(sibling, 0);
      combined.set(current, 32);
    }
    current = sha256d(combined);
    index = Math.floor(index / 2);
  }

  const computedRoot = _bytesToHex(current.reverse());
  return computedRoot === expectedMerkleRoot;
}

// ---------------------------------------------------------------------------
// Header chain verifier
// ---------------------------------------------------------------------------

export interface SPVChainResult {
  valid: boolean;
  chainLength: number;
  errors: string[];
}

/**
 * Verify that a sequence of {@link BlockHeader} objects form a valid chain.
 *
 * @param headers   Ordered list, oldest first.
 * @param checkPow  If true (default), also verify PoW for each header.
 */
export function verifyHeaderChain(
  headers: BlockHeader[],
  options: { checkPow?: boolean } = {}
): SPVChainResult {
  const checkPow = options.checkPow ?? true;
  const errors: string[] = [];

  if (headers.length === 0) {
    return { valid: true, chainLength: 0, errors: [] };
  }

  if (checkPow && !headers[0].meetsTarget()) {
    errors.push("Header 0 does not meet PoW target");
  }

  for (let i = 1; i < headers.length; i++) {
    const prevHash = headers[i - 1].blockHash();
    if (prevHash !== headers[i].prevBlock) {
      errors.push(
        `Header ${i} prevBlock mismatch: expected ${prevHash}, got ${headers[i].prevBlock}`
      );
    }
    if (checkPow && !headers[i].meetsTarget()) {
      errors.push(`Header ${i} does not meet PoW target`);
    }
  }

  return { valid: errors.length === 0, chainLength: headers.length, errors };
}

// ---------------------------------------------------------------------------
// Full SPV proof
// ---------------------------------------------------------------------------

export interface SPVProof {
  /** Transaction ID (display order). */
  txid: string;
  /** Merkle inclusion proof. */
  branch: MerkleBranch;
  /** Block header containing the transaction. */
  header: BlockHeader;
}

/**
 * Verify a complete SPV proof in one call.
 *
 * @param proof     {@link SPVProof} with txid, branch, and header.
 * @param checkPow  Also verify proof-of-work (default: true).
 * @returns ``true`` if both the Merkle branch and PoW are valid.
 */
export function verifySpvProof(
  proof: SPVProof,
  options: { checkPow?: boolean } = {}
): boolean {
  const checkPow = options.checkPow ?? true;
  if (checkPow && !proof.header.meetsTarget()) return false;
  return verifyMerkleBranch(proof.txid, proof.branch, proof.header.merkleRoot);
}
