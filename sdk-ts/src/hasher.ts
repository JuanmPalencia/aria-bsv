/**
 * aria-bsv/hasher — Canonical JSON hashing for ARIA.
 *
 * Produces identical hashes to the Python SDK's ``aria.core.hasher`` module.
 * Rules:
 *   - Object keys sorted recursively.
 *   - No NaN or Infinity values (throws ARIASerializationError).
 *   - null → JSON null.
 *   - Arrays preserve order.
 *   - Strings are UTF-8.
 *   - Result is SHA-256 of the canonical JSON bytes.
 */

export class ARIASerializationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ARIASerializationError";
  }
}

// ---------------------------------------------------------------------------
// canonicalJson — deterministic JSON serialisation
// ---------------------------------------------------------------------------

/**
 * Serialise *value* to a canonical JSON string.
 *
 * Matches the Python SDK's ``canonical_json()`` function exactly so that
 * hashes computed by either SDK are identical.
 *
 * @throws ARIASerializationError if *value* contains NaN, Infinity, or an
 *   unsupported type (function, symbol, bigint, undefined).
 */
export function canonicalJson(value: unknown): string {
  return _serialize(value);
}

function _serialize(value: unknown): string {
  if (value === null) return "null";

  if (typeof value === "boolean") return value ? "true" : "false";

  if (typeof value === "number") {
    if (!isFinite(value)) {
      throw new ARIASerializationError(
        `Non-finite number not allowed in ARIA canonical JSON: ${value}`
      );
    }
    // Produce the same representation as Python's json.dumps for floats.
    // For integers (no fractional part), we use integer representation.
    return JSON.stringify(value);
  }

  if (typeof value === "string") return JSON.stringify(value);

  if (Array.isArray(value)) {
    const items = value.map(_serialize);
    return "[" + items.join(", ") + "]";
  }

  if (typeof value === "object") {
    // Sort keys recursively — same as Python's sort_keys=True
    const sorted = Object.keys(value as Record<string, unknown>).sort();
    const pairs = sorted.map((k) => {
      const v = (value as Record<string, unknown>)[k];
      return `${JSON.stringify(k)}: ${_serialize(v)}`;
    });
    return "{" + pairs.join(", ") + "}";
  }

  throw new ARIASerializationError(
    `Unsupported type for ARIA canonical JSON: ${typeof value}`
  );
}

// ---------------------------------------------------------------------------
// hashObject — SHA-256 of canonical JSON
// ---------------------------------------------------------------------------

/**
 * Return the SHA-256 hex digest of *value*'s canonical JSON representation.
 *
 * @returns 64-character lowercase hex string (no "sha256:" prefix, matching
 *   the Python SDK's ``hash_object()`` return value).
 *
 * @throws ARIASerializationError if *value* contains non-serialisable data.
 */
export async function hashObject(value: unknown): Promise<string> {
  const json = canonicalJson(value);
  return sha256Hex(json);
}

/**
 * Synchronous variant — uses a pure-JS SHA-256 implementation.
 * Identical output to the async variant.
 */
export function hashObjectSync(value: unknown): string {
  const json = canonicalJson(value);
  return sha256HexSync(json);
}

// ---------------------------------------------------------------------------
// SHA-256 helpers
// ---------------------------------------------------------------------------

/**
 * Compute SHA-256 of *text* (UTF-8 encoded) and return a hex string.
 * Uses the Web Crypto API when available, falls back to the pure-JS impl.
 */
export async function sha256Hex(text: string): Promise<string> {
  if (
    typeof globalThis !== "undefined" &&
    globalThis.crypto?.subtle
  ) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    const hashBuffer = await globalThis.crypto.subtle.digest("SHA-256", data);
    return _bufferToHex(new Uint8Array(hashBuffer));
  }
  // Node.js environment
  if (typeof require !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { createHash } = require("crypto") as typeof import("crypto");
    return createHash("sha256").update(text, "utf8").digest("hex");
  }
  return sha256HexSync(text);
}

/**
 * Pure-JS synchronous SHA-256.
 * Used as a fallback when Web Crypto / Node crypto are unavailable.
 */
export function sha256HexSync(text: string): string {
  const bytes = _utf8Encode(text);
  const hash = _sha256(bytes);
  return _bufferToHex(hash);
}

/**
 * Compute SHA-256 of raw bytes (no string encoding) and return hex.
 * Use this when you have a {@link Uint8Array} and do not want any string
 * encoding to occur (e.g., in Merkle tree internal nodes).
 */
export function sha256HexSyncFromBytes(data: Uint8Array): string {
  return _bufferToHex(_sha256(data));
}

// ---------------------------------------------------------------------------
// Pure-JS SHA-256 (RFC 6234)
// ---------------------------------------------------------------------------

const K: number[] = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

function _rotr32(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

function _sha256(msg: Uint8Array): Uint8Array {
  // Pre-processing: add padding
  const msgLen = msg.length;
  const bitLen = msgLen * 8;
  // pad to 512-bit blocks: 1 bit + zeros + 64-bit length
  const padLen = ((msgLen + 8) >> 6) + 1;
  const padded = new Uint8Array(padLen * 64);
  padded.set(msg);
  padded[msgLen] = 0x80;
  // Store bit length as big-endian 64-bit
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 4, bitLen >>> 0, false);
  dv.setUint32(padded.length - 8, Math.floor(bitLen / 2 ** 32), false);

  // Initial hash values
  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let i = 0; i < padded.length; i += 64) {
    const chunk = new DataView(padded.buffer, i, 64);
    for (let j = 0; j < 16; j++) w[j] = chunk.getUint32(j * 4, false);
    for (let j = 16; j < 64; j++) {
      const s0 = _rotr32(w[j - 15], 7) ^ _rotr32(w[j - 15], 18) ^ (w[j - 15] >>> 3);
      const s1 = _rotr32(w[j - 2], 17) ^ _rotr32(w[j - 2], 19) ^ (w[j - 2] >>> 10);
      w[j] = (w[j - 16] + s0 + w[j - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = [h0, h1, h2, h3, h4, h5, h6, h7];
    for (let j = 0; j < 64; j++) {
      const S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + K[j] + w[j]) >>> 0;
      const S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e;
      e = (d + temp1) >>> 0;
      d = c; c = b; b = a;
      a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
  }

  const result = new Uint8Array(32);
  const rv = new DataView(result.buffer);
  [h0, h1, h2, h3, h4, h5, h6, h7].forEach((v, i) => rv.setUint32(i * 4, v, false));
  return result;
}

function _utf8Encode(str: string): Uint8Array {
  if (typeof TextEncoder !== "undefined") {
    return new TextEncoder().encode(str);
  }
  // Node.js fallback
  return Buffer.from(str, "utf8") as unknown as Uint8Array;
}

function _bufferToHex(buf: Uint8Array): string {
  return Array.from(buf)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
