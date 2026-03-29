/**
 * sCrypt adapter for ARIA contracts.
 *
 * In production, replace `ScryptMock` with the real `@scrypt-inc/scrypt-ts`
 * classes. This adapter lets contracts be tested in CI without the sCrypt
 * compiler toolchain installed.
 *
 * Production usage:
 * ```ts
 * import { SmartContract, prop, method, assert, sha256, Sha256, PubKeyHash,
 *          ByteString, toByteString } from "@scrypt-inc/scrypt-ts"
 * ```
 * and swap `ScryptBase` with `SmartContract`.
 */

// ---------------------------------------------------------------------------
// Minimal sCrypt-compatible types (no runtime dependency on scrypt-ts)
// ---------------------------------------------------------------------------

/** SHA-256 digest — 32 bytes as hex string (64 chars). */
export type Sha256 = string & { readonly __brand: "Sha256" }

/** 20-byte HASH160 of a public key — hex string (40 chars). */
export type PubKeyHash = string & { readonly __brand: "PubKeyHash" }

/** Arbitrary byte string — hex-encoded. */
export type ByteString = string & { readonly __brand: "ByteString" }

/** BSV block height. */
export type BlockHeight = number & { readonly __brand: "BlockHeight" }

export function toSha256(hex: string): Sha256 {
  if (!/^[0-9a-f]{64}$/i.test(hex)) throw new Error(`Invalid SHA-256 hex: "${hex}"`)
  return hex.toLowerCase() as Sha256
}

export function toPubKeyHash(hex: string): PubKeyHash {
  if (!/^[0-9a-f]{40}$/i.test(hex)) throw new Error(`Invalid PubKeyHash hex: "${hex}"`)
  return hex.toLowerCase() as PubKeyHash
}

export function toByteString(hex: string): ByteString {
  if (hex.length % 2 !== 0 || !/^[0-9a-f]*$/i.test(hex))
    throw new Error(`Invalid hex ByteString: "${hex}"`)
  return hex.toLowerCase() as ByteString
}

export function toBlockHeight(n: number): BlockHeight {
  if (!Number.isInteger(n) || n < 0) throw new Error(`Invalid block height: ${n}`)
  return n as BlockHeight
}

// ---------------------------------------------------------------------------
// Locking script helpers (pure TypeScript — no BSV lib needed)
// ---------------------------------------------------------------------------

const OP_0 = "00"
const OP_RETURN = "6a"

/**
 * Encode a byte string with its OP_PUSHDATA length prefix.
 * Handles single-byte push (≤ 75 bytes), OP_PUSHDATA1 (≤ 255), OP_PUSHDATA2.
 */
function pushData(hex: string): string {
  const len = hex.length / 2
  if (len <= 75) return len.toString(16).padStart(2, "0") + hex
  if (len <= 255) return "4c" + len.toString(16).padStart(2, "0") + hex
  return "4d" + (len & 0xff).toString(16).padStart(2, "0") + ((len >> 8) & 0xff).toString(16).padStart(2, "0") + hex
}

/** Build an OP_RETURN script from a hex payload. Max 100 KB. */
export function buildOpReturnScript(payloadHex: string): ByteString {
  return toByteString(OP_0 + OP_RETURN + pushData(payloadHex))
}

/** ARIA protocol magic bytes: ASCII "ARIA" = 41524941 */
export const ARIA_MAGIC = toByteString("41524941")

/** BRC-121 version byte: 0x01 */
export const BRC121_VERSION = toByteString("01")

// ---------------------------------------------------------------------------
// Contract execution result (what a real sCrypt SmartContract.verify() returns)
// ---------------------------------------------------------------------------

export interface ContractVerifyResult {
  success: boolean
  error?: string
}

// ---------------------------------------------------------------------------
// Base class mirroring the SmartContract interface that sCrypt provides.
// Extend this in tests / dev; in production point to scrypt-ts SmartContract.
// ---------------------------------------------------------------------------

export abstract class ScryptBase {
  /** Serialize the contract state to a hex string (simulated locking script). */
  abstract getLockingScriptHex(): string

  /** Simulate verifying that an unlock witness satisfies the contract. */
  abstract verify(witness: Record<string, unknown>): ContractVerifyResult
}
