/**
 * aria-bsv/ethereum — EVM anchor for ARIA epoch commitments.
 *
 * Anchors ARIA EPOCH_CLOSE Merkle roots to EVM-compatible chains via raw
 * JSON-RPC calls (eth_sendRawTransaction, eth_getLogs).
 * No external dependencies — uses fetch() only.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Payload committed to an EVM chain for a single ARIA epoch. */
export interface EthAnchorPayload {
  epochId: string;
  merkleRoot: string;
  recordsCount: number;
  chainId: number;
}

// ---------------------------------------------------------------------------
// buildEthAnchorData
// ---------------------------------------------------------------------------

/**
 * Encode an {@link EthAnchorPayload} into the hex data field embedded in an
 * EVM transaction or log.
 *
 * Layout (81 bytes total):
 *   [0..3]   "ARIA" ASCII magic (4 bytes)
 *   [4]      version byte 0x01 (1 byte)
 *   [5..40]  epochId UTF-8, padded / truncated to 36 bytes
 *   [41..72] merkleRoot raw bytes (32 bytes from 64-char hex)
 *   [73..76] recordsCount big-endian uint32 (4 bytes)
 *   [77..80] chainId big-endian uint32 (4 bytes)
 *
 * @returns "0x" + 162-character lowercase hex string.
 */
export function buildEthAnchorData(payload: EthAnchorPayload): string {
  const buf = new Uint8Array(81);
  const dv = new DataView(buf.buffer);

  // Magic "ARIA"
  buf[0] = 0x41; buf[1] = 0x52; buf[2] = 0x49; buf[3] = 0x41;
  // Version byte
  buf[4] = 0x01;
  // epochId — UTF-8, padded to 36 bytes (null-filled)
  const epochBytes = _utf8Encode(payload.epochId);
  buf.set(epochBytes.slice(0, 36), 5);
  // merkleRoot — parse 64-char hex to 32 bytes
  const rootHex = payload.merkleRoot.replace(/^0x/, "").padStart(64, "0").slice(0, 64);
  buf.set(_hexToBytes(rootHex), 41);
  // recordsCount (4 bytes BE)
  dv.setUint32(73, payload.recordsCount >>> 0, false);
  // chainId (4 bytes BE)
  dv.setUint32(77, payload.chainId >>> 0, false);

  return "0x" + _bytesToHex(buf);
}

// ---------------------------------------------------------------------------
// EthAnchor
// ---------------------------------------------------------------------------

/**
 * EVM anchor client.  Wraps a JSON-RPC endpoint and exposes two operations:
 *
 * - {@link sendAnchor} — broadcast a signed raw transaction.
 * - {@link getAnchors} — retrieve ARIA-prefixed logs from a contract.
 *
 * @example
 * ```ts
 * const anchor = new EthAnchor({ rpcUrl: "https://mainnet.infura.io/v3/KEY", chainId: 1 });
 * const result = await anchor.sendAnchor(rawSignedTxHex);
 * if (!result.propagated) console.error(result.message);
 * ```
 */
export class EthAnchor {
  private readonly _rpcUrl: string;
  private readonly _chainId: number;

  constructor(options: { rpcUrl: string; chainId?: number }) {
    this._rpcUrl = options.rpcUrl;
    this._chainId = options.chainId ?? 1;
  }

  /**
   * Broadcast a signed raw transaction via eth_sendRawTransaction.
   *
   * @param rawTxHex Hex-encoded signed transaction (with or without "0x" prefix).
   * @returns `{ txHash, propagated, message }`
   */
  async sendAnchor(
    rawTxHex: string
  ): Promise<{ txHash: string; propagated: boolean; message: string }> {
    const raw = rawTxHex.startsWith("0x") ? rawTxHex : `0x${rawTxHex}`;
    const body = JSON.stringify({
      jsonrpc: "2.0",
      method: "eth_sendRawTransaction",
      params: [raw],
      id: 1,
    });

    try {
      const resp = await _fetchImpl(this._rpcUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body,
      });
      const json = await resp.json() as EthRpcResponse;
      if (json.error) {
        return {
          txHash: "",
          propagated: false,
          message: json.error.message ?? "RPC error",
        };
      }
      const txHash = (json.result as string | undefined) ?? "";
      return {
        txHash,
        propagated: txHash.length > 0,
        message: txHash.length > 0 ? "OK" : "No txHash returned",
      };
    } catch (err) {
      return {
        txHash: "",
        propagated: false,
        message: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /**
   * Retrieve ARIA-prefixed anchors via eth_getLogs.
   *
   * @param contractAddress EVM contract address to query.
   * @param fromBlock       Starting block number (default 0).
   * @returns Array of parsed {@link EthAnchorPayload} objects.
   */
  async getAnchors(
    contractAddress: string,
    fromBlock = 0
  ): Promise<EthAnchorPayload[]> {
    const body = JSON.stringify({
      jsonrpc: "2.0",
      method: "eth_getLogs",
      params: [{
        address: contractAddress,
        fromBlock: `0x${fromBlock.toString(16)}`,
        toBlock: "latest",
      }],
      id: 1,
    });

    try {
      const resp = await _fetchImpl(this._rpcUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body,
      });
      const json = await resp.json() as EthRpcResponse;
      if (json.error) return [];
      const logs = (json.result as EthLog[] | null) ?? [];
      return logs.flatMap((log) => {
        const parsed = _parseAnchorData(log.data ?? "");
        return parsed ? [parsed] : [];
      });
    } catch {
      return [];
    }
  }
}

// ---------------------------------------------------------------------------
// Internal RPC / log types
// ---------------------------------------------------------------------------

interface EthRpcResponse {
  jsonrpc?: string;
  id?: number;
  result?: unknown;
  error?: { code?: number; message?: string };
}

interface EthLog {
  address?: string;
  topics?: string[];
  data?: string;
  blockNumber?: string;
  transactionHash?: string;
  logIndex?: string;
}

// ---------------------------------------------------------------------------
// Data encoding / parsing
// ---------------------------------------------------------------------------

/**
 * Parse an {@link EthAnchorPayload} from a raw hex data field.
 * Returns `null` if the data does not contain a valid ARIA v1 anchor.
 */
function _parseAnchorData(hexData: string): EthAnchorPayload | null {
  const raw = hexData.replace(/^0x/, "");
  if (raw.length < 162) return null; // 81 bytes = 162 hex chars

  const bytes = _hexToBytes(raw.slice(0, 162));

  // Validate magic "ARIA"
  if (
    bytes[0] !== 0x41 || bytes[1] !== 0x52 ||
    bytes[2] !== 0x49 || bytes[3] !== 0x41
  ) {
    return null;
  }
  // Validate version byte
  if (bytes[4] !== 0x01) return null;

  // epochId (bytes 5..40 — 36 bytes, UTF-8, strip null padding)
  const epochId = _utf8Decode(bytes.slice(5, 41)).replace(/\0+$/, "");
  // merkleRoot (bytes 41..72 — 32 bytes)
  const merkleRoot = _bytesToHex(bytes.slice(41, 73));
  // recordsCount and chainId
  const dv = new DataView(bytes.buffer);
  const recordsCount = dv.getUint32(73, false);
  const chainId = dv.getUint32(77, false);

  return { epochId, merkleRoot, recordsCount, chainId };
}

// ---------------------------------------------------------------------------
// Fetch abstraction (replaceable in tests)
// ---------------------------------------------------------------------------

type FetchFn = (
  url: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
  }
) => Promise<{
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
  text(): Promise<string>;
}>;

let _fetchImpl: FetchFn = async (url, init) => {
  if (typeof globalThis.fetch === "function") {
    return globalThis.fetch(url, init);
  }
  if (typeof require !== "undefined") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require("node-fetch") as { default: FetchFn };
      return mod.default(url, init);
    } catch {
      // node-fetch not installed
    }
  }
  throw new Error("No fetch available. Use Node 18+ or install node-fetch.");
};

/** Replace the fetch implementation (useful for testing). */
export function _setEthFetchImpl(fn: FetchFn): void {
  _fetchImpl = fn;
}

// ---------------------------------------------------------------------------
// Byte utilities
// ---------------------------------------------------------------------------

function _hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function _bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function _utf8Encode(str: string): Uint8Array {
  if (typeof TextEncoder !== "undefined") {
    return new TextEncoder().encode(str);
  }
  return Buffer.from(str, "utf8") as unknown as Uint8Array;
}

function _utf8Decode(bytes: Uint8Array): string {
  if (typeof TextDecoder !== "undefined") {
    return new TextDecoder().decode(bytes);
  }
  return Buffer.from(bytes).toString("utf8");
}
