/**
 * aria-bsv/broadcaster — ARC broadcaster for BSV transactions.
 *
 * Sends raw transactions to a TAAL ARC endpoint with exponential back-off
 * retry.  Mirrors the Python SDK's ``aria.wallet.arc`` module.
 */

import type { TxStatus, BroadcasterOptions } from "./types.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_ARC_URL = "https://arc.taal.com";
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_BASE_DELAY_MS = 500;

// ---------------------------------------------------------------------------
// ARCBroadcaster
// ---------------------------------------------------------------------------

/**
 * Broadcast raw BSV transactions via the TAAL ARC API.
 *
 * @example
 * ```ts
 * const bc = new ARCBroadcaster({ apiKey: "mainnet_..." });
 * const status = await bc.broadcast(rawTxHex);
 * if (!status.propagated) console.error(status.message);
 * ```
 */
export class ARCBroadcaster {
  private readonly _url: string;
  private readonly _apiKey: string | undefined;
  private readonly _maxRetries: number;
  private readonly _baseDelayMs: number;

  constructor(options: BroadcasterOptions = {}) {
    this._url = (options.apiUrl ?? DEFAULT_ARC_URL).replace(/\/$/, "");
    this._apiKey = options.apiKey;
    this._maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this._baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
  }

  /**
   * Broadcast a raw transaction.
   *
   * @param rawTx  Hex-encoded raw transaction bytes.
   * @returns {@link TxStatus} with propagation result.
   */
  async broadcast(rawTx: string): Promise<TxStatus> {
    const endpoint = `${this._url}/v1/tx`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    if (this._apiKey) {
      headers["Authorization"] = `Bearer ${this._apiKey}`;
    }

    const body = JSON.stringify({ rawTx });
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this._maxRetries; attempt++) {
      if (attempt > 0) {
        await _sleep(this._baseDelayMs * 2 ** (attempt - 1));
      }

      try {
        const response = await _fetch(endpoint, {
          method: "POST",
          headers,
          body,
        });

        if (response.ok) {
          const json = await response.json() as ARCResponse;
          return {
            txid: json.txid ?? "",
            propagated: true,
            message: json.txStatus ?? "OK",
          };
        }

        // Non-retryable 4xx errors
        if (response.status >= 400 && response.status < 500) {
          const text = await response.text();
          return {
            txid: "",
            propagated: false,
            message: `ARC error ${response.status}: ${text}`,
          };
        }

        // 5xx — retry
        lastError = new Error(`ARC HTTP ${response.status}`);
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
      }
    }

    return {
      txid: "",
      propagated: false,
      message: lastError?.message ?? "Unknown error after retries",
    };
  }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface ARCResponse {
  txid?: string;
  txStatus?: string;
  detail?: string;
}

// ---------------------------------------------------------------------------
// Fetch / sleep abstractions (replaceable in tests)
// ---------------------------------------------------------------------------

type FetchFn = (
  url: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
  }
) => Promise<{ ok: boolean; status: number; json(): Promise<unknown>; text(): Promise<string> }>;

let _fetchImpl: FetchFn = async (url, init) => {
  // Browser / Node 18+ global fetch
  if (typeof globalThis.fetch === "function") {
    return globalThis.fetch(url, init);
  }
  // Node < 18 — attempt dynamic require
  if (typeof require !== "undefined") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const nodeFetch = require("node-fetch") as {
        default: FetchFn;
      };
      return nodeFetch.default(url, init);
    } catch {
      // node-fetch not installed
    }
  }
  throw new Error("No fetch implementation available. Install node-fetch or use Node 18+.");
};

/** Replace the fetch implementation (useful for testing). */
export function _setFetchImpl(fn: FetchFn): void {
  _fetchImpl = fn;
}

async function _fetch(
  url: string,
  init: Parameters<FetchFn>[1]
): ReturnType<FetchFn> {
  return _fetchImpl(url, init);
}

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
