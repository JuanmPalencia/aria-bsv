/**
 * aria-bsv/overlay — BSV Overlay Services client (BRC-31).
 *
 * Mirrors ``aria.overlay`` from the Python SDK.
 *
 * Provides:
 * - {@link TopicManager}: submit transactions to overlay topics.
 * - {@link LookupService}: query outputs by system_id/epoch_id.
 * - {@link OverlayClient}: high-level combined interface.
 *
 * ARIA topics:
 * - ``tm_aria_epochs`` — EPOCH_OPEN and EPOCH_CLOSE transactions.
 * - ``ls_aria``        — Lookup service for audit records.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AdmittanceResult {
  /** 64-char hex transaction ID (may be empty if broadcast failed). */
  txid: string;
  /** Topic name. */
  topic: string;
  /** True if at least one output was admitted. */
  admitted: boolean;
  /** 0-based indices of admitted outputs. */
  admittedOutputs: number[];
  /** Human-readable status message. */
  message: string;
}

export interface LookupResult {
  txid: string;
  outputIndex: number;
  beef: string | null;
  data: Record<string, unknown>;
}

export interface OverlayClientOptions {
  baseUrl: string;
  apiKey?: string;
  epochTopic?: string;
  lookupService?: string;
}

// ---------------------------------------------------------------------------
// Fetch abstraction (replaceable in tests)
// ---------------------------------------------------------------------------

type FetchLike = (
  url: string,
  init: { method: string; headers: Record<string, string>; body: string }
) => Promise<{ ok: boolean; status: number; json(): Promise<unknown>; text(): Promise<string> }>;

let _fetchImpl: FetchLike = async (url, init) => {
  if (typeof globalThis.fetch === "function") {
    return globalThis.fetch(url, init);
  }
  if (typeof require !== "undefined") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require("node-fetch") as { default: FetchLike };
      return mod.default(url, init);
    } catch {
      // not installed
    }
  }
  throw new Error("No fetch available. Use Node 18+ or install node-fetch.");
};

/** Replace the fetch implementation (useful for testing). */
export function _setOverlayFetchImpl(fn: FetchLike): void {
  _fetchImpl = fn;
}

// ---------------------------------------------------------------------------
// TopicManager
// ---------------------------------------------------------------------------

/**
 * Submit BSV transactions to an overlay topic.
 *
 * @example
 * ```ts
 * const tm = new TopicManager({ baseUrl: "https://overlay.example.com", apiKey: "..." });
 * const result = await tm.submit(rawTxHex);
 * ```
 */
export class TopicManager {
  private readonly _base: string;
  private readonly _topic: string;
  private readonly _apiKey: string | undefined;

  constructor(options: {
    baseUrl: string;
    topic?: string;
    apiKey?: string;
  }) {
    this._base = options.baseUrl.replace(/\/$/, "");
    this._topic = options.topic ?? "tm_aria_epochs";
    this._apiKey = options.apiKey;
  }

  get topic(): string {
    return this._topic;
  }

  /**
   * Submit a raw hex transaction to the topic.
   *
   * @param rawTx Hex-encoded raw transaction.
   * @returns {@link AdmittanceResult} with the overlay node's decision.
   */
  async submit(rawTx: string): Promise<AdmittanceResult> {
    const url = `${this._base}/v1/submit`;
    const body = JSON.stringify({ rawTx, topics: [this._topic] });
    const headers = this._headers();

    const resp = await _fetchImpl(url, { method: "POST", headers, body });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Overlay node HTTP ${resp.status}: ${text.slice(0, 200)}`);
    }

    const json = await resp.json() as {
      txid?: string;
      topics?: Record<string, { outputsToAdmit?: number[] }>;
    };

    const topicData = json.topics?.[this._topic] ?? {};
    const admittedOutputs: number[] = topicData.outputsToAdmit ?? [];
    const admitted = admittedOutputs.length > 0;

    return {
      txid: json.txid ?? "",
      topic: this._topic,
      admitted,
      admittedOutputs,
      message: admitted ? "OK" : "No outputs admitted",
    };
  }

  private _headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    if (this._apiKey) h["Authorization"] = `Bearer ${this._apiKey}`;
    return h;
  }
}

// ---------------------------------------------------------------------------
// LookupService
// ---------------------------------------------------------------------------

/**
 * Query an overlay lookup service for BSV outputs.
 *
 * @example
 * ```ts
 * const ls = new LookupService({ baseUrl: "https://overlay.example.com" });
 * const results = await ls.lookup({ system_id: "my-system" });
 * ```
 */
export class LookupService {
  private readonly _base: string;
  private readonly _service: string;
  private readonly _apiKey: string | undefined;

  constructor(options: {
    baseUrl: string;
    serviceName?: string;
    apiKey?: string;
  }) {
    this._base = options.baseUrl.replace(/\/$/, "");
    this._service = options.serviceName ?? "ls_aria";
    this._apiKey = options.apiKey;
  }

  get serviceName(): string {
    return this._service;
  }

  /**
   * Query the lookup service.
   *
   * @param query  Service-specific query object.
   * @param limit  Max results (default 10).
   */
  async lookup(
    query: Record<string, unknown>,
    limit = 10
  ): Promise<LookupResult[]> {
    const url = `${this._base}/v1/lookup`;
    const body = JSON.stringify({ service: this._service, query, limit });
    const headers = this._headers();

    const resp = await _fetchImpl(url, { method: "POST", headers, body });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Lookup service HTTP ${resp.status}: ${text.slice(0, 200)}`);
    }

    const json = await resp.json() as { results?: unknown[] };
    return (json.results ?? []).map((item) => {
      const i = item as Record<string, unknown>;
      return {
        txid: String(i["txid"] ?? ""),
        outputIndex: Number(i["outputIndex"] ?? 0),
        beef: (i["beef"] as string | null | undefined) ?? null,
        data: (i["data"] as Record<string, unknown>) ?? {},
      };
    });
  }

  private _headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    if (this._apiKey) h["Authorization"] = `Bearer ${this._apiKey}`;
    return h;
  }
}

// ---------------------------------------------------------------------------
// OverlayClient
// ---------------------------------------------------------------------------

/**
 * High-level client combining {@link TopicManager} and {@link LookupService}.
 *
 * @example
 * ```ts
 * const client = new OverlayClient({ baseUrl: "https://overlay.example.com", apiKey: "..." });
 * await client.submitEpoch(rawTxHex);
 * const epochs = await client.findEpochs("my-system");
 * ```
 */
export class OverlayClient {
  readonly topicManager: TopicManager;
  readonly lookupService: LookupService;

  constructor(options: OverlayClientOptions) {
    this.topicManager = new TopicManager({
      baseUrl: options.baseUrl,
      topic: options.epochTopic ?? "tm_aria_epochs",
      apiKey: options.apiKey,
    });
    this.lookupService = new LookupService({
      baseUrl: options.baseUrl,
      serviceName: options.lookupService ?? "ls_aria",
      apiKey: options.apiKey,
    });
  }

  /** Submit an epoch transaction (OPEN or CLOSE) to the overlay network. */
  async submitEpoch(rawTx: string): Promise<AdmittanceResult> {
    return this.topicManager.submit(rawTx);
  }

  /**
   * Find epoch records for a system.
   *
   * @param systemId  ARIA system ID.
   * @param epochId   Optional specific epoch ID.
   * @param limit     Max results.
   */
  async findEpochs(
    systemId: string,
    epochId?: string,
    limit = 10
  ): Promise<LookupResult[]> {
    const query: Record<string, unknown> = { system_id: systemId };
    if (epochId !== undefined) query["epoch_id"] = epochId;
    return this.lookupService.lookup(query, limit);
  }

  /**
   * Find audit records for a specific epoch.
   */
  async findRecords(
    systemId: string,
    epochId: string,
    limit = 100
  ): Promise<LookupResult[]> {
    return this.lookupService.lookup(
      { system_id: systemId, epoch_id: epochId, type: "AUDIT_RECORD" },
      limit
    );
  }
}
