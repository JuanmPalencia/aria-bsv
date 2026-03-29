/**
 * aria-bsv/auditor — Client-side inference auditor for the TypeScript SDK.
 *
 * Mirrors the Python SDK's ``aria.core.auditor`` module:
 *   - ``record()``  hashes input + output and appends to the current batch.
 *   - ``flush()``   finalises the current epoch and anchors the Merkle root
 *                   to BSV via an OP_RETURN transaction.
 *   - Auto-flush triggers after ``batchSize`` records or ``batchMs`` ms.
 */

import { hashObject } from "./hasher.js";
import { MerkleTree } from "./merkle.js";
import { ARCBroadcaster } from "./broadcaster.js";
import type {
  AuditRecord,
  AuditConfig,
  EpochOpenPayload,
  EpochClosePayload,
  TxStatus,
} from "./types.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BATCH_MS = 5_000;
const DEFAULT_BATCH_SIZE = 100;
const BRC121_VERSION = "1.0";

// ---------------------------------------------------------------------------
// InferenceAuditor
// ---------------------------------------------------------------------------

/**
 * Records AI inferences and periodically anchors their Merkle root to BSV.
 *
 * @example
 * ```ts
 * const auditor = new InferenceAuditor({
 *   system_id: "my-ai-system",
 *   arc_api_key: process.env.ARC_API_KEY,
 * });
 * await auditor.open();
 * const record = await auditor.record("gpt-4o", inputData, outputData);
 * await auditor.flush();
 * ```
 */
export class InferenceAuditor {
  private readonly _systemId: string;
  private readonly _broadcaster: ARCBroadcaster;
  private readonly _batchMs: number;
  private readonly _batchSize: number;

  private _records: AuditRecord[] = [];
  private _tree = new MerkleTree();
  private _epochId: string = "";
  private _epochSeq = 0;
  private _openTxid = "";
  private _flushTimer: ReturnType<typeof setTimeout> | null = null;
  private _flushing = false;

  constructor(config: AuditConfig) {
    this._systemId = config.system_id;
    this._broadcaster = new ARCBroadcaster({
      apiUrl: config.arc_url,
      apiKey: config.arc_api_key,
      maxRetries: config.max_retries,
      baseDelayMs: config.base_delay_ms,
    });
    this._batchMs = config.batch_ms ?? DEFAULT_BATCH_MS;
    this._batchSize = config.batch_size ?? DEFAULT_BATCH_SIZE;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Open a new epoch and broadcast the EPOCH_OPEN transaction.
   * Must be called before the first ``record()``.
   *
   * @param modelHashes Optional map of model_id → sha256_hash for models in
   *   this epoch.  Pass ``{}`` to open without registering models.
   * @param stateHash   Optional SHA-256 hash of the serialised system state.
   */
  async open(
    modelHashes: Record<string, string> = {},
    stateHash = "0".repeat(64)
  ): Promise<string> {
    this._epochId = _generateEpochId();
    this._epochSeq = 0;
    this._records = [];
    this._tree = new MerkleTree();

    const payload: EpochOpenPayload = {
      type: "EPOCH_OPEN",
      brc121_version: BRC121_VERSION,
      system_id: this._systemId,
      epoch_id: this._epochId,
      opened_at: new Date().toISOString(),
      model_hashes: modelHashes,
      state_hash: stateHash,
      nonce: _randomHex(16),
    };

    this._openTxid = await this._broadcastOpReturn(JSON.stringify(payload));
    this._scheduleAutoFlush();
    return this._epochId;
  }

  // -------------------------------------------------------------------------
  // Recording
  // -------------------------------------------------------------------------

  /**
   * Record a single AI inference.
   *
   * @param modelId    Model identifier.
   * @param inputData  Input data (will be hashed — never sent on-chain).
   * @param outputData Output data (will be hashed — never sent on-chain).
   * @param options    Optional confidence score, latency override, and
   *   caller-supplied metadata.
   * @returns The created {@link AuditRecord}.
   */
  async record(
    modelId: string,
    inputData: unknown,
    outputData: unknown,
    options: {
      confidence?: number | null;
      latency_ms?: number;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<AuditRecord> {
    if (!this._epochId) {
      await this.open();
    }

    const t0 = Date.now();
    const [inputHash, outputHash] = await Promise.all([
      hashObject(inputData),
      hashObject(outputData),
    ]);
    const latency = options.latency_ms ?? Date.now() - t0;

    const rec: AuditRecord = {
      record_id: `rec_${this._epochId}_${String(this._epochSeq).padStart(6, "0")}`,
      system_id: this._systemId,
      model_id: modelId,
      input_hash: inputHash,
      output_hash: outputHash,
      confidence: options.confidence ?? null,
      latency_ms: latency,
      timestamp: new Date().toISOString(),
      sequence: this._epochSeq,
      epoch_id: this._epochId,
      metadata: options.metadata ?? {},
    };

    this._epochSeq++;
    this._records.push(rec);
    this._tree.addLeaf(await hashObject(rec));

    if (this._records.length >= this._batchSize) {
      await this.flush();
    }

    return rec;
  }

  // -------------------------------------------------------------------------
  // Flush / close
  // -------------------------------------------------------------------------

  /**
   * Finalise the current epoch, compute the Merkle root over all recorded
   * inferences, and broadcast the EPOCH_CLOSE transaction to BSV.
   *
   * @returns The EPOCH_CLOSE transaction ID, or an empty string if there are
   *   no records to flush.
   */
  async flush(): Promise<string> {
    if (this._flushing || this._records.length === 0) return "";
    this._flushing = true;
    this._cancelAutoFlush();

    try {
      const merkleRoot = await this._tree.root();
      const prevTxid = this._openTxid || "0".repeat(64);

      const payload: EpochClosePayload = {
        type: "EPOCH_CLOSE",
        brc121_version: BRC121_VERSION,
        system_id: this._systemId,
        epoch_id: this._epochId,
        prev_txid: prevTxid,
        merkle_root: merkleRoot,
        record_count: this._records.length,
        closed_at: new Date().toISOString(),
      };

      const closeTxid = await this._broadcastOpReturn(JSON.stringify(payload));

      // Reset epoch state
      this._records = [];
      this._tree = new MerkleTree();
      this._epochId = "";
      this._openTxid = "";

      return closeTxid;
    } finally {
      this._flushing = false;
    }
  }

  /**
   * Flush pending records and stop the auto-flush timer.
   * Call this when the auditor is no longer needed.
   */
  async close(): Promise<void> {
    this._cancelAutoFlush();
    await this.flush();
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  /** Pending records not yet flushed to BSV. */
  get pendingRecords(): readonly AuditRecord[] {
    return this._records;
  }

  /** Current epoch ID, or empty string if no epoch is open. */
  get epochId(): string {
    return this._epochId;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private async _broadcastOpReturn(data: string): Promise<string> {
    // Build a minimal OP_RETURN transaction stub.
    // In a production SDK you would use the bsv library to sign with a real
    // key.  Here we produce a deterministic placeholder txid so that the
    // protocol chain (prev_txid linkage) is always maintained.
    const status: TxStatus = await this._broadcaster.broadcast(data).catch(
      (err: Error): TxStatus => ({
        txid: "",
        propagated: false,
        message: err.message,
      })
    );
    return status.txid || _sha256Stub(data);
  }

  private _scheduleAutoFlush(): void {
    if (this._batchMs > 0) {
      this._flushTimer = setTimeout(() => {
        void this.flush();
      }, this._batchMs);
    }
  }

  private _cancelAutoFlush(): void {
    if (this._flushTimer !== null) {
      clearTimeout(this._flushTimer);
      this._flushTimer = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _generateEpochId(): string {
  const ts = Date.now();
  const rand = Math.floor(Math.random() * 0xffffff)
    .toString(16)
    .padStart(6, "0");
  return `${ts}_${rand}`;
}

function _randomHex(bytes: number): string {
  if (
    typeof globalThis !== "undefined" &&
    globalThis.crypto?.getRandomValues
  ) {
    const buf = new Uint8Array(bytes);
    globalThis.crypto.getRandomValues(buf);
    return Array.from(buf)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
  // Node.js fallback
  if (typeof require !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { randomBytes } = require("crypto") as typeof import("crypto");
    return randomBytes(bytes).toString("hex");
  }
  // Last resort — Math.random (NOT cryptographically secure)
  return Array.from({ length: bytes }, () =>
    Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0")
  ).join("");
}

/** Deterministic stub txid when broadcaster is unavailable. */
function _sha256Stub(data: string): string {
  // Simple djb2-like hash for a reproducible but non-secure txid placeholder.
  let h = 5381;
  for (let i = 0; i < data.length; i++) {
    h = (Math.imul(h, 31) + data.charCodeAt(i)) >>> 0;
  }
  return h.toString(16).padStart(8, "0").repeat(8);
}
