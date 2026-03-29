/**
 * aria-bsv/multimodel — Multi-model epoch auditor.
 *
 * A single EPOCH_OPEN commits to N models simultaneously.
 * The `multiModelRoot` is the Merkle root of all model hashes, providing a
 * single cryptographic commitment covering the entire model ensemble.
 */

import { hashObject } from "./hasher.js";
import { MerkleTree, computeMerkleRootSync } from "./merkle.js";
import type { AuditRecord, EpochOpenPayload, EpochClosePayload } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single model entry in the multi-model ensemble. */
export interface ModelEntry {
  modelId: string;
  /** SHA-256 hash of the serialised model file. */
  modelHash: string;
}

/**
 * EPOCH_OPEN payload for a multi-model epoch.
 * Extends {@link EpochOpenPayload} with the ensemble list and Merkle root.
 */
export interface MultiModelEpochOpen extends EpochOpenPayload {
  /** Ordered list of models committed in this epoch. */
  models: ModelEntry[];
  /**
   * Merkle root of all model hashes (BRC-121 algorithm with second-preimage
   * protection).  Allows a verifier to prove membership of any individual
   * model.
   */
  multiModelRoot: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BRC121_VERSION = "1.0";
const DEFAULT_BATCH_SIZE = 100;

// ---------------------------------------------------------------------------
// MultiModelAuditor
// ---------------------------------------------------------------------------

/**
 * Records AI inferences across a multi-model ensemble within a single epoch.
 *
 * @example
 * ```ts
 * const auditor = new MultiModelAuditor({
 *   systemId: "my-ensemble",
 *   models: [
 *     { modelId: "gpt-4o",    modelHash: "abc...def" },
 *     { modelId: "claude-3",  modelHash: "123...456" },
 *   ],
 * });
 * const open = await auditor.open();
 * await auditor.record("gpt-4o", inputData, outputData);
 * const { epochClose, records } = await auditor.flush();
 * ```
 */
export class MultiModelAuditor {
  private readonly _systemId: string;
  private readonly _models: ModelEntry[];
  private readonly _modelIds: Set<string>;
  private readonly _batchSize: number;

  private _records: AuditRecord[] = [];
  private _tree = new MerkleTree();
  private _epochId = "";
  private _epochSeq = 0;
  private _openPayload: MultiModelEpochOpen | null = null;

  constructor(config: {
    systemId: string;
    models: ModelEntry[];
    batchSize?: number;
  }) {
    this._systemId = config.systemId;
    this._models = config.models;
    this._modelIds = new Set(config.models.map((m) => m.modelId));
    this._batchSize = config.batchSize ?? DEFAULT_BATCH_SIZE;
  }

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Open a new multi-model epoch.
   *
   * Computes the `multiModelRoot` Merkle root over all registered model hashes
   * and returns the complete EPOCH_OPEN payload.
   *
   * @returns {@link MultiModelEpochOpen} payload ready for on-chain anchoring.
   */
  async open(): Promise<MultiModelEpochOpen> {
    this._epochId = _generateEpochId();
    this._epochSeq = 0;
    this._records = [];
    this._tree = new MerkleTree();

    // Compute Merkle root over all model hashes (BRC-121 algorithm)
    const modelHashes = this._models.map((m) => m.modelHash);
    const multiModelRoot = computeMerkleRootSync(modelHashes);

    // Build model_hashes map for the base EpochOpenPayload
    const model_hashes: Record<string, string> = {};
    for (const m of this._models) {
      model_hashes[m.modelId] = m.modelHash;
    }

    const payload: MultiModelEpochOpen = {
      type: "EPOCH_OPEN",
      brc121_version: BRC121_VERSION,
      system_id: this._systemId,
      epoch_id: this._epochId,
      opened_at: new Date().toISOString(),
      model_hashes,
      state_hash: "0".repeat(64),
      nonce: _randomHex(16),
      models: this._models,
      multiModelRoot,
    };

    this._openPayload = payload;
    return payload;
  }

  // -------------------------------------------------------------------------
  // Recording
  // -------------------------------------------------------------------------

  /**
   * Record a single AI inference for a model in the ensemble.
   *
   * @param modelId  Must be one of the models registered at construction.
   * @param input    Input data (hashed — never sent on-chain).
   * @param output   Output data (hashed — never sent on-chain).
   * @param options  Optional confidence, latency override, and metadata.
   * @returns The created {@link AuditRecord}.
   * @throws If `modelId` is not in the registered model list.
   */
  async record(
    modelId: string,
    input: unknown,
    output: unknown,
    options: {
      confidence?: number;
      latencyMs?: number;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<AuditRecord> {
    if (!this._epochId) {
      await this.open();
    }

    if (!this._modelIds.has(modelId)) {
      throw new Error(
        `Unknown modelId "${modelId}". Registered models: ${[...this._modelIds].join(", ")}`
      );
    }

    const t0 = Date.now();
    const [inputHash, outputHash] = await Promise.all([
      hashObject(input),
      hashObject(output),
    ]);
    const latency = options.latencyMs ?? Date.now() - t0;

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
      // Caller should flush; we do not auto-flush to keep this module simple
    }

    return rec;
  }

  // -------------------------------------------------------------------------
  // Flush
  // -------------------------------------------------------------------------

  /**
   * Finalise the current epoch.
   *
   * Computes the Merkle root over all recorded inferences and returns the
   * EPOCH_CLOSE payload along with the complete record list.
   *
   * @returns `{ epochClose, records }` — caller is responsible for anchoring.
   */
  async flush(): Promise<{ epochClose: EpochClosePayload; records: AuditRecord[] }> {
    const records = [...this._records];
    const merkle_root = await this._tree.root();

    const epochClose: EpochClosePayload = {
      type: "EPOCH_CLOSE",
      brc121_version: BRC121_VERSION,
      system_id: this._systemId,
      epoch_id: this._epochId || "no-epoch",
      prev_txid: "0".repeat(64),
      merkle_root,
      record_count: records.length,
      closed_at: new Date().toISOString(),
    };

    // Reset state
    this._records = [];
    this._tree = new MerkleTree();
    this._epochId = "";
    this._epochSeq = 0;
    this._openPayload = null;

    return { epochClose, records };
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  /** Return a snapshot of auditor statistics. */
  stats(): { buffered: number; models: string[]; epochId: string } {
    return {
      buffered: this._records.length,
      models: [...this._modelIds],
      epochId: this._epochId,
    };
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
  if (typeof globalThis !== "undefined" && globalThis.crypto?.getRandomValues) {
    const buf = new Uint8Array(bytes);
    globalThis.crypto.getRandomValues(buf);
    return Array.from(buf)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
  if (typeof require !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { randomBytes } = require("crypto") as typeof import("crypto");
    return randomBytes(bytes).toString("hex");
  }
  return Array.from({ length: bytes }, () =>
    Math.floor(Math.random() * 256)
      .toString(16)
      .padStart(2, "0")
  ).join("");
}
