/**
 * aria-bsv/dataset — Dataset anchoring to BSV.
 *
 * Mirrors ``aria.dataset`` from the Python SDK.
 * Hashes a dataset (bytes, text, JSON, or File) and broadcasts a
 * ``DATASET_ANCHOR`` OP_RETURN payload to BSV via the ARC broadcaster.
 *
 * On-chain payload:
 * ```json
 * {
 *   "type": "DATASET_ANCHOR",
 *   "brc121_version": "1.0",
 *   "system_id": "...",
 *   "dataset_id": "<uuid>",
 *   "content_hash": "sha256:<64hex>",
 *   "schema_hash": "sha256:<64hex>|null",
 *   "row_count": <int>|null,
 *   "column_names": [...]|null,
 *   "media_type": "<mime>",
 *   "anchored_at": "<ISO-8601 UTC>",
 *   "nonce": "<32hex>"
 * }
 * ```
 */

import { sha256HexSyncFromBytes } from "./hasher.js";
import { canonicalJson } from "./hasher.js";
import { ARCBroadcaster } from "./broadcaster.js";
import type { BroadcasterOptions } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DatasetAnchorPayload {
  type: "DATASET_ANCHOR";
  brc121_version: string;
  system_id: string;
  dataset_id: string;
  content_hash: string;
  schema_hash: string | null;
  row_count: number | null;
  column_names: string[] | null;
  media_type: string;
  anchored_at: string;
  nonce: string;
}

export interface DatasetAnchor {
  dataset_id: string;
  system_id: string;
  content_hash: string;
  schema_hash: string | null;
  row_count: number | null;
  column_names: string[] | null;
  media_type: string;
  anchored_at: string;
  txid: string;
  payload: DatasetAnchorPayload;
}

export interface AnchorOptions {
  media_type?: string;
  row_count?: number | null;
  column_names?: string[] | null;
}

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

/**
 * Return ``sha256:<hex>`` of raw bytes.
 */
export function hashBytes(data: Uint8Array): string {
  return "sha256:" + sha256HexSyncFromBytes(data);
}

/**
 * Return ``sha256:<hex>`` of the sorted column names list.
 * Order-independent: ``["b","a"]`` === ``["a","b"]``.
 */
export function hashColumns(columnNames: string[]): string {
  const sorted = [...columnNames].sort();
  const canonical = canonicalJson(sorted);
  const bytes = typeof canonical === "string"
    ? new TextEncoder().encode(canonical)
    : canonical as Uint8Array;
  return "sha256:" + sha256HexSyncFromBytes(bytes);
}

// ---------------------------------------------------------------------------
// DatasetAnchorer
// ---------------------------------------------------------------------------

export interface DatasetAnchorerOptions {
  system_id: string;
  broadcaster?: BroadcasterOptions;
}

/**
 * Anchors dataset hashes to BSV via OP_RETURN transactions.
 *
 * @example
 * ```ts
 * const anchorer = new DatasetAnchorer({ system_id: "my-system", broadcaster: { apiKey: "..." } });
 * const anchor = await anchorer.anchorBytes(csvBytes, { media_type: "text/csv" });
 * console.log(anchor.txid, anchor.content_hash);
 * ```
 */
export class DatasetAnchorer {
  private readonly _systemId: string;
  private readonly _broadcaster: ARCBroadcaster | null;

  constructor(options: DatasetAnchorerOptions) {
    this._systemId = options.system_id;
    this._broadcaster = options.broadcaster
      ? new ARCBroadcaster(options.broadcaster)
      : null;
  }

  /**
   * Anchor raw bytes to BSV.
   */
  async anchorBytes(
    data: Uint8Array,
    options: AnchorOptions = {}
  ): Promise<DatasetAnchor> {
    return this._anchor(data, options.media_type ?? "application/octet-stream", options);
  }

  /**
   * Anchor a UTF-8 text string (or CSV content) to BSV.
   */
  async anchorText(
    text: string,
    options: AnchorOptions & { encoding?: string; media_type?: string } = {}
  ): Promise<DatasetAnchor> {
    const bytes = new TextEncoder().encode(text);
    return this._anchor(bytes, options.media_type ?? "text/plain", options);
  }

  /**
   * Anchor a JSON-serialisable object to BSV.
   * Uses canonical JSON (sorted keys) so dict key order does not affect the hash.
   */
  async anchorJson(
    obj: unknown,
    options: AnchorOptions = {}
  ): Promise<DatasetAnchor> {
    const canonical = canonicalJson(obj);
    const bytes = typeof canonical === "string"
      ? new TextEncoder().encode(canonical)
      : canonical as Uint8Array;
    return this._anchor(bytes, "application/json", options);
  }

  /**
   * Anchor a File or Blob (browser) or Buffer (Node.js) to BSV.
   */
  async anchorFile(
    file: File | Blob | ArrayBuffer,
    options: AnchorOptions & { filename?: string } = {}
  ): Promise<DatasetAnchor> {
    let bytes: Uint8Array;
    if (file instanceof ArrayBuffer) {
      bytes = new Uint8Array(file);
    } else {
      bytes = new Uint8Array(await (file as Blob).arrayBuffer());
    }
    const mediaType = options.media_type
      ?? (file instanceof File ? _guessMimeFromName(file.name) : "application/octet-stream");
    return this._anchor(bytes, mediaType, options);
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  private async _anchor(
    data: Uint8Array,
    mediaType: string,
    options: AnchorOptions
  ): Promise<DatasetAnchor> {
    const contentHash = hashBytes(data);
    const schemaHash = options.column_names?.length
      ? hashColumns(options.column_names)
      : null;

    const datasetId = _uuid4();
    const nonce = _randomHex(16);
    const anchoredAt = new Date().toISOString();

    const payload: DatasetAnchorPayload = {
      type: "DATASET_ANCHOR",
      brc121_version: "1.0",
      system_id: this._systemId,
      dataset_id: datasetId,
      content_hash: contentHash,
      schema_hash: schemaHash,
      row_count: options.row_count ?? null,
      column_names: options.column_names ?? null,
      media_type: mediaType,
      anchored_at: anchoredAt,
      nonce,
    };

    let txid = "";
    if (this._broadcaster) {
      const status = await this._broadcaster.broadcast(JSON.stringify(payload));
      txid = status.txid;
    }

    return {
      dataset_id: datasetId,
      system_id: this._systemId,
      content_hash: contentHash,
      schema_hash: schemaHash,
      row_count: options.row_count ?? null,
      column_names: options.column_names ?? null,
      media_type: mediaType,
      anchored_at: anchoredAt,
      txid,
      payload,
    };
  }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/**
 * Verify that *data* matches the ``content_hash`` stored in *anchor*.
 *
 * @returns ``true`` if ``SHA-256(data) === anchor.content_hash``.
 */
export function verifyDatasetAnchor(
  data: Uint8Array,
  anchor: DatasetAnchor
): boolean {
  return hashBytes(data) === anchor.content_hash;
}

// ---------------------------------------------------------------------------
// MIME type guessing
// ---------------------------------------------------------------------------

const _MIME_MAP: Record<string, string> = {
  ".csv": "text/csv",
  ".tsv": "text/tab-separated-values",
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".parquet": "application/vnd.apache.parquet",
  ".arrow": "application/vnd.apache.arrow.file",
  ".txt": "text/plain",
  ".xml": "application/xml",
  ".zip": "application/zip",
  ".gz": "application/gzip",
};

function _guessMimeFromName(filename: string): string {
  const dot = filename.lastIndexOf(".");
  if (dot === -1) return "application/octet-stream";
  const ext = filename.slice(dot).toLowerCase();
  return _MIME_MAP[ext] ?? "application/octet-stream";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _uuid4(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function _randomHex(bytes: number): string {
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    const buf = new Uint8Array(bytes);
    crypto.getRandomValues(buf);
    return Array.from(buf).map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  if (typeof require !== "undefined") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { randomBytes } = require("crypto") as typeof import("crypto");
    return randomBytes(bytes).toString("hex");
  }
  return Array.from({ length: bytes }, () =>
    Math.floor(Math.random() * 256).toString(16).padStart(2, "0")
  ).join("");
}
