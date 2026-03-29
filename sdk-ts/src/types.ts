/**
 * aria-bsv/types — Core type definitions for the ARIA TypeScript SDK.
 *
 * All types mirror the Python SDK's data model exactly so that records
 * produced by the Python SDK can be verified by this SDK and vice-versa.
 */

// ---------------------------------------------------------------------------
// AuditRecord
// ---------------------------------------------------------------------------

/** A single recorded inference, anchored via the ARIA protocol. */
export interface AuditRecord {
  /** Unique identifier: ``rec_{epoch_id}_{seq:06d}`` */
  record_id: string;
  /** The ARIA system that produced this record. */
  system_id: string;
  /** Model identifier registered in the auditor. */
  model_id: string;
  /** SHA-256 hash of the canonical input. */
  input_hash: string;
  /** SHA-256 hash of the canonical output. */
  output_hash: string;
  /** Model confidence score in [0.0, 1.0], or null if unavailable. */
  confidence: number | null;
  /** Inference wall-clock duration in milliseconds. */
  latency_ms: number;
  /** ISO-8601 UTC timestamp of the inference. */
  timestamp: string;
  /** Sequence number within the current epoch (0-based). */
  sequence: number;
  /** Epoch this record belongs to. */
  epoch_id: string;
  /** Additional caller-supplied metadata. */
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Epoch payloads
// ---------------------------------------------------------------------------

/** BSV OP_RETURN payload for EPOCH_OPEN transactions. */
export interface EpochOpenPayload {
  type: "EPOCH_OPEN";
  /** BRC-121 protocol version. */
  brc121_version: string;
  system_id: string;
  epoch_id: string;
  /** ISO-8601 UTC timestamp when the epoch was opened. */
  opened_at: string;
  /** SHA-256 hashes of the registered AI models. */
  model_hashes: Record<string, string>;
  /** SHA-256 hash of the serialised system state. */
  state_hash: string;
  /** 16-byte random nonce (hex-encoded) for replay protection. */
  nonce: string;
}

/** BSV OP_RETURN payload for EPOCH_CLOSE transactions. */
export interface EpochClosePayload {
  type: "EPOCH_CLOSE";
  brc121_version: string;
  system_id: string;
  epoch_id: string;
  /** BSV txid of the corresponding EPOCH_OPEN transaction. */
  prev_txid: string;
  /** Merkle root of all AuditRecord hashes in this epoch. */
  merkle_root: string;
  /** Number of records in this epoch. */
  record_count: number;
  /** ISO-8601 UTC timestamp when the epoch was closed. */
  closed_at: string;
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

export interface VerificationResult {
  valid: boolean;
  system_id: string;
  epoch_id: string;
  record_count: number;
  open_txid: string;
  close_txid: string;
  errors: string[];
}

// ---------------------------------------------------------------------------
// Broadcaster
// ---------------------------------------------------------------------------

export interface TxStatus {
  txid: string;
  propagated: boolean;
  message: string;
}

export interface BroadcasterOptions {
  apiUrl?: string;
  apiKey?: string;
  maxRetries?: number;
  baseDelayMs?: number;
}

// ---------------------------------------------------------------------------
// Auditor config
// ---------------------------------------------------------------------------

export interface AuditConfig {
  system_id: string;
  arc_url?: string;
  arc_api_key?: string;
  batch_ms?: number;
  batch_size?: number;
  /** Override max broadcast retries (default 3). Set to 0 in tests. */
  max_retries?: number;
  /** Override base retry delay in ms (default 500). */
  base_delay_ms?: number;
}
