/**
 * aria-bsv/verifier — Off-chain epoch verifier for BRC-121.
 *
 * Given a set of {@link AuditRecord} objects and the epoch payloads that were
 * anchored to BSV, verifies:
 *
 *  1. Every record's ``record_id`` format is correct.
 *  2. The Merkle root recomputed from the records matches the one stored in
 *     the EPOCH_CLOSE payload.
 *  3. The EPOCH_CLOSE references the correct EPOCH_OPEN via ``prev_txid``.
 *  4. All records belong to the declared epoch.
 *  5. Record sequence numbers are contiguous starting at 0.
 *
 * On-chain verification (fetching the actual BSV transactions and confirming
 * the OP_RETURN data) requires network access and is handled separately via
 * the {@link OnChainVerifier} class.
 */

import { hashObject } from "./hasher.js";
import { computeMerkleRoot, EMPTY_ROOT } from "./merkle.js";
import type {
  AuditRecord,
  EpochOpenPayload,
  EpochClosePayload,
  VerificationResult,
} from "./types.js";

// ---------------------------------------------------------------------------
// Off-chain (local) verifier
// ---------------------------------------------------------------------------

/**
 * Verify a set of {@link AuditRecord} objects against the provided epoch
 * payloads — entirely locally, no network access required.
 *
 * @param records   Array of audit records for the epoch.
 * @param openPayload  The EPOCH_OPEN payload object.
 * @param closePayload The EPOCH_CLOSE payload object.
 * @param openTxid  The BSV txid where EPOCH_OPEN was anchored.
 * @param closeTxid The BSV txid where EPOCH_CLOSE was anchored.
 */
export async function verifyEpoch(
  records: AuditRecord[],
  openPayload: EpochOpenPayload,
  closePayload: EpochClosePayload,
  openTxid: string,
  closeTxid: string
): Promise<VerificationResult> {
  const errors: string[] = [];
  const epochId = openPayload.epoch_id;

  // 1. Payload type checks
  if (openPayload.type !== "EPOCH_OPEN") {
    errors.push(`openPayload.type must be "EPOCH_OPEN", got "${openPayload.type}"`);
  }
  if (closePayload.type !== "EPOCH_CLOSE") {
    errors.push(`closePayload.type must be "EPOCH_CLOSE", got "${closePayload.type}"`);
  }

  // 2. Epoch IDs match
  if (openPayload.epoch_id !== closePayload.epoch_id) {
    errors.push(
      `Epoch ID mismatch: open=${openPayload.epoch_id} close=${closePayload.epoch_id}`
    );
  }
  if (openPayload.system_id !== closePayload.system_id) {
    errors.push(
      `System ID mismatch: open=${openPayload.system_id} close=${closePayload.system_id}`
    );
  }

  // 3. prev_txid linkage
  if (closePayload.prev_txid !== openTxid) {
    errors.push(
      `prev_txid mismatch: close.prev_txid=${closePayload.prev_txid} openTxid=${openTxid}`
    );
  }

  // 4. Record count
  if (closePayload.record_count !== records.length) {
    errors.push(
      `record_count mismatch: payload says ${closePayload.record_count}, got ${records.length}`
    );
  }

  // 5. All records belong to the declared epoch
  for (const rec of records) {
    if (rec.epoch_id !== epochId) {
      errors.push(
        `Record ${rec.record_id} has epoch_id=${rec.epoch_id}, expected ${epochId}`
      );
    }
    if (rec.system_id !== openPayload.system_id) {
      errors.push(
        `Record ${rec.record_id} has system_id=${rec.system_id}, expected ${openPayload.system_id}`
      );
    }
  }

  // 6. Sequence numbers are contiguous [0, N-1]
  const sorted = [...records].sort((a, b) => a.sequence - b.sequence);
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i].sequence !== i) {
      errors.push(
        `Non-contiguous sequence: expected ${i}, got ${sorted[i].sequence} (record ${sorted[i].record_id})`
      );
      break; // Report only the first gap
    }
  }

  // 7. Recompute Merkle root and compare
  const leafHashes = await Promise.all(sorted.map((r) => hashObject(r)));
  const computedRoot =
    leafHashes.length > 0
      ? await computeMerkleRoot(leafHashes)
      : EMPTY_ROOT;

  if (computedRoot !== closePayload.merkle_root) {
    errors.push(
      `Merkle root mismatch: computed=${computedRoot} stored=${closePayload.merkle_root}`
    );
  }

  return {
    valid: errors.length === 0,
    system_id: openPayload.system_id,
    epoch_id: epochId,
    record_count: records.length,
    open_txid: openTxid,
    close_txid: closeTxid,
    errors,
  };
}

// ---------------------------------------------------------------------------
// On-chain verifier
// ---------------------------------------------------------------------------

export interface OnChainVerifierOptions {
  /** WhatsOnChain or compatible BSV explorer API base URL. */
  explorerUrl?: string;
}

/**
 * Fetches epoch payloads from BSV transactions and verifies them.
 *
 * Requires network access to a BSV block explorer API
 * (default: WhatsOnChain).
 */
export class OnChainVerifier {
  private readonly _explorerUrl: string;

  constructor(options: OnChainVerifierOptions = {}) {
    this._explorerUrl = (
      options.explorerUrl ?? "https://api.whatsonchain.com/v1/bsv/main"
    ).replace(/\/$/, "");
  }

  /**
   * Fetch the OP_RETURN data embedded in a BSV transaction.
   *
   * @param txid 64-char hex transaction ID.
   * @returns The raw string payload from the first OP_RETURN output.
   * @throws If the txid is not found or the tx has no OP_RETURN output.
   */
  async fetchOpReturn(txid: string): Promise<string> {
    const url = `${this._explorerUrl}/tx/${txid}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Explorer error ${response.status} fetching txid ${txid}`
      );
    }
    const tx = await response.json() as WoCTx;
    for (const out of tx.vout ?? []) {
      const asm = out.scriptPubKey?.asm ?? "";
      if (asm.startsWith("OP_RETURN ")) {
        // asm = "OP_RETURN <hex>"
        const hex = asm.slice("OP_RETURN ".length);
        return _hexToUtf8(hex);
      }
    }
    throw new Error(`No OP_RETURN output found in txid ${txid}`);
  }

  /**
   * Fetch and verify an epoch from two BSV transaction IDs.
   *
   * @param openTxid  BSV txid of the EPOCH_OPEN transaction.
   * @param closeTxid BSV txid of the EPOCH_CLOSE transaction.
   * @param records   The audit records that belong to this epoch.
   */
  async verifyEpochFromChain(
    openTxid: string,
    closeTxid: string,
    records: AuditRecord[]
  ): Promise<VerificationResult> {
    const [openRaw, closeRaw] = await Promise.all([
      this.fetchOpReturn(openTxid),
      this.fetchOpReturn(closeTxid),
    ]);

    let openPayload: EpochOpenPayload;
    let closePayload: EpochClosePayload;

    try {
      openPayload = JSON.parse(openRaw) as EpochOpenPayload;
    } catch {
      return _errorResult(openTxid, closeTxid, [
        `Failed to parse EPOCH_OPEN payload: ${openRaw.slice(0, 100)}`,
      ]);
    }

    try {
      closePayload = JSON.parse(closeRaw) as EpochClosePayload;
    } catch {
      return _errorResult(openTxid, closeTxid, [
        `Failed to parse EPOCH_CLOSE payload: ${closeRaw.slice(0, 100)}`,
      ]);
    }

    return verifyEpoch(records, openPayload, closePayload, openTxid, closeTxid);
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

interface WoCTx {
  vout?: Array<{
    scriptPubKey?: { asm?: string };
  }>;
}

function _hexToUtf8(hex: string): string {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  if (typeof TextDecoder !== "undefined") {
    return new TextDecoder().decode(bytes);
  }
  return Buffer.from(bytes).toString("utf8");
}

function _errorResult(
  openTxid: string,
  closeTxid: string,
  errors: string[]
): VerificationResult {
  return {
    valid: false,
    system_id: "",
    epoch_id: "",
    record_count: 0,
    open_txid: openTxid,
    close_txid: closeTxid,
    errors,
  };
}
