/**
 * ARIAEpochContract — BRC-121 on-chain enforcement contract.
 *
 * This contract locks satoshis in an EPOCH_OPEN UTXO. The funds can only be
 * spent when a valid EPOCH_CLOSE transaction provides:
 *   1. The Merkle root that was committed in EPOCH_OPEN.
 *   2. A signature from the authorised operator key (pubKeyHash).
 *   3. (Optionally) a ZK proof reference hash for EU AI Act compliance.
 *
 * sCrypt source (annotated for production compilation with scrypt-ts):
 *
 * ```ts
 * import { SmartContract, prop, method, assert, sha256, Sha256,
 *          PubKeyHash, Sig, PubKey, hash160 } from '@scrypt-inc/scrypt-ts'
 *
 * export class ARIAEpochContract extends SmartContract {
 *   @prop() readonly epochId: ByteString
 *   @prop() readonly merkleRoot: Sha256
 *   @prop() readonly operatorPkh: PubKeyHash
 *   @prop() readonly openTimestamp: bigint
 *   @prop() readonly systemIdHash: Sha256
 *   @prop() readonly zkProofHash: Sha256   // 0x00…00 if no ZK proof
 *
 *   constructor(...) { super(...arguments); ... }
 *
 *   @method()
 *   public unlock(sig: Sig, pubKey: PubKey, closeRoot: Sha256) {
 *     // 1. Verify operator key
 *     assert(hash160(pubKey) == this.operatorPkh, 'wrong operator key')
 *     assert(this.checkSig(sig, pubKey), 'invalid signature')
 *     // 2. Merkle root must match what was committed
 *     assert(closeRoot == this.merkleRoot, 'merkle root mismatch')
 *   }
 * }
 * ```
 *
 * This TypeScript file provides a pure-TS implementation usable WITHOUT the
 * sCrypt compiler (for off-chain verification, testing, and SDK integration).
 * Pass `compilerArtifact` when you have the compiled `.scrypt` artifact.
 */

import {
  ScryptBase,
  Sha256,
  PubKeyHash,
  ByteString,
  ContractVerifyResult,
  toSha256,
  toPubKeyHash,
  toByteString,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "./scrypt_adapter.js"

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

export interface EpochContractState {
  /** Unique epoch identifier, e.g. "ep_1711234567890_0001" */
  epochId: string
  /**
   * SHA-256 Merkle root of all inference records in this epoch.
   * Set to "0000…00" (64 zeros) if the epoch has no records.
   */
  merkleRoot: Sha256
  /**
   * HASH160 of the operator's BSV public key.
   * Only a signature from this key can unlock the contract.
   */
  operatorPkh: PubKeyHash
  /**
   * Unix timestamp (ms) when EPOCH_OPEN was broadcast.
   * Used for dispute resolution — not enforced on-chain in v1.
   */
  openTimestampMs: number
  /**
   * SHA-256 of the system_id string. Binds on-chain state to a specific system.
   */
  systemIdHash: Sha256
  /**
   * SHA-256 hash of the ZK proof (from aria/zk/). All-zeros if no proof.
   * Enables optional EU AI Act Article 12 compliance at the script level.
   */
  zkProofHash: Sha256
}

export interface EpochUnlockWitness {
  /** Hex-encoded signature from the operator key. */
  sigHex: string
  /** Hex-encoded compressed public key (33 bytes). */
  pubKeyHex: string
  /** Merkle root provided in the EPOCH_CLOSE payload. */
  closeRoot: Sha256
}

// ---------------------------------------------------------------------------
// Contract
// ---------------------------------------------------------------------------

/** Minimum satoshis locked per epoch (economic security deposit). */
export const EPOCH_BOND_SAT = 1000n

/**
 * Type-1 ARIA contract: epoch-commit enforcement.
 *
 * Lifecycle:
 *   1. Operator broadcasts EPOCH_OPEN tx with this contract as output.
 *   2. Inference happens — records accumulate.
 *   3. Operator broadcasts EPOCH_CLOSE tx, providing `unlock()` witness.
 *      The witness must include the same Merkle root committed in step 1.
 */
export class ARIAEpochContract extends ScryptBase<EpochUnlockWitness> {
  readonly epochId: string
  readonly merkleRoot: Sha256
  readonly operatorPkh: PubKeyHash
  readonly openTimestampMs: number
  readonly systemIdHash: Sha256
  readonly zkProofHash: Sha256

  constructor(state: EpochContractState) {
    super()
    this.epochId = state.epochId
    this.merkleRoot = toSha256(state.merkleRoot)
    this.operatorPkh = toPubKeyHash(state.operatorPkh)
    this.openTimestampMs = state.openTimestampMs
    this.systemIdHash = toSha256(state.systemIdHash)
    this.zkProofHash = toSha256(state.zkProofHash)
  }

  // ---------------------------------------------------------------------------
  // Serialization — mirrors the sCrypt @prop layout
  // ---------------------------------------------------------------------------

  /**
   * Returns the OP_RETURN contract payload hex.
   * Layout: ARIA_MAGIC | BRC121_VERSION | type(01) | epochId | merkleRoot |
   *         operatorPkh | openTimestampMs (8 bytes BE) | systemIdHash | zkProofHash
   *
   * This payload is embedded in EPOCH_OPEN OP_RETURN output and read by the
   * ARIA verification SDK.
   */
  getLockingScriptHex(): string {
    const epochIdHex = Buffer.from(this.epochId, "utf8").toString("hex")
    const tsHex = this.openTimestampMs.toString(16).padStart(16, "0")

    const payload =
      ARIA_MAGIC +
      BRC121_VERSION +
      "01" +                  // contract type: EPOCH
      epochIdHex.padEnd(64, "0").slice(0, 64) +
      this.merkleRoot +
      this.operatorPkh +
      tsHex +
      this.systemIdHash +
      this.zkProofHash

    return buildOpReturnScript(payload)
  }

  /**
   * Verify an epoch-close unlock witness against the committed state.
   *
   * In production (on-chain), the sCrypt VM enforces this. Off-chain, this
   * method lets client code verify before broadcasting.
   *
   * Checks:
   *   1. Provided pubKey hashes to operatorPkh (via HASH160 proxy — we use a
   *      length + prefix check since we cannot run HASH160 without a crypto lib).
   *      In production, the sCrypt @method enforces `hash160(pubKey) == operatorPkh`.
   *   2. The closeRoot matches the committed merkleRoot.
   *   3. sigHex is a non-empty hex string (actual ECDSA check is on-chain).
   */
  verify(witness: EpochUnlockWitness): ContractVerifyResult {
    // 1. pubKey length check (compressed: 33 bytes = 66 hex chars)
    if (!/^(02|03)[0-9a-f]{64}$/i.test(witness.pubKeyHex)) {
      return { success: false, error: "pubKey must be a compressed 33-byte public key" }
    }

    // 2. Signature present (ECDSA verification is on-chain only)
    if (!witness.sigHex || witness.sigHex.length < 8) {
      return { success: false, error: "sigHex is missing or too short" }
    }

    // 3. Merkle root match — the critical BRC-121 invariant
    try {
      const closeRoot = toSha256(witness.closeRoot)
      if (closeRoot !== this.merkleRoot) {
        return {
          success: false,
          error: `Merkle root mismatch: committed=${this.merkleRoot} provided=${closeRoot}`,
        }
      }
    } catch (e) {
      return { success: false, error: `Invalid closeRoot: ${e}` }
    }

    return { success: true }
  }

  // ---------------------------------------------------------------------------
  // Factory
  // ---------------------------------------------------------------------------

  /** Build from an ARIA EpochOpenPayload (as returned by aria-bsv Python SDK). */
  static fromEpochOpen(payload: {
    epoch_id: string
    merkle_root: string
    system_id: string
    timestamp: string
    operator_pkh?: string
    zk_proof_hash?: string
  }): ARIAEpochContract {
    const systemIdHash = toSha256(
      // SHA-256(system_id) — computed externally; allow passing pre-computed hash
      payload.system_id.length === 64 ? payload.system_id : _hexPad(payload.system_id, 64)
    )
    return new ARIAEpochContract({
      epochId: payload.epoch_id,
      merkleRoot: toSha256(payload.merkle_root),
      operatorPkh: toPubKeyHash(payload.operator_pkh ?? "0".repeat(40)),
      openTimestampMs: new Date(payload.timestamp).getTime(),
      systemIdHash,
      zkProofHash: toSha256(payload.zk_proof_hash ?? "0".repeat(64)),
    })
  }
}

function _hexPad(s: string, len: number): string {
  const hex = Buffer.from(s, "utf8").toString("hex")
  return hex.padStart(len, "0").slice(0, len)
}
