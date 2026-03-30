/**
 * ARIATimelockContract — Time-locked ARIA commitment contract.
 *
 * Locks satoshis until two conditions are BOTH satisfied:
 *   1. BSV block height ≥ unlockHeight   (nLockTime / CHECKLOCKTIMEVERIFY)
 *   2. The preimage of an ARIA commitment hash is provided.
 *
 * This provides a "dead man's switch" pattern for AI systems: if the operator
 * fails to close the epoch (e.g. system outage), the funds are recoverable
 * only after the timelock expires, preventing operators from abandoning
 * commitments silently.
 *
 * sCrypt source (for production compilation):
 *
 * ```ts
 * import { SmartContract, prop, method, assert, sha256, Sha256,
 *          PubKeyHash, Sig, PubKey, hash160, SigHash,
 *          MethodCallOptions, ContractTransaction } from '@scrypt-inc/scrypt-ts'
 *
 * export class ARIATimelockContract extends SmartContract {
 *   @prop() readonly commitHash: Sha256
 *   @prop() readonly operatorPkh: PubKeyHash
 *   @prop() readonly unlockHeight: bigint
 *   @prop() readonly epochId: ByteString
 *
 *   constructor(...) { super(...arguments); ... }
 *
 *   // Normal unlock: within timelock, operator provides pre-image
 *   @method()
 *   public unlock(sig: Sig, pubKey: PubKey, preimage: ByteString) {
 *     assert(hash160(pubKey) == this.operatorPkh, 'wrong operator key')
 *     assert(this.checkSig(sig, pubKey), 'invalid signature')
 *     assert(sha256(preimage) == this.commitHash, 'preimage mismatch')
 *   }
 *
 *   // Timelock recovery: after unlockHeight, any preimage holder can claim
 *   @method(SigHash.ANYONECANPAY_SINGLE)
 *   public recoverAfterTimelock(preimage: ByteString) {
 *     assert(this.ctx.locktime >= this.unlockHeight, 'timelock not expired')
 *     assert(sha256(preimage) == this.commitHash, 'preimage mismatch')
 *   }
 * }
 * ```
 */

import {
  ScryptBase,
  Sha256,
  PubKeyHash,
  BlockHeight,
  ByteString,
  ContractVerifyResult,
  toSha256,
  toPubKeyHash,
  toBlockHeight,
  toByteString,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "./scrypt_adapter.js"

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

export interface TimelockContractState {
  epochId: string
  /**
   * SHA-256 of the commitment data (e.g. EPOCH_CLOSE payload bytes).
   * The preimage must be revealed to unlock before the timelock expires.
   */
  commitHash: Sha256
  /** HASH160 of operator's BSV public key. */
  operatorPkh: PubKeyHash
  /**
   * BSV block height at which the timelock expires.
   * After this block, recovery can be initiated without operator signature.
   */
  unlockHeight: BlockHeight
  /** Creation timestamp (ms) — off-chain metadata. */
  createdAtMs: number
}

export interface TimelockNormalWitness {
  mode: "normal"
  sigHex: string
  pubKeyHex: string
  /** Hex-encoded preimage whose SHA-256 equals commitHash. */
  preimageHex: string
}

export interface TimelockRecoveryWitness {
  mode: "recovery"
  /** Current block height — must be ≥ unlockHeight. */
  currentHeight: number
  preimageHex: string
}

export type TimelockWitness = TimelockNormalWitness | TimelockRecoveryWitness

// ---------------------------------------------------------------------------
// Contract
// ---------------------------------------------------------------------------

/** Minimum timelock: 144 blocks ≈ 24 hours on BSV. */
export const MIN_TIMELOCK_BLOCKS = 144

/**
 * Type-2 ARIA contract: timelock + commitment hash enforcement.
 *
 * Two unlock paths:
 * - **Normal**: operator provides sig + commitment preimage (anytime before expiry)
 * - **Recovery**: anyone with preimage can claim after `unlockHeight` blocks
 *
 * The preimage is typically the serialized EPOCH_CLOSE payload, ensuring that
 * the epoch can always be recovered even if the operator key is lost.
 */
export class ARIATimelockContract extends ScryptBase<TimelockWitness> {
  readonly epochId: string
  readonly commitHash: Sha256
  readonly operatorPkh: PubKeyHash
  readonly unlockHeight: BlockHeight
  readonly createdAtMs: number

  constructor(state: TimelockContractState) {
    super()
    this.epochId = state.epochId
    this.commitHash = toSha256(state.commitHash)
    this.operatorPkh = toPubKeyHash(state.operatorPkh)
    this.unlockHeight = toBlockHeight(state.unlockHeight)
    this.createdAtMs = state.createdAtMs
  }

  /**
   * OP_RETURN payload layout:
   * ARIA_MAGIC | BRC121_VERSION | type(02) | epochId | commitHash |
   * operatorPkh | unlockHeight (4 bytes BE) | createdAtMs (8 bytes BE)
   */
  getLockingScriptHex(): string {
    const epochIdHex = Buffer.from(this.epochId, "utf8").toString("hex").padEnd(64, "0").slice(0, 64)
    const heightHex = this.unlockHeight.toString(16).padStart(8, "0")
    const tsHex = this.createdAtMs.toString(16).padStart(16, "0")

    const payload =
      ARIA_MAGIC +
      BRC121_VERSION +
      "02" +          // contract type: TIMELOCK
      epochIdHex +
      this.commitHash +
      this.operatorPkh +
      heightHex +
      tsHex

    return buildOpReturnScript(payload)
  }

  /**
   * Off-chain verification of an unlock witness.
   *
   * Normal path:  validates pubKey format, sig presence, and preimage SHA-256.
   * Recovery path: validates that currentHeight ≥ unlockHeight and preimage.
   *
   * NOTE: Actual ECDSA signature verification and SHA-256 of preimage are
   * enforced on-chain by the sCrypt VM. Off-chain we use structural checks.
   */
  verify(witness: TimelockWitness): ContractVerifyResult {
    if (witness.mode === "normal") {
      if (!/^(02|03)[0-9a-f]{64}$/i.test(witness.pubKeyHex)) {
        return { success: false, error: "pubKey must be a compressed 33-byte key" }
      }
      if (!witness.sigHex || witness.sigHex.length < 8) {
        return { success: false, error: "sigHex is missing or too short" }
      }
      return this._checkPreimage(witness.preimageHex)
    }

    if (witness.mode === "recovery") {
      if (witness.currentHeight < this.unlockHeight) {
        return {
          success: false,
          error: `Timelock not expired: current=${witness.currentHeight} required=${this.unlockHeight}`,
        }
      }
      return this._checkPreimage(witness.preimageHex)
    }

    return { success: false, error: "Unknown witness mode" }
  }

  /** True if the current block height has passed the timelock. */
  isExpired(currentHeight: number): boolean {
    return currentHeight >= this.unlockHeight
  }

  /**
   * Verify that SHA-256(preimageHex bytes) === commitHash.
   * Off-chain we delegate the actual SHA-256 computation to the caller;
   * this method validates the preimage hex format and length.
   *
   * To fully verify off-chain, call `sha256Hex(preimageHex)` from hasher.ts
   * and compare with `contract.commitHash`.
   */
  private _checkPreimage(preimageHex: string): ContractVerifyResult {
    if (!preimageHex || preimageHex.length === 0) {
      return { success: false, error: "preimage must not be empty" }
    }
    if (preimageHex.length % 2 !== 0 || !/^[0-9a-f]+$/i.test(preimageHex)) {
      return { success: false, error: "preimage must be a valid hex string" }
    }
    // Signal that SHA-256 match must be checked by caller using hasher.ts
    return { success: true }
  }

  // ---------------------------------------------------------------------------
  // Factory
  // ---------------------------------------------------------------------------

  /**
   * Create a timelock contract for an epoch.
   * @param epochId       Epoch identifier.
   * @param commitHash    SHA-256 of the EPOCH_CLOSE payload bytes.
   * @param operatorPkh   HASH160 of operator public key.
   * @param currentHeight BSV block height at contract creation time.
   * @param timelockBlocks Number of blocks to lock (default: 144 = ~24 h).
   */
  static forEpoch(
    epochId: string,
    commitHash: Sha256,
    operatorPkh: PubKeyHash,
    currentHeight: number,
    timelockBlocks: number = MIN_TIMELOCK_BLOCKS,
  ): ARIATimelockContract {
    if (timelockBlocks < MIN_TIMELOCK_BLOCKS) {
      throw new Error(`timelockBlocks must be ≥ ${MIN_TIMELOCK_BLOCKS}, got ${timelockBlocks}`)
    }
    return new ARIATimelockContract({
      epochId,
      commitHash,
      operatorPkh,
      unlockHeight: toBlockHeight(currentHeight + timelockBlocks),
      createdAtMs: Date.now(),
    })
  }
}
