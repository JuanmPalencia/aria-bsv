/**
 * ARIARegistryContract — On-chain AI system registry for BRC-121.
 *
 * Maintains an immutable, censorship-resistant directory of registered AI
 * systems on BSV. Each registration is an OP_RETURN output referencing the
 * system_id hash, model hashes, and EU AI Act risk classification.
 *
 * This is a "data-only" contract: no satoshis need to be locked. The
 * registration output acts as a timestamped, immutable record.
 *
 * sCrypt source (for production compilation):
 *
 * ```ts
 * // Registry entries are pure data OP_RETURN outputs — no script logic.
 * // The `ARIARegistryContract` class computes the correct payload and
 * // provides verification that a given tx output is a valid ARIA registration.
 * // No SmartContract extension needed — this is a data witness pattern.
 * ```
 *
 * EU AI Act Article 49 requires operators of high-risk AI systems to register
 * in official databases. ARIA's on-chain registry supplements this requirement
 * with a tamper-evident, publicly verifiable record.
 */

import {
  Sha256,
  PubKeyHash,
  ByteString,
  toSha256,
  toPubKeyHash,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "./scrypt_adapter.js"

// ---------------------------------------------------------------------------
// EU AI Act risk classification (Annex III)
// ---------------------------------------------------------------------------

export enum EUAIActRisk {
  /** Minimal risk — no specific obligations. */
  MINIMAL = "MINIMAL",
  /** Limited risk — transparency obligations. */
  LIMITED = "LIMITED",
  /** High risk — full BRC-121 compliance required (Art. 9–15). */
  HIGH = "HIGH",
  /** Unacceptable risk — prohibited under EU AI Act Art. 5. */
  UNACCEPTABLE = "UNACCEPTABLE",
}

const RISK_BYTE: Record<EUAIActRisk, string> = {
  [EUAIActRisk.MINIMAL]: "00",
  [EUAIActRisk.LIMITED]: "01",
  [EUAIActRisk.HIGH]: "02",
  [EUAIActRisk.UNACCEPTABLE]: "ff",
}

// ---------------------------------------------------------------------------
// Registration types
// ---------------------------------------------------------------------------

export interface RegistryEntry {
  /** Unique system identifier (e.g. "kairos-v3"). Max 64 chars. */
  systemId: string
  /**
   * SHA-256 hashes of all deployed model files. Computed via `hash_file()`
   * in the Python SDK or `hashBytes()` in sdk-ts.
   */
  modelHashes: Sha256[]
  /** HASH160 of the operator's BSV public key. */
  operatorPkh: PubKeyHash
  /** EU AI Act risk classification. */
  riskLevel: EUAIActRisk
  /** IANA-registered jurisdiction code, e.g. "EU", "US-CA". Max 8 chars. */
  jurisdiction: string
  /** Registration timestamp (ms). */
  registeredAtMs: number
  /** Optional: previous registration txid (for updates / re-registration). */
  prevRegistrationTxid?: string
}

export interface RegistryVerifyResult {
  valid: boolean
  systemId?: string
  riskLevel?: EUAIActRisk
  modelCount?: number
  error?: string
}

// ---------------------------------------------------------------------------
// Contract payload sizes
// ---------------------------------------------------------------------------

/** Maximum number of model hashes per registration. */
export const MAX_MODEL_HASHES = 8

/** Maximum system ID length in bytes. */
export const MAX_SYSTEM_ID_BYTES = 64

// ---------------------------------------------------------------------------
// ARIARegistryContract
// ---------------------------------------------------------------------------

/**
 * Type-3 ARIA contract: system registration record.
 *
 * Produces an OP_RETURN payload that uniquely identifies an AI system on BSV.
 * Any observer can independently verify system registration history by
 * scanning BSV transactions for ARIA_MAGIC + type(03) outputs.
 */
export class ARIARegistryContract {
  readonly systemId: string
  readonly modelHashes: Sha256[]
  readonly operatorPkh: PubKeyHash
  readonly riskLevel: EUAIActRisk
  readonly jurisdiction: string
  readonly registeredAtMs: number
  readonly prevRegistrationTxid: string | undefined

  constructor(entry: RegistryEntry) {
    if (entry.systemId.length === 0 || entry.systemId.length > MAX_SYSTEM_ID_BYTES) {
      throw new Error(`systemId must be 1–${MAX_SYSTEM_ID_BYTES} chars`)
    }
    if (entry.modelHashes.length === 0 || entry.modelHashes.length > MAX_MODEL_HASHES) {
      throw new Error(`modelHashes must be 1–${MAX_MODEL_HASHES} elements`)
    }
    if (entry.jurisdiction.length === 0 || entry.jurisdiction.length > 8) {
      throw new Error(`jurisdiction must be 1–8 chars`)
    }

    this.systemId = entry.systemId
    this.modelHashes = entry.modelHashes.map(toSha256)
    this.operatorPkh = toPubKeyHash(entry.operatorPkh)
    this.riskLevel = entry.riskLevel
    this.jurisdiction = entry.jurisdiction
    this.registeredAtMs = entry.registeredAtMs
    this.prevRegistrationTxid = entry.prevRegistrationTxid
  }

  // ---------------------------------------------------------------------------
  // Serialization
  // ---------------------------------------------------------------------------

  /**
   * OP_RETURN payload layout:
   * ARIA_MAGIC | BRC121_VERSION | type(03) | risk(1b) | modelCount(1b) |
   * systemIdLen(1b) | systemIdHex | operatorPkh | registeredAtMs(8b) |
   * jurisdictionHex | [modelHash × modelCount] | [prevTxidOrZero]
   */
  getLockingScriptHex(): string {
    const sysHex = Buffer.from(this.systemId, "utf8").toString("hex")
    const sysLen = (sysHex.length / 2).toString(16).padStart(2, "0")
    const jurisHex = Buffer.from(this.jurisdiction.padEnd(8, "\0"), "utf8").toString("hex")
    const tsHex = this.registeredAtMs.toString(16).padStart(16, "0")
    const prevHex = this.prevRegistrationTxid
      ? this.prevRegistrationTxid.toLowerCase()
      : "0".repeat(64)

    const payload =
      ARIA_MAGIC +
      BRC121_VERSION +
      "03" +                                    // contract type: REGISTRY
      RISK_BYTE[this.riskLevel] +
      this.modelHashes.length.toString(16).padStart(2, "0") +
      sysLen +
      sysHex +
      this.operatorPkh +
      tsHex +
      jurisHex +
      this.modelHashes.join("") +
      prevHex

    return buildOpReturnScript(payload)
  }

  /**
   * Parse and validate an OP_RETURN script hex as an ARIA registry entry.
   * Suitable for scanning BSV transactions in a block explorer or SPV node.
   */
  static parseFromScript(scriptHex: string): RegistryVerifyResult {
    try {
      // Strip OP_0 + OP_RETURN prefix and pushdata
      const clean = scriptHex.toUpperCase().startsWith("00") ? scriptHex.slice(2) : scriptHex
      if (!clean.toLowerCase().startsWith("6a")) {
        return { valid: false, error: "Not an OP_RETURN output" }
      }

      // Locate ARIA_MAGIC
      const magic = "41524941" // "ARIA"
      const magicIdx = clean.toLowerCase().indexOf(magic)
      if (magicIdx === -1) {
        return { valid: false, error: "ARIA magic bytes not found" }
      }

      const data = clean.toLowerCase().slice(magicIdx)
      // data[0..7]  = magic (41524941)
      // data[8..9]  = version
      // data[10..11] = type (must be 03)
      const type = data.slice(10, 12)
      if (type !== "03") {
        return { valid: false, error: `Not a registry entry (type=${type})` }
      }

      const riskByte = data.slice(12, 14)
      const riskLevel = (Object.entries(RISK_BYTE).find(([, v]) => v === riskByte)?.[0] ?? "UNKNOWN") as EUAIActRisk
      const modelCount = parseInt(data.slice(14, 16), 16)

      if (modelCount < 1 || modelCount > MAX_MODEL_HASHES) {
        return { valid: false, error: `Invalid model count: ${modelCount}` }
      }

      const sysLen = parseInt(data.slice(16, 18), 16)
      const sysHex = data.slice(18, 18 + sysLen * 2)
      const systemId = Buffer.from(sysHex, "hex").toString("utf8")

      return { valid: true, systemId, riskLevel, modelCount }
    } catch (e) {
      return { valid: false, error: `Parse error: ${e}` }
    }
  }

  // ---------------------------------------------------------------------------
  // Factory
  // ---------------------------------------------------------------------------

  /** Create a registration entry for a high-risk AI system (EU AI Act default). */
  static highRisk(params: {
    systemId: string
    modelHashes: string[]
    operatorPkh: string
    jurisdiction?: string
  }): ARIARegistryContract {
    return new ARIARegistryContract({
      systemId: params.systemId,
      modelHashes: params.modelHashes.map(toSha256),
      operatorPkh: toPubKeyHash(params.operatorPkh),
      riskLevel: EUAIActRisk.HIGH,
      jurisdiction: params.jurisdiction ?? "EU",
      registeredAtMs: Date.now(),
    })
  }
}
