/**
 * Contracts module tests — BRC-121 sCrypt smart contracts.
 *
 * Tests cover all three contract types plus the shared adapter utilities.
 * No sCrypt compiler needed — all tests run in pure TypeScript via Vitest.
 */

import { describe, it, expect } from "vitest"
import {
  toSha256,
  toPubKeyHash,
  toByteString,
  toBlockHeight,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "../contracts/scrypt_adapter.js"
import { ARIAEpochContract, EPOCH_BOND_SAT } from "../contracts/epoch_contract.js"
import type { EpochContractState, EpochUnlockWitness } from "../contracts/epoch_contract.js"
import { ARIATimelockContract, MIN_TIMELOCK_BLOCKS } from "../contracts/timelock_contract.js"
import type { TimelockContractState } from "../contracts/timelock_contract.js"
import {
  ARIARegistryContract,
  EUAIActRisk,
  MAX_MODEL_HASHES,
  MAX_SYSTEM_ID_BYTES,
} from "../contracts/registry_contract.js"

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const ZERO32 = "0".repeat(64)  // zeroed SHA-256
const ZERO20 = "0".repeat(40)  // zeroed PubKeyHash
const VALID_PKH = "a".repeat(40)  // 20-byte HASH160
const VALID_HASH = "b".repeat(64)  // SHA-256
const VALID_HASH2 = "c".repeat(64)
const COMPRESSED_PUBKEY = "02" + "d".repeat(64)  // compressed pub key (33 b)
const DUMMY_SIG = "3045" + "aa".repeat(34)  // DER-ish sig (≥ 8 hex chars)
const EPOCH_ID = "ep_1711234567890_0001"

function makeEpochState(overrides?: Partial<EpochContractState>): EpochContractState {
  return {
    epochId: EPOCH_ID,
    merkleRoot: toSha256(VALID_HASH),
    operatorPkh: toPubKeyHash(VALID_PKH),
    openTimestampMs: 1711234567890,
    systemIdHash: toSha256(VALID_HASH2),
    zkProofHash: toSha256(ZERO32),
    ...overrides,
  }
}

function makeTimelockState(overrides?: Partial<TimelockContractState>): TimelockContractState {
  return {
    epochId: EPOCH_ID,
    commitHash: toSha256(VALID_HASH),
    operatorPkh: toPubKeyHash(VALID_PKH),
    unlockHeight: toBlockHeight(800144),
    createdAtMs: 1711234567890,
    ...overrides,
  }
}

// ===========================================================================
// 1. Adapter primitives
// ===========================================================================

describe("scrypt_adapter — toSha256", () => {
  it("accepts a valid 64-char lowercase hex string", () => {
    expect(() => toSha256(VALID_HASH)).not.toThrow()
    expect(toSha256(VALID_HASH)).toBe(VALID_HASH)
  })

  it("accepts a 64-char uppercase hex string (normalises to lower)", () => {
    expect(toSha256(VALID_HASH.toUpperCase())).toBe(VALID_HASH)
  })

  it("rejects a string that is too short", () => {
    expect(() => toSha256("abc123")).toThrow("Invalid SHA-256")
  })

  it("rejects a string that is too long", () => {
    expect(() => toSha256("a".repeat(65))).toThrow("Invalid SHA-256")
  })

  it("rejects non-hex characters", () => {
    expect(() => toSha256("z".repeat(64))).toThrow("Invalid SHA-256")
  })
})

describe("scrypt_adapter — toPubKeyHash", () => {
  it("accepts a valid 40-char hex string", () => {
    expect(() => toPubKeyHash(VALID_PKH)).not.toThrow()
    expect(toPubKeyHash(VALID_PKH)).toBe(VALID_PKH)
  })

  it("rejects wrong length", () => {
    expect(() => toPubKeyHash("a".repeat(39))).toThrow("Invalid PubKeyHash")
    expect(() => toPubKeyHash("a".repeat(41))).toThrow("Invalid PubKeyHash")
  })

  it("rejects non-hex characters", () => {
    expect(() => toPubKeyHash("g".repeat(40))).toThrow("Invalid PubKeyHash")
  })
})

describe("scrypt_adapter — toBlockHeight", () => {
  it("accepts zero", () => {
    expect(toBlockHeight(0)).toBe(0)
  })

  it("accepts a positive integer", () => {
    expect(toBlockHeight(800000)).toBe(800000)
  })

  it("rejects a float", () => {
    expect(() => toBlockHeight(1.5)).toThrow("Invalid block height")
  })

  it("rejects a negative number", () => {
    expect(() => toBlockHeight(-1)).toThrow("Invalid block height")
  })
})

describe("scrypt_adapter — buildOpReturnScript", () => {
  it("starts with OP_0 + OP_RETURN (006a)", () => {
    const script = buildOpReturnScript(ARIA_MAGIC)
    expect(script.startsWith("006a")).toBe(true)
  })

  it("embeds ARIA magic bytes", () => {
    const script = buildOpReturnScript(ARIA_MAGIC)
    expect(script).toContain("41524941")
  })

  it("ARIA_MAGIC equals ASCII 'ARIA'", () => {
    expect(Buffer.from(ARIA_MAGIC, "hex").toString("ascii")).toBe("ARIA")
  })

  it("BRC121_VERSION equals 0x01", () => {
    expect(BRC121_VERSION).toBe("01")
  })
})

// ===========================================================================
// 2. ARIAEpochContract
// ===========================================================================

describe("ARIAEpochContract — constructor", () => {
  it("constructs without error with valid state", () => {
    expect(() => new ARIAEpochContract(makeEpochState())).not.toThrow()
  })

  it("normalises hex fields to lowercase", () => {
    const c = new ARIAEpochContract(
      makeEpochState({ merkleRoot: toSha256(VALID_HASH.toUpperCase()) }),
    )
    expect(c.merkleRoot).toBe(VALID_HASH)
  })

  it("stores correct epochId", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.epochId).toBe(EPOCH_ID)
  })

  it("EPOCH_BOND_SAT is ≥ 1000n satoshis", () => {
    expect(EPOCH_BOND_SAT).toBeGreaterThanOrEqual(1000n)
  })
})

describe("ARIAEpochContract — getLockingScriptHex", () => {
  it("returns a string starting with 006a (OP_0 OP_RETURN)", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.getLockingScriptHex()).toMatch(/^006a/)
  })

  it("embeds ARIA magic 41524941", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.getLockingScriptHex()).toContain("41524941")
  })

  it("embeds contract type 01 for epoch", () => {
    const c = new ARIAEpochContract(makeEpochState())
    const script = c.getLockingScriptHex()
    // Layout after magic+version: type byte 01
    const magicIdx = script.indexOf("41524941")
    const typeOffset = magicIdx + 8 + 2  // after magic (8) + version (2)
    expect(script.slice(typeOffset, typeOffset + 2)).toBe("01")
  })

  it("embeds the merkle root", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.getLockingScriptHex()).toContain(VALID_HASH)
  })

  it("embeds the operator pubkey hash", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.getLockingScriptHex()).toContain(VALID_PKH)
  })

  it("is deterministic — same state → same hex", () => {
    const state = makeEpochState()
    const hex1 = new ARIAEpochContract(state).getLockingScriptHex()
    const hex2 = new ARIAEpochContract(state).getLockingScriptHex()
    expect(hex1).toBe(hex2)
  })
})

describe("ARIAEpochContract — verify", () => {
  function validWitness(): EpochUnlockWitness {
    return {
      sigHex: DUMMY_SIG,
      pubKeyHex: COMPRESSED_PUBKEY,
      closeRoot: toSha256(VALID_HASH),
    }
  }

  it("returns success=true for a valid witness", () => {
    const c = new ARIAEpochContract(makeEpochState())
    expect(c.verify(validWitness())).toEqual({ success: true })
  })

  it("fails when pubKey is uncompressed (starts with 04)", () => {
    const c = new ARIAEpochContract(makeEpochState())
    const w = validWitness()
    w.pubKeyHex = "04" + "e".repeat(64)
    expect(c.verify(w).success).toBe(false)
    expect(c.verify(w).error).toContain("compressed")
  })

  it("fails when sig is empty", () => {
    const c = new ARIAEpochContract(makeEpochState())
    const w = validWitness()
    w.sigHex = ""
    expect(c.verify(w).success).toBe(false)
  })

  it("fails when closeRoot differs from committed merkleRoot", () => {
    const c = new ARIAEpochContract(makeEpochState())
    const w = validWitness()
    w.closeRoot = toSha256(VALID_HASH2)
    const result = c.verify(w)
    expect(result.success).toBe(false)
    expect(result.error).toContain("mismatch")
  })

  it("fails when closeRoot is not valid hex-64", () => {
    const c = new ARIAEpochContract(makeEpochState())
    const w = { ...validWitness(), closeRoot: "bad" as ReturnType<typeof toSha256> }
    expect(c.verify(w).success).toBe(false)
  })
})

// ===========================================================================
// 3. ARIATimelockContract
// ===========================================================================

describe("ARIATimelockContract — constructor", () => {
  it("constructs without error with valid state", () => {
    expect(() => new ARIATimelockContract(makeTimelockState())).not.toThrow()
  })

  it("MIN_TIMELOCK_BLOCKS is at least 144 (≈24h)", () => {
    expect(MIN_TIMELOCK_BLOCKS).toBeGreaterThanOrEqual(144)
  })

  it("forEpoch factory enforces MIN_TIMELOCK_BLOCKS", () => {
    expect(() =>
      ARIATimelockContract.forEpoch(
        EPOCH_ID,
        toSha256(VALID_HASH),
        toPubKeyHash(VALID_PKH),
        800000,
        10,  // below minimum
      ),
    ).toThrow()
  })

  it("forEpoch creates contract with correct unlockHeight", () => {
    const blocks = 200
    const currentHeight = 800000
    const c = ARIATimelockContract.forEpoch(
      EPOCH_ID,
      toSha256(VALID_HASH),
      toPubKeyHash(VALID_PKH),
      currentHeight,
      blocks,
    )
    expect(c.unlockHeight).toBe(currentHeight + blocks)
  })
})

describe("ARIATimelockContract — getLockingScriptHex", () => {
  it("starts with 006a (OP_0 OP_RETURN)", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    expect(c.getLockingScriptHex()).toMatch(/^006a/)
  })

  it("contains contract type 02", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    const script = c.getLockingScriptHex()
    const magicIdx = script.indexOf("41524941")
    const typeOffset = magicIdx + 8 + 2
    expect(script.slice(typeOffset, typeOffset + 2)).toBe("02")
  })

  it("embeds commitHash", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    expect(c.getLockingScriptHex()).toContain(VALID_HASH)
  })
})

describe("ARIATimelockContract — verify normal", () => {
  it("succeeds with valid sig + preimage", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    const result = c.verify({
      mode: "normal",
      sigHex: DUMMY_SIG,
      pubKeyHex: COMPRESSED_PUBKEY,
      preimageHex: "deadbeef",
    })
    expect(result.success).toBe(true)
  })

  it("fails when pubKey is not 33-byte compressed", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    const result = c.verify({
      mode: "normal",
      sigHex: DUMMY_SIG,
      pubKeyHex: "04" + "f".repeat(64),
      preimageHex: "deadbeef",
    })
    expect(result.success).toBe(false)
  })

  it("fails when preimage is empty", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    const result = c.verify({
      mode: "normal",
      sigHex: DUMMY_SIG,
      pubKeyHex: COMPRESSED_PUBKEY,
      preimageHex: "",
    })
    expect(result.success).toBe(false)
  })

  it("fails when preimage is not valid hex", () => {
    const c = new ARIATimelockContract(makeTimelockState())
    const result = c.verify({
      mode: "normal",
      sigHex: DUMMY_SIG,
      pubKeyHex: COMPRESSED_PUBKEY,
      preimageHex: "xyz",
    })
    expect(result.success).toBe(false)
  })
})

describe("ARIATimelockContract — verify recovery", () => {
  const LOCK_HEIGHT = 800144

  it("succeeds after timelock expires", () => {
    const c = new ARIATimelockContract(makeTimelockState({ unlockHeight: toBlockHeight(LOCK_HEIGHT) }))
    const result = c.verify({
      mode: "recovery",
      currentHeight: LOCK_HEIGHT,
      preimageHex: "cafebabe01",
    })
    expect(result.success).toBe(true)
  })

  it("fails before timelock expires", () => {
    const c = new ARIATimelockContract(makeTimelockState({ unlockHeight: toBlockHeight(LOCK_HEIGHT) }))
    const result = c.verify({
      mode: "recovery",
      currentHeight: LOCK_HEIGHT - 1,
      preimageHex: "cafebabe01",
    })
    expect(result.success).toBe(false)
    expect(result.error).toContain("not expired")
  })

  it("isExpired returns true at unlock height", () => {
    const c = new ARIATimelockContract(makeTimelockState({ unlockHeight: toBlockHeight(LOCK_HEIGHT) }))
    expect(c.isExpired(LOCK_HEIGHT)).toBe(true)
    expect(c.isExpired(LOCK_HEIGHT - 1)).toBe(false)
  })
})

// ===========================================================================
// 4. ARIARegistryContract
// ===========================================================================

describe("ARIARegistryContract — constructor", () => {
  it("constructs without error with valid entry", () => {
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "kairos-v3",
          modelHashes: [toSha256(VALID_HASH)],
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU",
          registeredAtMs: Date.now(),
        }),
    ).not.toThrow()
  })

  it(`rejects systemId longer than ${MAX_SYSTEM_ID_BYTES} chars`, () => {
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "x".repeat(MAX_SYSTEM_ID_BYTES + 1),
          modelHashes: [toSha256(VALID_HASH)],
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU",
          registeredAtMs: Date.now(),
        }),
    ).toThrow("systemId")
  })

  it("rejects empty systemId", () => {
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "",
          modelHashes: [toSha256(VALID_HASH)],
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU",
          registeredAtMs: Date.now(),
        }),
    ).toThrow("systemId")
  })

  it(`rejects more than ${MAX_MODEL_HASHES} model hashes`, () => {
    const hashes = Array.from({ length: MAX_MODEL_HASHES + 1 }, (_, i) =>
      toSha256(i.toString(16).padStart(64, "0")),
    )
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "test-system",
          modelHashes: hashes,
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU",
          registeredAtMs: Date.now(),
        }),
    ).toThrow("modelHashes")
  })

  it("rejects empty modelHashes array", () => {
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "test-system",
          modelHashes: [],
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU",
          registeredAtMs: Date.now(),
        }),
    ).toThrow("modelHashes")
  })

  it("rejects jurisdiction longer than 8 chars", () => {
    expect(
      () =>
        new ARIARegistryContract({
          systemId: "test-system",
          modelHashes: [toSha256(VALID_HASH)],
          operatorPkh: toPubKeyHash(VALID_PKH),
          riskLevel: EUAIActRisk.HIGH,
          jurisdiction: "EU-LONGERSTRING",
          registeredAtMs: Date.now(),
        }),
    ).toThrow("jurisdiction")
  })
})

describe("ARIARegistryContract — getLockingScriptHex", () => {
  function makeContract() {
    return new ARIARegistryContract({
      systemId: "kairos-v3",
      modelHashes: [toSha256(VALID_HASH), toSha256(VALID_HASH2)],
      operatorPkh: toPubKeyHash(VALID_PKH),
      riskLevel: EUAIActRisk.HIGH,
      jurisdiction: "EU",
      registeredAtMs: 1711234567890,
    })
  }

  it("starts with 006a (OP_0 OP_RETURN)", () => {
    expect(makeContract().getLockingScriptHex()).toMatch(/^006a/)
  })

  it("embeds ARIA magic bytes", () => {
    expect(makeContract().getLockingScriptHex()).toContain("41524941")
  })

  it("contains contract type 03", () => {
    const script = makeContract().getLockingScriptHex()
    const magicIdx = script.indexOf("41524941")
    const typeOffset = magicIdx + 8 + 2
    expect(script.slice(typeOffset, typeOffset + 2)).toBe("03")
  })

  it("embeds operatorPkh", () => {
    expect(makeContract().getLockingScriptHex()).toContain(VALID_PKH)
  })

  it("embeds system ID as UTF-8 hex", () => {
    const systemIdHex = Buffer.from("kairos-v3", "utf8").toString("hex")
    expect(makeContract().getLockingScriptHex()).toContain(systemIdHex)
  })

  it("is deterministic", () => {
    const hex1 = makeContract().getLockingScriptHex()
    const hex2 = makeContract().getLockingScriptHex()
    expect(hex1).toBe(hex2)
  })
})

describe("ARIARegistryContract — static parseFromScript", () => {
  it("validates a self-generated script as a registry entry", () => {
    const c = new ARIARegistryContract({
      systemId: "test-ai",
      modelHashes: [toSha256(VALID_HASH)],
      operatorPkh: toPubKeyHash(VALID_PKH),
      riskLevel: EUAIActRisk.HIGH,
      jurisdiction: "EU",
      registeredAtMs: 1711234567890,
    })
    const result = ARIARegistryContract.parseFromScript(c.getLockingScriptHex())
    expect(result.valid).toBe(true)
    expect(result.systemId).toBe("test-ai")
    expect(result.riskLevel).toBe(EUAIActRisk.HIGH)
    expect(result.modelCount).toBe(1)
  })

  it("rejects a non-OP_RETURN script", () => {
    const result = ARIARegistryContract.parseFromScript("76a914" + "00".repeat(20) + "88ac")
    expect(result.valid).toBe(false)
  })

  it("rejects a script without ARIA magic", () => {
    // OP_RETURN + random payload
    const result = ARIARegistryContract.parseFromScript("006a" + "04" + "deadbeef")
    expect(result.valid).toBe(false)
    expect(result.error).toContain("magic")
  })
})

describe("ARIARegistryContract — highRisk factory", () => {
  it("creates a HIGH risk entry with default EU jurisdiction", () => {
    const c = ARIARegistryContract.highRisk({
      systemId: "my-ai",
      modelHashes: [VALID_HASH],
      operatorPkh: VALID_PKH,
    })
    expect(c.riskLevel).toBe(EUAIActRisk.HIGH)
    expect(c.jurisdiction).toBe("EU")
    expect(c.systemId).toBe("my-ai")
    expect(c.modelHashes).toHaveLength(1)
  })

  it("accepts a custom jurisdiction", () => {
    const c = ARIARegistryContract.highRisk({
      systemId: "my-ai",
      modelHashes: [VALID_HASH],
      operatorPkh: VALID_PKH,
      jurisdiction: "US-CA",
    })
    expect(c.jurisdiction).toBe("US-CA")
  })
})

describe("EUAIActRisk enum", () => {
  it("has the four EU AI Act risk levels", () => {
    expect(EUAIActRisk.MINIMAL).toBeDefined()
    expect(EUAIActRisk.LIMITED).toBeDefined()
    expect(EUAIActRisk.HIGH).toBeDefined()
    expect(EUAIActRisk.UNACCEPTABLE).toBeDefined()
  })
})
