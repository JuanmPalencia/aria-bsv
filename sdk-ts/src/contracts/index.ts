/**
 * ARIA sCrypt smart contracts — BRC-121 on-chain enforcement layer.
 *
 * Provides TypeScript classes for constructing and verifying the three
 * contract types defined in BRC-121:
 *
 * - Type 01 → `ARIAEpochContract`  — epoch pre-commitment bond
 * - Type 02 → `ARIATimelockContract` — dead man's switch
 * - Type 03 → `ARIARegistryContract` — AI system registry
 *
 * The base types, branded type validators, and OP_RETURN builder are
 * re-exported from the adapter so callers only need one import path.
 *
 * @module contracts
 */

// ---------------------------------------------------------------------------
// Adapter primitives
// ---------------------------------------------------------------------------

export {
  toSha256,
  toPubKeyHash,
  toByteString,
  toBlockHeight,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "./scrypt_adapter.js"

export type {
  Sha256,
  PubKeyHash,
  ByteString,
  BlockHeight,
  ContractVerifyResult,
} from "./scrypt_adapter.js"

// Note: ScryptBase is intentionally not re-exported — it is an internal
// implementation detail. Users extend their own contracts via scrypt-ts.

// ---------------------------------------------------------------------------
// Type 01 — Epoch contract
// ---------------------------------------------------------------------------

export { ARIAEpochContract, EPOCH_BOND_SAT } from "./epoch_contract.js"
export type {
  EpochContractState,
  EpochUnlockWitness,
} from "./epoch_contract.js"

// ---------------------------------------------------------------------------
// Type 02 — Timelock contract
// ---------------------------------------------------------------------------

export { ARIATimelockContract, MIN_TIMELOCK_BLOCKS } from "./timelock_contract.js"
export type {
  TimelockContractState,
  TimelockNormalWitness,
  TimelockRecoveryWitness,
  TimelockWitness,
} from "./timelock_contract.js"

// ---------------------------------------------------------------------------
// Type 03 — Registry contract
// ---------------------------------------------------------------------------

export { ARIARegistryContract, EUAIActRisk, MAX_MODEL_HASHES, MAX_SYSTEM_ID_BYTES } from "./registry_contract.js"
export type { RegistryEntry, RegistryVerifyResult } from "./registry_contract.js"
