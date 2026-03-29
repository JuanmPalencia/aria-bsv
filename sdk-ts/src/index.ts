/**
 * aria-bsv — Auditable Real-time Inference Architecture, TypeScript SDK.
 *
 * @packageDocumentation
 */

// Core types
export type {
  AuditRecord,
  EpochOpenPayload,
  EpochClosePayload,
  VerificationResult,
  TxStatus,
  BroadcasterOptions,
  AuditConfig,
} from "./types.js";

// Hashing
export {
  ARIASerializationError,
  canonicalJson,
  hashObject,
  hashObjectSync,
  sha256Hex,
  sha256HexSync,
} from "./hasher.js";

// Merkle tree
export {
  EMPTY_ROOT,
  MerkleTree,
  computeMerkleRoot,
  computeMerkleRootSync,
  generateMerkleProof,
  verifyMerkleProof,
} from "./merkle.js";
export type { MerkleProof } from "./merkle.js";

// BSV broadcaster
export { ARCBroadcaster } from "./broadcaster.js";

// Inference auditor
export { InferenceAuditor } from "./auditor.js";

// Epoch verifier
export { verifyEpoch, OnChainVerifier } from "./verifier.js";
export type { OnChainVerifierOptions } from "./verifier.js";

// Streaming auditor
export { StreamingSession, ARIAStreamingAuditor, auditOpenAIStream, auditAnthropicStream } from "./streaming.js";

// Dataset anchoring
export { hashBytes, hashColumns, DatasetAnchorer, verifyDatasetAnchor } from "./dataset.js";
export type { DatasetAnchor, DatasetAnchorOptions, DatasetAnchorerConfig } from "./dataset.js";

// SPV verifier
export {
  sha256d,
  reverseHex,
  BlockHeader,
  SPVError,
  verifyMerkleBranch,
  verifyHeaderChain,
  verifySpvProof,
} from "./spv.js";
export type { MerkleBranch, SPVProof, SPVChainResult } from "./spv.js";

// Overlay services (BRC-31)
export { TopicManager, LookupService, OverlayClient } from "./overlay.js";
export type { AdmittanceResult, LookupResult, OverlayClientOptions } from "./overlay.js";

// sCrypt contracts — BRC-121 on-chain enforcement (types 01/02/03)
export {
  ARIAEpochContract,
  EPOCH_BOND_SAT,
  ARIATimelockContract,
  MIN_TIMELOCK_BLOCKS,
  ARIARegistryContract,
  EUAIActRisk,
  MAX_MODEL_HASHES,
  MAX_SYSTEM_ID_BYTES,
  toSha256,
  toPubKeyHash,
  toByteString,
  toBlockHeight,
  buildOpReturnScript,
  ARIA_MAGIC,
  BRC121_VERSION,
} from "./contracts/index.js";
export type {
  EpochContractState,
  EpochUnlockWitness,
  TimelockContractState,
  TimelockNormalWitness,
  TimelockRecoveryWitness,
  TimelockWitness,
  RegistryEntry,
  RegistryVerifyResult,
  ContractVerifyResult,
  Sha256,
  PubKeyHash,
  ByteString,
  BlockHeight,
} from "./contracts/index.js";
