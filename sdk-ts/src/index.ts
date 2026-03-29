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
