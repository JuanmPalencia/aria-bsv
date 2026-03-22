"""
aria — Auditable Real-time Inference Architecture (BRC-120 reference implementation).

Phase 0 exports (core cryptographic primitives):
"""

from aria.core import (
    ARIA_VERSION,
    ARIABroadcastError,
    ARIAConfigError,
    ARIAError,
    ARIAMerkleTree,
    ARIASerializationError,
    ARIAStorageError,
    ARIATamperDetected,
    ARIAVerificationError,
    ARIAWalletError,
    AuditRecord,
    MerkleProof,
    canonical_json,
    hash_file,
    hash_object,
    verify_proof,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ARIA_VERSION",
    "AuditRecord",
    "ARIAMerkleTree",
    "MerkleProof",
    "verify_proof",
    "canonical_json",
    "hash_object",
    "hash_file",
    "ARIAError",
    "ARIAConfigError",
    "ARIASerializationError",
    "ARIAWalletError",
    "ARIABroadcastError",
    "ARIAStorageError",
    "ARIAVerificationError",
    "ARIATamperDetected",
]
