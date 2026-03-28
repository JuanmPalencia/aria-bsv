"""aria.core — cryptographic primitives and protocol logic for ARIA (BRC-121)."""

from aria.core.epoch import EpochConfig, EpochManager, EpochOpenResult, EpochCloseResult
from aria.core.errors import (
    ARIABroadcastError,
    ARIAConfigError,
    ARIAError,
    ARIASerializationError,
    ARIAStorageError,
    ARIATamperDetected,
    ARIAVerificationError,
    ARIAWalletError,
)
from aria.core.hasher import canonical_json, hash_file, hash_object
from aria.core.merkle import ARIAMerkleTree, MerkleProof, verify_proof
from aria.core.record import ARIA_VERSION, AuditRecord

__all__ = [
    # epoch
    "EpochConfig",
    "EpochManager",
    "EpochOpenResult",
    "EpochCloseResult",
    # errors
    "ARIAError",
    "ARIAConfigError",
    "ARIASerializationError",
    "ARIAWalletError",
    "ARIABroadcastError",
    "ARIAStorageError",
    "ARIAVerificationError",
    "ARIATamperDetected",
    # hasher
    "canonical_json",
    "hash_object",
    "hash_file",
    # record
    "ARIA_VERSION",
    "AuditRecord",
    # merkle
    "ARIAMerkleTree",
    "MerkleProof",
    "verify_proof",
]
