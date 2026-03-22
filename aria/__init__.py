"""
aria — Auditable Real-time Inference Architecture (BRC-120 reference implementation).

Phase 0 exports: core cryptographic primitives.
Phase 1 exports: EpochManager, wallet and broadcaster interfaces.
"""

from aria.core import (
    # Phase 1 — epoch lifecycle
    EpochConfig,
    EpochCloseResult,
    EpochManager,
    EpochOpenResult,
    # Phase 0 — errors
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
from aria.broadcaster import ARCBroadcaster, BroadcasterInterface, TxStatus
from aria.wallet import BRC100Wallet, DirectWallet, WalletInterface

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # epoch
    "EpochConfig",
    "EpochCloseResult",
    "EpochManager",
    "EpochOpenResult",
    # crypto
    "ARIA_VERSION",
    "AuditRecord",
    "ARIAMerkleTree",
    "MerkleProof",
    "verify_proof",
    "canonical_json",
    "hash_object",
    "hash_file",
    # errors
    "ARIAError",
    "ARIAConfigError",
    "ARIASerializationError",
    "ARIAWalletError",
    "ARIABroadcastError",
    "ARIAStorageError",
    "ARIAVerificationError",
    "ARIATamperDetected",
    # broadcaster
    "ARCBroadcaster",
    "BroadcasterInterface",
    "TxStatus",
    # wallet
    "BRC100Wallet",
    "DirectWallet",
    "WalletInterface",
]
