"""
aria — Auditable Real-time Inference Architecture (BRC-121 reference implementation).

Quick start (zero blockchain knowledge required)::

    from aria.quick import ARIAQuick

    with ARIAQuick("my-system") as aria:
        aria.record("gpt-4", {"prompt": "hello"}, {"text": "hi"}, confidence=0.99)
        summary = aria.close()
        print(summary)

Full API::

    from aria import AuditConfig, InferenceAuditor, SQLiteStorage
    from aria.compliance import ComplianceChecker
    from aria.drift import DriftDetector
    from aria.events import InMemoryEventBus, EventType
    from aria.alerts import SlackAlertChannel
    from aria.metrics import ARIAMetrics

Phase 0: Core cryptographic primitives.
Phase 1: EpochManager, wallet and broadcaster interfaces.
Phase 2: InferenceAuditor, AuditConfig, storage.
Phase 3–17: Verification, ZK proofs, analytics, watchdog, compliance,
             metrics, events, alerts, drift detection, ARIAQuick.
Phase 18–22: AI SDK integrations (OpenAI, Anthropic, HuggingFace, LlamaIndex),
             analytics (A/B testing, cost tracker, canary, lineage),
             regulatory tools (GDPR, model cards, regulatory export),
             infrastructure (shadow mode, replay, SIEM, multitenancy),
             multi-chain (Ethereum, Nostr), federation, MLflow/W&B.
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
from aria.auditor import AuditConfig, InferenceAuditor, Receipt
from aria.storage import EpochRow, SQLiteStorage, StorageInterface
from aria.verify import TxFetcher, VerificationResult, Verifier, WhatsOnChainFetcher
from aria.zk import (
    AggregateProof,
    AllModelsRegistered,
    Claim,
    ClaimResult,
    CommitmentProver,
    ConfidencePercentile,
    EpochStatement,
    EZKLProver,
    LatencyBound,
    MerkleAggregator,
    MockProver,
    ModelUnchanged,
    NoPIIInInputs,
    NovaAggregator,
    OutputDistribution,
    ProofAggregatorInterface,
    ProverInterface,
    ProverTier,
    ProvingKey,
    RecordCountRange,
    VerifyingKey,
    ZKProof,
)

__version__ = "0.5.0"

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
    # auditor (Phase 2)
    "AuditConfig",
    "InferenceAuditor",
    "Receipt",
    # storage
    "EpochRow",
    "SQLiteStorage",
    "StorageInterface",
    # verify (Phase 3)
    "TxFetcher",
    "VerificationResult",
    "Verifier",
    "WhatsOnChainFetcher",
]
