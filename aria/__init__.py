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
Phase 23: Completeness features — auto_config, scheduled_epochs, query,
          selftest, export_bundle, notifications, compare, backup,
          dashboard, import_from, certify, reports.
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
from aria.auto_config import auto_config as auto_config_fn, auto_wallet, get_or_create_wif
from aria.scheduled_epochs import EpochScheduler, ScheduleConfig, parse_strategy
from aria.query import RecordQuery, QueryStats
from aria.selftest import selftest, SelftestReport
from aria.export_bundle import create_bundle, create_bundle_bytes
from aria.notifications import NotificationManager, Notification
from aria.compare import ModelComparator, ComparisonResult
from aria.backup import backup, restore, list_backups
from aria.dashboard import create_dashboard_app, serve as serve_dashboard
from aria.import_from import from_jsonl, from_openai_log, from_mlflow_export, from_wandb_export, save_imported
from aria.certify import Certifier, Certificate
from aria.reports import MultiReport, MultiEpochReport
from aria.config_file import ARIAProjectConfig, load_config, generate_config_template
from aria.retry_queue import RetryQueue, RetryWorker
from aria.offline import OfflineAuditor, OfflineEpochResult
from aria.pipeline import PipelineAuditor, PipelineTrace, PipelineStep
from aria.cost_estimator import CostEstimator, CostEstimate
from aria.jupyter import NotebookTracker, NotebookRecord
from aria.webhook_receiver import WebhookProcessor, WebhookEvent
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
    # auto_config
    "auto_config_fn",
    "auto_wallet",
    "get_or_create_wif",
    # scheduled_epochs
    "EpochScheduler",
    "ScheduleConfig",
    "parse_strategy",
    # query
    "RecordQuery",
    "QueryStats",
    # selftest
    "selftest",
    "SelftestReport",
    # export_bundle
    "create_bundle",
    "create_bundle_bytes",
    # notifications
    "NotificationManager",
    "Notification",
    # compare
    "ModelComparator",
    "ComparisonResult",
    # backup
    "backup",
    "restore",
    "list_backups",
    # dashboard
    "create_dashboard_app",
    "serve_dashboard",
    # import_from
    "from_jsonl",
    "from_openai_log",
    "from_mlflow_export",
    "from_wandb_export",
    "save_imported",
    # certify
    "Certifier",
    "Certificate",
    # reports
    "MultiReport",
    "MultiEpochReport",
    # config_file
    "ARIAProjectConfig",
    "load_config",
    "generate_config_template",
    # retry_queue
    "RetryQueue",
    "RetryWorker",
    # offline
    "OfflineAuditor",
    "OfflineEpochResult",
    # pipeline
    "PipelineAuditor",
    "PipelineTrace",
    "PipelineStep",
    # cost_estimator
    "CostEstimator",
    "CostEstimate",
    # jupyter
    "NotebookTracker",
    "NotebookRecord",
    # webhook_receiver
    "WebhookProcessor",
    "WebhookEvent",
]
