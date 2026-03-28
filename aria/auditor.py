"""InferenceAuditor — main entry point for ARIA audit integration."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .broadcaster.arc import ARCBroadcaster
from .broadcaster.base import BroadcasterInterface
from .core.epoch import EpochConfig, EpochManager, EpochOpenResult
from .core.errors import ARIAConfigError, ARIAError
from .core.hasher import hash_object
from .core.record import AuditRecord
from .storage.base import StorageInterface
from .storage.sqlite import SQLiteStorage
from .wallet.base import WalletInterface
from .wallet.brc100 import BRC100Wallet
from .wallet.direct import DirectWallet
from .zk.aggregate import MerkleAggregator
from .zk.base import ProverInterface
from .zk.claims import Claim
from .zk.statement import EpochStatement

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditConfig:
    """Configuration for an InferenceAuditor instance.

    Exactly one of ``bsv_key`` or ``brc100_url`` must be set.

    Attributes:
        system_id:   Unique identifier for the AI system registered in ARIA.
        bsv_key:     WIF-encoded private key for DirectWallet signing.
                     Mutually exclusive with ``brc100_url``.
        brc100_url:  Endpoint of an external BRC-100 wallet service.
                     Mutually exclusive with ``bsv_key``.
        storage:     SQLAlchemy DSN for local persistence.
                     Defaults to ``"sqlite:///aria.db"`` in the current directory.
        batch_ms:    Maximum epoch duration in milliseconds before auto-close.
        batch_size:  Maximum records per epoch before early close.
        arc_url:     Base URL of the TAAL ARC broadcaster.
        arc_api_key: Optional Bearer token for authenticated ARC access.
        network:     ``"mainnet"`` or ``"testnet"``.
        pii_fields:  Input dict keys that are redacted before hashing
                     (the key is removed from the dict copy; the hash is computed
                     on the sanitised version only — original data is unchanged).
    """

    system_id: str
    bsv_key: str | None = None
    brc100_url: str | None = None
    storage: str = "sqlite:///aria.db"
    batch_ms: int = 5_000
    batch_size: int = 500
    arc_url: str = "https://arc.taal.com"
    arc_api_key: str | None = None
    network: str = "mainnet"
    pii_fields: list[str] = field(default_factory=list)
    # ZK extension — all optional.
    zk_prover: ProverInterface | None = None
    model_paths: dict[str, str] = field(default_factory=dict)
    zk_claims: list[Claim] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ARIAConfigError("system_id must not be empty")
        if self.bsv_key is None and self.brc100_url is None:
            raise ARIAConfigError("one of bsv_key or brc100_url must be set")
        if self.bsv_key is not None and self.brc100_url is not None:
            raise ARIAConfigError("bsv_key and brc100_url are mutually exclusive")
        if self.batch_ms <= 0:
            raise ARIAConfigError("batch_ms must be positive")
        if self.batch_size <= 0:
            raise ARIAConfigError("batch_size must be positive")
        if self.network not in ("mainnet", "testnet"):
            raise ARIAConfigError("network must be 'mainnet' or 'testnet'")

    def __repr__(self) -> str:
        key_display = "<set>" if self.bsv_key is not None else "None"
        return (
            f"AuditConfig(system_id={self.system_id!r}, bsv_key={key_display}, "
            f"brc100_url={self.brc100_url!r}, network={self.network!r}, "
            f"storage={self.storage!r})"
        )


@dataclass
class Receipt:
    """Audit receipt for a single recorded inference.

    Attributes:
        record_id:   Unique ID of the record (``rec_{epoch_id}_{seq:06d}``).
        epoch_id:    Epoch the record belongs to.
        open_txid:   BSV txid of the EPOCH_OPEN (set after broadcast).
        close_txid:  BSV txid of the EPOCH_CLOSE (empty until epoch closes).
        record_hash: SHA-256 hash of the canonical AuditRecord.
        model_id:    Model that produced this record.
    """

    record_id: str
    epoch_id: str
    open_txid: str
    close_txid: str
    record_hash: str
    model_id: str


# ---------------------------------------------------------------------------
# Internal BatchManager
# ---------------------------------------------------------------------------


class _BatchManager:
    """Manages epoch lifecycle in a dedicated asyncio background thread.

    Guarantees:
      - Records are persisted to storage BEFORE any BSV broadcast.
      - EPOCH_OPEN is broadcast before the first record is handed back to the caller.
      - EPOCH_CLOSE is triggered after ``batch_ms`` or ``batch_size`` records.
    """

    def __init__(
        self,
        epoch_manager: EpochManager,
        storage: StorageInterface,
        config: AuditConfig,
        model_hashes: dict[str, str],
        initial_state: dict[str, Any],
    ) -> None:
        self._epoch_manager = epoch_manager
        self._storage = storage
        self._config = config
        self._model_hashes = model_hashes
        self._initial_state = initial_state

        self._lock = threading.Lock()
        self._pending: list[AuditRecord] = []
        self._pending_proofs: list[Any] = []   # ZKProof objects if ZK enabled
        self._current_open: EpochOpenResult | None = None
        self._sequence: int = 0
        self._proving_keys: dict[str, Any] = {}  # model_id -> ProvingKey

        # Signal to flush early (batch_size reached).
        self._flush_now = False

        # The background event loop lives in its own thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="aria-batch", daemon=True
        )
        self._thread.start()

        # Epoch-ready event: set when EPOCH_OPEN is confirmed and callers can submit records.
        self._epoch_ready = threading.Event()

        # Start the lifecycle coroutine.
        asyncio.run_coroutine_threadsafe(self._lifecycle(), self._loop)

    # ------------------------------------------------------------------
    # Public (called from main thread)
    # ------------------------------------------------------------------

    def new_record(self, model_id: str, input_hash: str, output_hash: str,
                   confidence: float | None, latency_ms: int,
                   metadata: dict[str, Any]) -> AuditRecord:
        """Create, persist, and queue an AuditRecord. Thread-safe."""
        # Wait until the current epoch is open on BSV.
        self._epoch_ready.wait(timeout=30.0)

        with self._lock:
            open_result = self._current_open
            if open_result is None:
                raise ARIAError("no open epoch available — auditor may be shutting down")
            seq = self._sequence
            self._sequence += 1

        record = AuditRecord(
            epoch_id=open_result.epoch_id,
            model_id=model_id,
            input_hash=input_hash,
            output_hash=output_hash,
            sequence=seq,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        # Persist BEFORE adding to in-memory queue (no-loss guarantee).
        self._storage.save_record(record)

        with self._lock:
            self._pending.append(record)
            flush = len(self._pending) >= self._config.batch_size

        if flush:
            self._loop.call_soon_threadsafe(self._set_flush)

        return record

    def flush(self) -> None:
        """Request an immediate epoch close (blocks until the close is done)."""
        done_event = threading.Event()

        async def _do_flush() -> None:
            await self._close_current_epoch()
            await self._open_new_epoch()
            done_event.set()

        asyncio.run_coroutine_threadsafe(_do_flush(), self._loop)
        done_event.wait(timeout=60.0)

    def stop(self) -> None:
        """Flush remaining records and stop the background loop."""
        self.flush()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)

    # ------------------------------------------------------------------
    # Background loop helpers
    # ------------------------------------------------------------------

    def _set_flush(self) -> None:
        self._flush_now = True

    async def _lifecycle(self) -> None:
        """Main epoch loop — runs forever in the background thread."""
        await self._open_new_epoch()
        while True:
            # Wait for batch_ms, checking early-flush flag every 100ms.
            deadline = time.monotonic() + self._config.batch_ms / 1000
            while time.monotonic() < deadline:
                if self._flush_now:
                    break
                await asyncio.sleep(0.1)
            self._flush_now = False

            await self._close_current_epoch()
            await self._open_new_epoch()

    async def _open_new_epoch(self) -> None:
        self._epoch_ready.clear()
        with self._lock:
            self._sequence = 0

        # ZK setup: compile circuits and generate keys (once per model, cached).
        if self._config.zk_prover and not self._proving_keys:
            for model_id, model_path in self._config.model_paths.items():
                try:
                    pk, vk = await self._config.zk_prover.setup(model_id, model_path)
                    self._proving_keys[model_id] = pk
                    self._storage.save_vk(vk)
                except Exception as exc:
                    _log.warning("ZK setup failed for model %s: %s", model_id, exc)

        try:
            open_result = await self._epoch_manager.open_epoch(
                model_hashes=self._model_hashes,
                system_state=self._initial_state,
            )
            self._storage.save_epoch_open(
                epoch_id=open_result.epoch_id,
                system_id=self._config.system_id,
                open_txid=open_result.txid,
                model_hashes=open_result.model_hashes,
                state_hash=open_result.state_hash,
                opened_at=open_result.timestamp,
            )
        except Exception as exc:
            _log.error("EPOCH_OPEN failed: %s", exc)
            # Keep _epoch_ready cleared — new_record() will time out gracefully.
            return

        with self._lock:
            self._current_open = open_result
        self._epoch_ready.set()

    async def _close_current_epoch(self) -> None:
        with self._lock:
            open_result = self._current_open
            records = list(self._pending)
            proofs = list(self._pending_proofs)
            self._pending.clear()
            self._pending_proofs.clear()
            self._current_open = None

        self._epoch_ready.clear()

        if open_result is None:
            return

        # Build EpochStatement (ZK claims + aggregate proof).
        statement: EpochStatement | None = None
        if self._config.zk_claims or self._config.zk_prover:
            statement = await self._build_statement(
                open_result.epoch_id, records, proofs, open_result.txid
            )

        try:
            close_result = await self._epoch_manager.close_epoch(
                epoch_id=open_result.epoch_id,
                records=records,
                statement=statement,
            )
            self._storage.save_epoch_close(
                epoch_id=open_result.epoch_id,
                close_txid=close_result.txid,
                merkle_root=close_result.merkle_root,
                records_count=close_result.records_count,
                closed_at=int(time.time()),
            )
        except Exception as exc:
            _log.error("EPOCH_CLOSE failed for %s: %s", open_result.epoch_id, exc)

    async def _build_statement(
        self,
        epoch_id: str,
        records: list[AuditRecord],
        proofs: list[Any],
        open_txid: str,
    ) -> EpochStatement:
        """Evaluate claims and aggregate proofs into an EpochStatement."""
        import datetime

        claim_results = [c.evaluate(records) for c in self._config.zk_claims]

        aggregate = None
        if proofs:
            agg = MerkleAggregator()
            aggregate = agg.aggregate(proofs, epoch_id)

        return EpochStatement(
            epoch_id=epoch_id,
            system_id=self._config.system_id,
            claims=claim_results,
            aggregate_proof=aggregate,
            open_txid=open_txid,
            closed_at=datetime.datetime.now(datetime.timezone.utc),
            n_records=len(records),
        )

    async def _prove_record(
        self,
        record: AuditRecord,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """Generate ZK proof for a single record and persist it."""
        prover = self._config.zk_prover
        if prover is None:
            return
        pk = self._proving_keys.get(record.model_id)
        if pk is None:
            _log.debug("No proving key for model %s — skipping ZK proof", record.model_id)
            return
        try:
            proof = await prover.prove(
                model_id=record.model_id,
                input_data=input_data if isinstance(input_data, dict) else {"input": input_data},
                output_data=output_data if isinstance(output_data, dict) else {"output": output_data},
                pk=pk,
                record_id=record.record_id,
                epoch_id=record.epoch_id,
            )
            self._storage.save_proof(proof)
            with self._lock:
                self._pending_proofs.append(proof)
        except Exception as exc:
            _log.warning("ZK prove failed for record %s: %s", record.record_id, exc)


# ---------------------------------------------------------------------------
# InferenceAuditor
# ---------------------------------------------------------------------------


class InferenceAuditor:
    """Primary SDK entry point for auditing AI inferences.

    Creates and manages all internal components (wallet, broadcaster, storage,
    epoch manager, batch manager) from a single ``AuditConfig`` object.

    Usage (manual)::

        config = AuditConfig(system_id="kairos-v2", bsv_key=os.environ["BSV_WIF"])
        auditor = InferenceAuditor(config, model_hashes={"triage": hash_file("triage.pkl")})
        record_id = auditor.record("triage", raw_input, raw_output, confidence=0.97)

    Usage (decorator)::

        @auditor.track("triage")
        def run_triage(patient_data):
            return model.predict(patient_data)

    Args:
        config:         Full auditor configuration.
        model_hashes:   Mapping of model_id → ``"sha256:<hex>"`` for every model
                        that will run in each epoch.  Use ``hash_file()`` to compute.
        initial_state:  System operational state committed in EPOCH_OPEN.
                        Must be JSON-serialisable.  Do NOT include user PII.
    """

    def __init__(
        self,
        config: AuditConfig,
        model_hashes: dict[str, str] | None = None,
        initial_state: dict[str, Any] | None = None,
        *,
        # Dependency injection for tests — if provided, skips auto-construction.
        _wallet: WalletInterface | None = None,
        _broadcaster: BroadcasterInterface | None = None,
        _storage: StorageInterface | None = None,
    ) -> None:
        self._config = config
        self._record_hooks: list[Callable[[AuditRecord], None]] = []

        # Storage
        self._storage: StorageInterface = _storage or SQLiteStorage(dsn=config.storage)

        # Broadcaster
        self._broadcaster: BroadcasterInterface = _broadcaster or ARCBroadcaster(
            api_url=config.arc_url,
            api_key=config.arc_api_key,
        )

        # Wallet
        if _wallet is not None:
            self._wallet: WalletInterface = _wallet
        elif config.bsv_key is not None:
            self._wallet = DirectWallet(
                wif=config.bsv_key,
                broadcaster=self._broadcaster,
                network=config.network,
            )
        else:
            assert config.brc100_url is not None
            self._wallet = BRC100Wallet(endpoint=config.brc100_url)

        # EpochManager
        epoch_cfg = EpochConfig(system_id=config.system_id, network=config.network)
        self._epoch_manager = EpochManager(
            config=epoch_cfg,
            wallet=self._wallet,
            broadcaster=self._broadcaster,
        )

        # BatchManager starts background loop immediately.
        self._batch = _BatchManager(
            epoch_manager=self._epoch_manager,
            storage=self._storage,
            config=config,
            model_hashes=model_hashes or {},
            initial_state=initial_state or {},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        model_id: str,
        input: Any,
        output: Any,
        confidence: float | None = None,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a single inference.  Thread-safe.  Returns the record_id.

        Args:
            model_id:    Must match a key in the ``model_hashes`` dict.
            input:       Raw input to the model (only the hash is stored / sent).
            output:      Raw output of the model (only the hash is stored / sent).
            confidence:  Optional model confidence score in [0.0, 1.0].
            latency_ms:  Inference wall-clock time in milliseconds.
            metadata:    Arbitrary extra context (JSON-serialisable).

        Returns:
            The ``record_id`` (``"rec_{epoch_id}_{seq:06d}"``).
        """
        sanitised_input = self._sanitise(input)
        input_hash = hash_object(sanitised_input)
        output_hash = hash_object(output)
        rec = self._batch.new_record(
            model_id=model_id,
            input_hash=input_hash,
            output_hash=output_hash,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        # Fire record hooks (events, metrics, telemetry) — never block the caller.
        for hook in self._record_hooks:
            try:
                hook(rec)
            except Exception as exc:
                _log.warning("record hook error (%s): %s", getattr(hook, "__name__", hook), exc)

        # Schedule ZK proof generation asynchronously (non-blocking).
        if self._config.zk_prover:
            asyncio.run_coroutine_threadsafe(
                self._batch._prove_record(rec, sanitised_input, output),
                self._batch._loop,
            )
        return rec.record_id

    def track(self, model_id: str) -> Callable:  # type: ignore[type-arg]
        """Decorator that automatically records inputs and outputs.

        Works with both synchronous and asynchronous functions.
        Does not alter the decorated function's return value.

        Args:
            model_id: Model identifier committed in the EPOCH_OPEN.

        Example::

            @auditor.track("classifier")
            def classify(image):
                return model.predict(image)
        """

        def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.monotonic()
                    result = await func(*args, **kwargs)
                    latency = int((time.monotonic() - t0) * 1000)
                    try:
                        self.record(model_id, args, result,
                                    latency_ms=latency)
                    except Exception as exc:
                        _log.warning("ARIA record failed (async): %s", exc)
                    return result

                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.monotonic()
                    result = func(*args, **kwargs)
                    latency = int((time.monotonic() - t0) * 1000)
                    try:
                        self.record(model_id, args, result,
                                    latency_ms=latency)
                    except Exception as exc:
                        _log.warning("ARIA record failed: %s", exc)
                    return result

                return sync_wrapper

        return decorator

    def get_receipt(self, record_id: str) -> Receipt:
        """Return the audit receipt for a previously recorded inference.

        Args:
            record_id: As returned by ``record()`` or the ``@track`` decorator.

        Returns:
            Receipt with record_id, epoch_id, open_txid, close_txid.

        Raises:
            ARIAError: If the record_id is not found in local storage.
        """
        rec = self._storage.get_record(record_id)
        if rec is None:
            raise ARIAError(f"record {record_id!r} not found in local storage")

        epoch = self._storage.get_epoch(rec.epoch_id)
        open_txid = epoch.open_txid if epoch else ""
        close_txid = epoch.close_txid if epoch else ""

        return Receipt(
            record_id=rec.record_id,
            epoch_id=rec.epoch_id,
            open_txid=open_txid,
            close_txid=close_txid,
            record_hash=rec.hash(),
            model_id=rec.model_id,
        )

    def add_record_hook(self, hook: Callable[[AuditRecord], None]) -> None:
        """Register a callback called after every successful ``record()``.

        Hooks receive the raw ``AuditRecord`` object.  Common uses::

            auditor.add_record_hook(bus.emit_record)       # event bus
            auditor.add_record_hook(metrics.on_record)     # prometheus
            auditor.add_record_hook(otel.on_record)        # opentelemetry

        Exceptions raised by hooks are caught and logged.
        """
        self._record_hooks.append(hook)

    @property
    def storage(self) -> StorageInterface:
        """Read-only access to the underlying storage backend."""
        return self._storage

    def flush(self) -> None:
        """Force an immediate epoch close and open a fresh one.

        Useful at application shutdown or before a model version change.
        Blocks until the close broadcast is confirmed.
        """
        self._batch.flush()

    def close(self) -> None:
        """Gracefully shut down the auditor — flushes pending records."""
        self._batch.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitise(self, obj: Any) -> Any:
        """Remove PII fields from a dict copy before hashing."""
        if not self._config.pii_fields or not isinstance(obj, dict):
            return obj
        return {k: v for k, v in obj.items() if k not in self._config.pii_fields}
