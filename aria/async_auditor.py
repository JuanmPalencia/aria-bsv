"""
aria.async_auditor — Native async InferenceAuditor for asyncio applications.

Drop-in async replacement for InferenceAuditor. Uses asyncio.Lock for
thread-safety, asyncio.Task for background epoch flushing, and native
await-able record() and flush() methods.

Usage::

    async with AsyncInferenceAuditor(config, model_hashes) as auditor:
        receipt = await auditor.record("gpt-4o", input_data, output_data)
        # ...
    # flush() called automatically on __aexit__
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

from .auditor import AuditConfig, Receipt
from .broadcaster.arc import ARCBroadcaster
from .broadcaster.base import BroadcasterInterface
from .core.epoch import EpochConfig, EpochManager, EpochOpenResult
from .core.errors import ARIAError
from .core.hasher import hash_object
from .core.record import AuditRecord
from .storage.base import StorageInterface
from .storage.sqlite import SQLiteStorage
from .wallet.base import WalletInterface
from .wallet.brc100 import BRC100Wallet
from .wallet.direct import DirectWallet

_log = logging.getLogger(__name__)


class AsyncInferenceAuditor:
    """Native async auditor for asyncio / FastAPI applications.

    All epoch state is protected by ``asyncio.Lock``.  BSV I/O (wallet and
    broadcaster) is synchronous internally but is dispatched via
    ``asyncio.to_thread`` so the event loop is never blocked.

    A background ``asyncio.Task`` drives the periodic auto-flush when the
    auditor is used as an async context manager.  Manual usage without the
    context manager is supported — call ``flush()`` explicitly to close
    the current epoch and broadcast to BSV.

    Args:
        config:          Full auditor configuration.
        model_hashes:    Mapping of model_id → ``"sha256:<hex>"`` for every
                         model that will run in each epoch.
        initial_state:   System operational state committed in EPOCH_OPEN.
                         Must be JSON-serialisable.  Do NOT include user PII.
        _wallet:         Injected wallet (tests only).
        _broadcaster:    Injected broadcaster (tests only).
        _storage:        Injected storage backend (tests only).
        _epoch_manager:  Injected epoch manager (tests only).
    """

    def __init__(
        self,
        config: AuditConfig,
        model_hashes: dict[str, str] | None = None,
        initial_state: dict[str, Any] | None = None,
        *,
        _wallet: WalletInterface | None = None,
        _broadcaster: BroadcasterInterface | None = None,
        _storage: StorageInterface | None = None,
        _epoch_manager: EpochManager | None = None,
    ) -> None:
        self._config = config
        self._model_hashes: dict[str, str] = model_hashes or {}
        self._initial_state: dict[str, Any] = initial_state or {}

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
        if _epoch_manager is not None:
            self._epoch_manager: EpochManager = _epoch_manager
        else:
            epoch_cfg = EpochConfig(
                system_id=config.system_id, network=config.network
            )
            self._epoch_manager = EpochManager(
                config=epoch_cfg,
                wallet=self._wallet,
                broadcaster=self._broadcaster,
            )

        # Epoch state — protected by _lock
        self._lock: asyncio.Lock = asyncio.Lock()
        self._pending_records: list[AuditRecord] = []
        self._current_open: EpochOpenResult | None = None
        self._sequence: int = 0

        # Background auto-flush task (created in __aenter__)
        self._auto_flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AsyncInferenceAuditor:
        await self._open_epoch()
        self._auto_flush_task = asyncio.create_task(
            self._auto_flush_loop(), name="aria-async-flush"
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._auto_flush_task is not None and not self._auto_flush_task.done():
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
            except asyncio.CancelledError:
                pass
            self._auto_flush_task = None
        await self.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record(
        self,
        model_id: str,
        input_data: Any,
        output_data: Any,
        *,
        confidence: float | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Receipt:
        """Record a single inference and return an audit Receipt.

        The input is sanitised (PII fields stripped) before hashing.
        The record is persisted to local storage before any BSV broadcast.

        Args:
            model_id:    Must match a key in the ``model_hashes`` dict.
            input_data:  Raw input to the model (only its hash is stored).
            output_data: Raw output of the model (only its hash is stored).
            confidence:  Optional model confidence score in [0.0, 1.0].
            latency_ms:  Inference wall-clock time in milliseconds.
            metadata:    Arbitrary extra context (JSON-serialisable).

        Returns:
            Receipt with record_id, epoch_id, open_txid, record_hash, model_id.
        """
        sanitised_input = self._sanitise(input_data)
        input_hash = hash_object(sanitised_input)
        output_hash = hash_object(output_data)
        latency_int = int(latency_ms) if latency_ms is not None else 0

        try:
            async with self._lock:
                # Ensure an epoch is open — open one lazily if needed.
                if self._current_open is None:
                    await self._open_epoch_locked()

                open_result = self._current_open
                if open_result is None:
                    raise ARIAError("no open epoch available")

                seq = self._sequence
                self._sequence += 1

                rec = AuditRecord(
                    epoch_id=open_result.epoch_id,
                    model_id=model_id,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    sequence=seq,
                    confidence=confidence,
                    latency_ms=latency_int,
                    metadata=metadata or {},
                )

                # Persist BEFORE adding to in-memory pending list (no-loss guarantee).
                await asyncio.to_thread(self._storage.save_record, rec)
                self._pending_records.append(rec)
                should_flush = len(self._pending_records) >= self._config.batch_size

            # Trigger flush outside the lock to avoid deadlock.
            if should_flush:
                await self.flush()

            epoch = await asyncio.to_thread(self._storage.get_epoch, rec.epoch_id)
            open_txid = epoch.open_txid if epoch else open_result.txid
            close_txid = epoch.close_txid if epoch else ""

            return Receipt(
                record_id=rec.record_id,
                epoch_id=rec.epoch_id,
                open_txid=open_txid,
                close_txid=close_txid,
                record_hash=rec.hash(),
                model_id=rec.model_id,
            )

        except Exception as exc:
            _log.error("AsyncInferenceAuditor.record failed: %s", exc)
            # Return a partial receipt so caller always gets something back.
            epoch_id = (self._current_open.epoch_id if self._current_open else "unknown")
            return Receipt(
                record_id=f"rec_{epoch_id}_error",
                epoch_id=epoch_id,
                open_txid="",
                close_txid="",
                record_hash="",
                model_id=model_id,
            )

    async def flush(self) -> None:
        """Close the current epoch and broadcast to BSV.

        Idempotent: if there are no pending records and no open epoch the
        call is a no-op.
        """
        async with self._lock:
            if self._current_open is None and not self._pending_records:
                return
            await self._close_epoch_locked()
            # Re-open a fresh epoch for subsequent records.
            await self._open_epoch_locked()

    async def get_receipt(self, record_id: str) -> Receipt | None:
        """Return the audit receipt for a previously recorded inference.

        Args:
            record_id: As returned by ``record()`` or the ``@track`` decorator.

        Returns:
            Receipt if found, None otherwise.
        """
        rec = await asyncio.to_thread(self._storage.get_record, record_id)
        if rec is None:
            return None

        epoch = await asyncio.to_thread(self._storage.get_epoch, rec.epoch_id)
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

    def track(self, model_id: str) -> Callable:  # type: ignore[type-arg]
        """Decorator that automatically records inputs and outputs.

        Only wraps **async** functions.  The decorated function's return value
        is unchanged.  If the audit record fails the error is logged and the
        original return value is still returned.

        Args:
            model_id: Model identifier committed in the EPOCH_OPEN.

        Example::

            @auditor.track("classifier")
            async def classify(image):
                return await model.predict(image)
        """

        def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"AsyncInferenceAuditor.track() requires an async function; "
                    f"got {func!r}.  Use InferenceAuditor for sync functions."
                )

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.monotonic()
                result = await func(*args, **kwargs)
                latency = (time.monotonic() - t0) * 1000
                try:
                    await self.record(
                        model_id,
                        args,
                        result,
                        latency_ms=latency,
                    )
                except Exception as exc:
                    _log.warning("ARIA async record failed in track(): %s", exc)
                return result

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _auto_flush_loop(self) -> None:
        """Background task — flush every ``batch_ms`` milliseconds."""
        interval = self._config.batch_ms / 1000.0
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.flush()
                except Exception as exc:
                    _log.error("Auto-flush error: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _open_epoch(self) -> None:
        """Open a new epoch (acquires lock internally)."""
        async with self._lock:
            await self._open_epoch_locked()

    async def _open_epoch_locked(self) -> None:
        """Open a new epoch.  Must be called while holding ``_lock``."""
        self._sequence = 0
        try:
            open_result = await self._epoch_manager.open_epoch(
                model_hashes=self._model_hashes,
                system_state=self._initial_state,
            )
            await asyncio.to_thread(
                self._storage.save_epoch_open,
                open_result.epoch_id,
                self._config.system_id,
                open_result.txid,
                open_result.model_hashes,
                open_result.state_hash,
                open_result.timestamp,
            )
            self._current_open = open_result
        except Exception as exc:
            _log.error("EPOCH_OPEN failed: %s", exc)
            self._current_open = None

    async def _close_epoch_locked(self) -> None:
        """Close the current epoch and broadcast EPOCH_CLOSE to BSV.

        Must be called while holding ``_lock``.
        """
        open_result = self._current_open
        records = list(self._pending_records)

        # Clear state immediately so new records land in the next epoch.
        self._current_open = None
        self._pending_records = []

        if open_result is None:
            return

        try:
            close_result = await self._epoch_manager.close_epoch(
                epoch_id=open_result.epoch_id,
                records=records,
            )
            await asyncio.to_thread(
                self._storage.save_epoch_close,
                open_result.epoch_id,
                close_result.txid,
                close_result.merkle_root,
                close_result.records_count,
                int(time.time()),
            )
        except Exception as exc:
            _log.error(
                "EPOCH_CLOSE failed for %s: %s", open_result.epoch_id, exc
            )

    def _sanitise(self, obj: Any) -> Any:
        """Remove PII fields from a dict copy before hashing."""
        if not self._config.pii_fields or not isinstance(obj, dict):
            return obj
        return {k: v for k, v in obj.items() if k not in self._config.pii_fields}
