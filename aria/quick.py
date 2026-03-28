"""
aria.quick — Zero-blockchain-knowledge API for ARIA.

ARIAQuick wraps every ARIA subsystem behind a minimal interface so you can
audit AI models on BSV with almost no blockchain knowledge.

What it handles automatically
--------------------------------
- SQLite database creation and management
- Epoch lifecycle (open → inferences → close → anchor)
- BSV transaction broadcasting (if a key is provided)
- Watchdog health monitoring
- Compliance checking
- Drift detection
- Metrics collection

Minimal usage (no BSV key required for local auditing)::

    from aria.quick import ARIAQuick

    with ARIAQuick("my-system") as aria:
        aria.record("gpt-4", {"prompt": "hello"}, {"text": "hi"}, confidence=0.99)
        aria.record("gpt-4", {"prompt": "bye"},   {"text": "cya"}, confidence=0.95)
        summary = aria.close()
        print(summary)

Decorator usage::

    with ARIAQuick("my-system") as aria:

        @aria.track("gpt-4")
        def chat(prompt: str) -> str:
            return llm.generate(prompt)

        answer = chat("What is BSV?")

Full usage with BSV anchoring::

    aria = ARIAQuick(
        system_id="prod-system",
        db_path="aria.db",
        bsv_wif="cNqK...",           # WIF private key for BSV testnet
        watchdog=True,
        compliance=True,
    )
    aria.start()
    ...
    aria.stop()

One-shot function::

    from aria.quick import quick_audit

    summary = quick_audit("my-app", [
        {"model_id": "gpt-4", "input_data": {"q": "hi"}, "output_data": {"a": "hello"}},
    ])
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal no-op wallet for local-only mode
# ---------------------------------------------------------------------------

class _LocalWallet:
    """No-op wallet for local-only mode (no BSV key required)."""

    _counter = 0

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        _LocalWallet._counter += 1
        import hashlib
        raw = f"local-{_LocalWallet._counter}-{time.time()}".encode()
        return "local-" + hashlib.sha256(raw).hexdigest()[:60]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EpochSummary:
    """Summary returned by ``ARIAQuick.close()``."""
    epoch_id:        str
    system_id:       str
    records_count:   int
    open_txid:       str
    close_txid:      str
    merkle_root:     str
    anchored:        bool
    compliant:       bool | None        # None if compliance checking disabled
    compliance_violations: list[str] = field(default_factory=list)
    elapsed_ms:      float = 0.0

    def __str__(self) -> str:
        status = "ANCHORED" if self.anchored else "LOCAL-ONLY"
        compliance = (
            "COMPLIANT" if self.compliant
            else f"NON-COMPLIANT ({len(self.compliance_violations)} issues)"
            if self.compliant is not None
            else "NOT-CHECKED"
        )
        return (
            f"EpochSummary [{status}] [{compliance}]\n"
            f"  epoch_id:    {self.epoch_id}\n"
            f"  system_id:   {self.system_id}\n"
            f"  records:     {self.records_count}\n"
            f"  open_txid:   {self.open_txid or '(none)'}\n"
            f"  close_txid:  {self.close_txid or '(none)'}\n"
            f"  merkle_root: {self.merkle_root[:16] if self.merkle_root else ''}...\n"
            f"  elapsed_ms:  {self.elapsed_ms:.1f}"
        )


@dataclass
class DriftSummary:
    """Summary returned by ``ARIAQuick.check_drift()``."""
    epoch_a:        str
    epoch_b:        str
    test:           str
    statistic:      float
    threshold:      float
    drift_detected: bool
    detail:         str

    def __str__(self) -> str:
        flag = "DRIFT DETECTED" if self.drift_detected else "No drift"
        return (
            f"DriftSummary [{flag}]\n"
            f"  {self.epoch_a} → {self.epoch_b}\n"
            f"  {self.test.upper()} statistic: {self.statistic:.4f} "
            f"(threshold: {self.threshold})"
        )


# ---------------------------------------------------------------------------
# ARIAQuick
# ---------------------------------------------------------------------------

class ARIAQuick:
    """Zero-blockchain-knowledge ARIA API.

    Args:
        system_id:       Your system/application name.
        db_path:         SQLite database path.  Default: ``aria_<system_id>.db``.
        bsv_wif:         WIF-encoded BSV private key for on-chain anchoring.
                         If None, epochs are recorded locally without BSV txs.
        watchdog:        Start background health monitoring (default False).
        compliance:      Run EU AI Act + BRC-121 compliance on close (default True).
        drift_threshold: JS-divergence threshold for drift alerts (default 0.10).
        batch_ms:        Auto-close epoch after this many milliseconds (default 3600000).
        batch_size:      Auto-close epoch after this many records (default 100000).
    """

    def __init__(
        self,
        system_id: str,
        db_path: str | None = None,
        bsv_wif: str | None = None,
        watchdog: bool = False,
        compliance: bool = True,
        drift_threshold: float = 0.10,
        batch_ms: int = 3_600_000,   # 1 hour — callers call close() explicitly
        batch_size: int = 100_000,
    ) -> None:
        self._system_id = system_id
        self._db_path = db_path or f"aria_{system_id}.db"
        self._bsv_wif = bsv_wif
        self._watchdog_enabled = watchdog
        self._compliance_enabled = compliance
        self._drift_threshold = drift_threshold
        self._batch_ms = batch_ms
        self._batch_size = batch_size

        self._started = False
        self._current_epoch_id: str | None = None
        self._closed_epoch_ids: list[str] = []

        # Subsystems (lazy)
        self._auditor: Any = None
        self._storage: Any = None
        self._watchdog: Any = None
        self._compliance_checker: Any = None

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def start(self) -> "ARIAQuick":
        """Initialise all subsystems.

        Returns self for chaining::

            aria = ARIAQuick("system").start()
        """
        if self._started:
            return self
        self._init_subsystems()
        self._started = True
        return self

    def stop(self) -> None:
        """Flush the current batch and stop background services."""
        if not self._started:
            return
        if self._auditor:
            try:
                self._auditor.close()
            except Exception as exc:
                _log.warning("ARIAQuick.stop: error closing auditor: %s", exc)
        if self._watchdog:
            try:
                self._watchdog.stop()
            except Exception:
                pass
        self._started = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ARIAQuick":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(
        self,
        model_id: str,
        input_data: Any,
        output_data: Any,
        confidence: float | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Audit a single inference.

        Args:
            model_id:    Identifier for the model (e.g. ``"gpt-4"``).
            input_data:  Model input — any JSON-serialisable object.
            output_data: Model output — any JSON-serialisable object.
            confidence:  Optional confidence score [0.0, 1.0].
            latency_ms:  Inference latency in milliseconds.
            metadata:    Optional extra key-value pairs.

        Returns:
            record_id (str) — unique identifier for this inference record.
        """
        if not self._started:
            self.start()
        return self._auditor.record(
            model_id,
            input_data,
            output_data,
            confidence=confidence,
            latency_ms=int(latency_ms or 0),
            metadata=metadata or {},
        )

    def close(self) -> EpochSummary:
        """Flush the current epoch to storage and run compliance checks.

        Internally calls ``InferenceAuditor.flush()`` which:
        1. Closes the current epoch (writes Merkle root, broadcasts BSV tx).
        2. Opens a fresh epoch for the next batch.

        Returns:
            EpochSummary with epoch metadata and compliance status.
        """
        if not self._started:
            self.start()

        t0 = time.time()

        # Capture epoch before flush (it'll be in storage as the latest open/pending)
        epoch_before = self._current_epoch_id

        # Flush: close current epoch, open new one
        self._auditor.flush()
        elapsed_ms = (time.time() - t0) * 1000

        # After flush, read the most recently CLOSED epoch
        epochs = self._storage.list_epochs(system_id=self._system_id, limit=50)

        # Find the closed epoch (has close_txid or is the epoch we tracked)
        epoch_row = None
        for ep in epochs:
            if ep.close_txid is not None:
                if epoch_before is None or ep.epoch_id == epoch_before:
                    epoch_row = ep
                    break

        # Fallback: first closed epoch
        if epoch_row is None:
            for ep in epochs:
                if ep.close_txid is not None:
                    epoch_row = ep
                    break

        if epoch_row is not None:
            self._closed_epoch_ids.append(epoch_row.epoch_id)

        # Compliance check
        compliant: bool | None = None
        violations: list[str] = []
        if self._compliance_enabled and self._compliance_checker and epoch_row:
            report = self._compliance_checker.check_epoch(epoch_row.epoch_id)
            compliant = report.passed
            violations = report.violations

        if epoch_row:
            return EpochSummary(
                epoch_id=epoch_row.epoch_id,
                system_id=self._system_id,
                records_count=epoch_row.records_count,
                open_txid=epoch_row.open_txid or "",
                close_txid=epoch_row.close_txid or "",
                merkle_root=epoch_row.merkle_root or "",
                anchored=bool(
                    epoch_row.close_txid
                    and not epoch_row.close_txid.startswith("local-")
                    and epoch_row.close_txid != "pending"
                ),
                compliant=compliant,
                compliance_violations=violations,
                elapsed_ms=elapsed_ms,
            )

        # No epoch found in storage — return minimal summary
        return EpochSummary(
            epoch_id=epoch_before or "",
            system_id=self._system_id,
            records_count=0,
            open_txid="", close_txid="", merkle_root="",
            anchored=False, compliant=compliant,
            compliance_violations=violations,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # track decorator
    # ------------------------------------------------------------------

    def track(
        self,
        model_id: str,
        capture_inputs: bool = True,
        capture_outputs: bool = True,
    ) -> Callable:
        """Decorator — automatically audit every call to the wrapped function.

        Args:
            model_id:        Model identifier string.
            capture_inputs:  Store function args as input_data (default True).
            capture_outputs: Store return value as output_data (default True).

        Example::

            @aria.track("my-llm")
            def generate(prompt: str) -> str:
                return llm.run(prompt)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t0 = time.time()
                result = func(*args, **kwargs)
                latency_ms = (time.time() - t0) * 1000

                input_data  = {"args": args, "kwargs": kwargs} if capture_inputs else {}
                output_data = result if capture_outputs else {}

                try:
                    self.record(
                        model_id=model_id,
                        input_data=input_data,
                        output_data=output_data,
                        latency_ms=latency_ms,
                    )
                except Exception as exc:
                    _log.warning("ARIAQuick.track: record error: %s", exc)

                return result
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self,
        epoch_a: str | None = None,
        epoch_b: str | None = None,
        test: str = "js",
    ) -> DriftSummary:
        """Compare two epochs for distributional drift in confidence scores.

        If epoch_a / epoch_b are not provided, the two most recent closed
        epochs are used automatically.

        Args:
            epoch_a: Reference (baseline) epoch ID.
            epoch_b: Current epoch ID to compare against baseline.
            test:    Statistical test: ``"js"`` (default), ``"ks"``, ``"kl"``.

        Returns:
            DriftSummary describing whether drift was detected.

        Raises:
            RuntimeError: if fewer than 2 closed epochs are available.
        """
        if not self._started:
            self.start()

        from .drift import DriftDetector

        if epoch_a is None or epoch_b is None:
            candidates = self._closed_epoch_ids[-2:]
            if len(candidates) < 2:
                epochs = self._storage.list_epochs(
                    system_id=self._system_id, limit=10
                )
                closed = [e.epoch_id for e in epochs if e.close_txid is not None]
                closed = list(reversed(closed))   # chronological
                candidates = closed[-2:]
            if len(candidates) < 2:
                raise RuntimeError(
                    "Need at least 2 closed epochs to check drift. "
                    "Call close() at least twice first."
                )
            epoch_a, epoch_b = candidates[0], candidates[1]

        detector = DriftDetector(
            self._storage,
            threshold=self._drift_threshold,
            test=test,
        )
        result = detector.compare(epoch_a, epoch_b)
        return DriftSummary(
            epoch_a=result.epoch_a,
            epoch_b=result.epoch_b,
            test=result.test,
            statistic=result.statistic,
            threshold=result.threshold,
            drift_detected=result.drift_detected,
            detail=result.detail,
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self, last_n: int = 5) -> str:
        """Return a human-readable summary of recent epochs."""
        if not self._started:
            self.start()

        epochs = self._storage.list_epochs(system_id=self._system_id, limit=last_n)
        if not epochs:
            return f"ARIAQuick '{self._system_id}': no epochs yet."

        lines = [
            f"ARIAQuick Summary — system: {self._system_id}",
            f"  DB: {self._db_path}",
            f"  Epochs (last {len(epochs)}):",
        ]
        for e in epochs:
            state = "open" if not e.close_txid else "closed"
            anchored = (
                "anchored"
                if (e.close_txid and not e.close_txid.startswith("local-") and e.close_txid != "pending")
                else "local"
            )
            lines.append(
                f"    {e.epoch_id[:20]}...  {state}  {anchored}  "
                f"{e.records_count} records"
            )
        return "\n".join(lines)

    def compliance_report(self, epoch_id: str | None = None) -> str:
        """Return a text compliance report for an epoch (default: last closed)."""
        if not self._compliance_checker:
            return "Compliance checking is disabled."

        if epoch_id is None:
            if self._closed_epoch_ids:
                epoch_id = self._closed_epoch_ids[-1]
            else:
                epochs = self._storage.list_epochs(system_id=self._system_id, limit=5)
                closed = [e for e in epochs if e.close_txid is not None]
                if not closed:
                    return "No closed epochs found."
                epoch_id = closed[0].epoch_id

        report = self._compliance_checker.check_epoch(epoch_id)
        return report.to_text()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_epoch_id(self) -> str | None:
        """ID of the currently open epoch, or None if not started."""
        if not self._started or self._auditor is None:
            return None
        # Access the internal batch manager's open epoch
        try:
            batch = self._auditor._batch
            if batch._current_open is not None:
                ep_id = batch._current_open.epoch_id
                self._current_epoch_id = ep_id
                return ep_id
        except Exception:
            pass
        return self._current_epoch_id

    @property
    def system_id(self) -> str:
        return self._system_id

    @property
    def storage(self):
        """Direct access to the underlying StorageInterface."""
        if not self._started:
            self.start()
        return self._storage

    @property
    def auditor(self):
        """Direct access to the underlying InferenceAuditor."""
        if not self._started:
            self.start()
        return self._auditor

    # ------------------------------------------------------------------
    # Private initialisation
    # ------------------------------------------------------------------

    def _init_subsystems(self) -> None:
        """Initialise all ARIA subsystems."""
        from .storage.sqlite import SQLiteStorage
        from .auditor import AuditConfig, InferenceAuditor
        from .broadcaster.base import BroadcasterInterface, TxStatus

        # Storage — use SQLite DSN
        dsn = (
            f"sqlite:///{self._db_path}"
            if not self._db_path.startswith("sqlite")
            else self._db_path
        )
        self._storage = SQLiteStorage(dsn)

        # Wallet — real or local no-op
        if self._bsv_wif:
            wallet = None   # Let InferenceAuditor build from WIF
            bsv_key = self._bsv_wif
        else:
            wallet = _LocalWallet()
            bsv_key = "local-mode"   # Satisfies AuditConfig validation

        # No-op broadcaster for local mode
        class _LocalBroadcaster(BroadcasterInterface):
            async def broadcast(self, raw_tx: str) -> TxStatus:
                return TxStatus(txid="local-" + raw_tx[:32], propagated=False)

        cfg = AuditConfig(
            system_id=self._system_id,
            bsv_key=bsv_key,
            batch_ms=self._batch_ms,
            batch_size=self._batch_size,
        )

        inject_kwargs: dict[str, Any] = {"_storage": self._storage}
        if wallet is not None:
            inject_kwargs["_wallet"] = wallet
            inject_kwargs["_broadcaster"] = _LocalBroadcaster()

        self._auditor = InferenceAuditor(cfg, **inject_kwargs)

        # Record hook: track current epoch_id
        def _epoch_hook(record: Any) -> None:
            self._current_epoch_id = record.epoch_id
        self._auditor.add_record_hook(_epoch_hook)

        # Compliance checker
        if self._compliance_enabled:
            from .compliance import ComplianceChecker
            self._compliance_checker = ComplianceChecker(self._storage)

        # Watchdog (optional)
        if self._watchdog_enabled:
            try:
                from .watchdog import WatchdogDaemon
                self._watchdog = WatchdogDaemon(
                    storage=self._storage,
                    system_id=self._system_id,
                )
                self._watchdog.start()
            except Exception as exc:
                _log.warning("ARIAQuick: could not start watchdog: %s", exc)


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def quick_audit(
    system_id: str,
    records: list[dict[str, Any]],
    db_path: str | None = None,
    bsv_wif: str | None = None,
) -> EpochSummary:
    """One-shot: audit a batch of inferences and return the epoch summary.

    Args:
        system_id: Your system name.
        records:   List of dicts with keys: ``model_id``, ``input_data``,
                   ``output_data``, and optionally ``confidence``, ``latency_ms``.
        db_path:   SQLite database path (default: ``aria_<system_id>.db``).
        bsv_wif:   Optional WIF key for BSV anchoring.

    Returns:
        EpochSummary — the result of the closed epoch.

    Example::

        from aria.quick import quick_audit

        summary = quick_audit("my-app", [
            {"model_id": "gpt-4", "input_data": {"q": "hi"}, "output_data": {"a": "hello"}},
        ])
        print(summary)
    """
    aria = ARIAQuick(
        system_id=system_id,
        db_path=db_path,
        bsv_wif=bsv_wif,
        watchdog=False,
        compliance=True,
    )
    aria.start()

    for r in records:
        aria.record(
            model_id=r["model_id"],
            input_data=r.get("input_data", {}),
            output_data=r.get("output_data", {}),
            confidence=r.get("confidence"),
            latency_ms=r.get("latency_ms"),
            metadata=r.get("metadata"),
        )

    summary = aria.close()
    aria.stop()
    return summary
