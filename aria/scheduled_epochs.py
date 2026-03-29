"""
aria.scheduled_epochs — Automatic epoch lifecycle management.

Handles opening and closing epochs automatically based on configurable
strategies, eliminating the need for manual epoch management.

Strategies:
- ``every_n_records``  — close after N records, open a new one
- ``every_n_minutes``  — close after N minutes, open a new one
- ``every_n_bytes``    — close after N bytes of payload data
- ``hybrid``           — close on whichever threshold is hit first

Usage::

    from aria.quick import ARIAQuick

    # Close epoch every 100 records
    aria = ARIAQuick("my-system", epoch_strategy="every_100_records")

    # Close epoch every 5 minutes
    aria = ARIAQuick("my-system", epoch_strategy="every_5_minutes")

    # Or use the scheduler directly
    from aria.scheduled_epochs import EpochScheduler, ScheduleConfig

    scheduler = EpochScheduler(auditor, ScheduleConfig(max_records=100, max_minutes=5))
    scheduler.start()
    # ... record inferences ...
    scheduler.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for automatic epoch scheduling.

    Set any combination of thresholds — the epoch closes when any is hit.

    Attributes:
        max_records:  Close after this many records (0 = disabled).
        max_minutes:  Close after this many minutes (0 = disabled).
        max_bytes:    Close after this many payload bytes (0 = disabled).
        on_close:     Optional callback invoked after each epoch close.
        on_error:     Optional callback invoked when epoch close fails.
        auto_reopen:  Automatically open a new epoch after closing (default True).
    """
    max_records: int = 0
    max_minutes: float = 0
    max_bytes: int = 0
    on_close: Callable[[dict[str, Any]], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    auto_reopen: bool = True


def parse_strategy(strategy: str) -> ScheduleConfig:
    """Parse a human-friendly strategy string into a ScheduleConfig.

    Supported formats::

        "every_100_records"
        "every_5_minutes"
        "every_1mb"
        "records:100,minutes:5"   # hybrid

    Args:
        strategy: Strategy string.

    Returns:
        ScheduleConfig with the parsed thresholds.
    """
    s = strategy.strip().lower()

    # Simple formats: every_N_records, every_N_minutes
    if s.startswith("every_") and s.endswith("_records"):
        n = int(s.replace("every_", "").replace("_records", ""))
        return ScheduleConfig(max_records=n)

    if s.startswith("every_") and s.endswith("_minutes"):
        n = float(s.replace("every_", "").replace("_minutes", ""))
        return ScheduleConfig(max_minutes=n)

    if s.startswith("every_") and s.endswith("mb"):
        n = int(s.replace("every_", "").replace("mb", ""))
        return ScheduleConfig(max_bytes=n * 1_048_576)

    # Hybrid: "records:100,minutes:5"
    if ":" in s:
        cfg = ScheduleConfig()
        for part in s.split(","):
            key, val = part.strip().split(":")
            if key == "records":
                cfg.max_records = int(val)
            elif key == "minutes":
                cfg.max_minutes = float(val)
            elif key == "bytes":
                cfg.max_bytes = int(val)
        return cfg

    raise ValueError(
        f"Unknown strategy {strategy!r}. "
        "Use 'every_N_records', 'every_N_minutes', 'every_Nmb', "
        "or 'records:N,minutes:N'."
    )


class EpochScheduler:
    """Background scheduler for automatic epoch lifecycle.

    Monitors record count and elapsed time, closing epochs when any
    configured threshold is hit.  Thread-safe.

    Args:
        open_fn:  Callable that opens a new epoch.  Returns epoch_id.
        close_fn: Callable that closes the current epoch.
        config:   Scheduling thresholds.
    """

    def __init__(
        self,
        open_fn: Callable[[], str],
        close_fn: Callable[[], Any],
        config: ScheduleConfig,
    ) -> None:
        self._open_fn = open_fn
        self._close_fn = close_fn
        self._config = config

        self._lock = threading.Lock()
        self._record_count = 0
        self._bytes_count = 0
        self._epoch_opened_at: float = 0
        self._running = False
        self._timer_thread: threading.Thread | None = None
        self._current_epoch_id: str | None = None

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def current_epoch_id(self) -> str | None:
        return self._current_epoch_id

    def start(self) -> str:
        """Open the first epoch and start the scheduler.

        Returns:
            The epoch_id of the opened epoch.
        """
        with self._lock:
            if self._running:
                return self._current_epoch_id or ""
            self._running = True
            epoch_id = self._open_epoch()

            if self._config.max_minutes > 0:
                self._start_timer()

            return epoch_id

    def stop(self) -> None:
        """Close the current epoch and stop the scheduler."""
        with self._lock:
            self._running = False
        self._stop_timer()
        self._try_close()

    def notify_record(self, payload_bytes: int = 0) -> None:
        """Called after each record is added.  Checks thresholds.

        Args:
            payload_bytes: Size of the serialised record payload (optional).
        """
        should_close = False
        with self._lock:
            if not self._running:
                return
            self._record_count += 1
            self._bytes_count += payload_bytes

            if self._config.max_records > 0 and self._record_count >= self._config.max_records:
                should_close = True
            if self._config.max_bytes > 0 and self._bytes_count >= self._config.max_bytes:
                should_close = True

        if should_close:
            self._rotate()

    def _rotate(self) -> None:
        """Close the current epoch and optionally open a new one."""
        self._try_close()
        if self._config.auto_reopen and self._running:
            with self._lock:
                self._open_epoch()

    def _try_close(self) -> None:
        """Attempt to close the current epoch, invoking callbacks."""
        try:
            result = self._close_fn()
            _log.info("Epoch closed: records=%d", self._record_count)
            if self._config.on_close:
                self._config.on_close({"records": self._record_count, "result": result})
        except Exception as exc:
            _log.error("Failed to close epoch: %s", exc)
            if self._config.on_error:
                self._config.on_error(exc)

    def _open_epoch(self) -> str:
        """Open a new epoch and reset counters."""
        self._record_count = 0
        self._bytes_count = 0
        self._epoch_opened_at = time.monotonic()
        self._current_epoch_id = self._open_fn()
        _log.info("Epoch opened: %s", self._current_epoch_id)
        return self._current_epoch_id

    # ------------------------------------------------------------------
    # Timer thread for time-based closing
    # ------------------------------------------------------------------

    def _start_timer(self) -> None:
        self._timer_thread = threading.Thread(
            target=self._timer_loop, daemon=True, name="aria-epoch-scheduler"
        )
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        self._running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=2)
        self._timer_thread = None

    def _timer_loop(self) -> None:
        interval = max(1.0, self._config.max_minutes * 60 / 10)  # check 10× per period
        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            elapsed = time.monotonic() - self._epoch_opened_at
            if elapsed >= self._config.max_minutes * 60:
                _log.info("Time threshold reached (%.1f min), rotating epoch", elapsed / 60)
                self._rotate()
