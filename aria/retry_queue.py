"""
aria.retry_queue — Persistent retry queue for failed BSV broadcasts.

When ARC is down or unreachable, records are buffered in a local SQLite
queue. A background worker (`RetryWorker`) periodically retries pending
items with exponential backoff.

Usage::

    from aria.retry_queue import RetryQueue, RetryWorker

    queue = RetryQueue()                             # ~/.aria/retry_queue.db
    queue.enqueue(raw_tx="0100...", epoch_id="ep-1")

    worker = RetryWorker(queue, broadcaster)
    worker.start()                                   # background thread
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

QUEUE_DB_PATH = Path.home() / ".aria" / "retry_queue.db"


@dataclass
class QueueItem:
    """A single queued broadcast attempt."""

    item_id: int
    raw_tx: str
    epoch_id: str
    payload_type: str
    attempts: int
    last_error: str
    created_at: float
    next_retry_at: float
    metadata: dict[str, Any]

    @property
    def is_ready(self) -> bool:
        return time.time() >= self.next_retry_at


class RetryQueue:
    """Persistent SQLite-backed queue for failed broadcasts.

    Thread-safe. Uses WAL mode for concurrent reads.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or QUEUE_DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retry_queue (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_tx TEXT NOT NULL,
                    epoch_id TEXT NOT NULL DEFAULT '',
                    payload_type TEXT NOT NULL DEFAULT 'broadcast',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    next_retry_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def enqueue(
        self,
        raw_tx: str,
        epoch_id: str = "",
        payload_type: str = "broadcast",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a failed broadcast to the retry queue.

        Returns:
            The item_id of the queued item.
        """
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO retry_queue
                   (raw_tx, epoch_id, payload_type, attempts, created_at, next_retry_at, metadata)
                   VALUES (?, ?, ?, 0, ?, ?, ?)""",
                (raw_tx, epoch_id, payload_type, now, now, json.dumps(metadata or {})),
            )
            item_id = cursor.lastrowid
            _log.info("Queued broadcast for retry: item_id=%d epoch=%s", item_id, epoch_id)
            return item_id

    def peek_ready(self, limit: int = 10) -> list[QueueItem]:
        """Return up to *limit* items ready for retry (next_retry_at <= now)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM retry_queue
                   WHERE next_retry_at <= ?
                   ORDER BY next_retry_at ASC
                   LIMIT ?""",
                (time.time(), limit),
            ).fetchall()
            return [self._row_to_item(r) for r in rows]

    def mark_success(self, item_id: int) -> None:
        """Remove a successfully broadcast item from the queue."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM retry_queue WHERE item_id = ?", (item_id,))
            _log.info("Retry success, removed item_id=%d", item_id)

    def mark_failed(self, item_id: int, error: str, backoff_base: float = 30.0) -> None:
        """Increment attempt count and schedule next retry with exponential backoff."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM retry_queue WHERE item_id = ?", (item_id,)
            ).fetchone()
            if row is None:
                return
            attempts = row["attempts"] + 1
            delay = min(backoff_base * (2 ** attempts), 3600)  # cap at 1 hour
            next_at = time.time() + delay
            conn.execute(
                """UPDATE retry_queue
                   SET attempts = ?, last_error = ?, next_retry_at = ?
                   WHERE item_id = ?""",
                (attempts, error, next_at, item_id),
            )
            _log.warning(
                "Retry failed item_id=%d attempt=%d next_retry=%.0fs error=%s",
                item_id, attempts, delay, error,
            )

    def dead_letters(self, max_attempts: int = 10) -> list[QueueItem]:
        """Return items that have exceeded max retry attempts."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retry_queue WHERE attempts >= ? ORDER BY created_at",
                (max_attempts,),
            ).fetchall()
            return [self._row_to_item(r) for r in rows]

    def purge_dead_letters(self, max_attempts: int = 10) -> int:
        """Remove items that have exceeded max retry attempts."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM retry_queue WHERE attempts >= ?", (max_attempts,)
            )
            count = cursor.rowcount
            if count:
                _log.info("Purged %d dead letter items", count)
            return count

    def count(self) -> dict[str, int]:
        """Return counts by status."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM retry_queue").fetchone()[0]
            ready = conn.execute(
                "SELECT COUNT(*) FROM retry_queue WHERE next_retry_at <= ?",
                (time.time(),),
            ).fetchone()[0]
            return {"total": total, "ready": ready, "waiting": total - ready}

    def all_items(self) -> list[QueueItem]:
        """Return all items in the queue."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM retry_queue ORDER BY created_at").fetchall()
            return [self._row_to_item(r) for r in rows]

    def clear(self) -> int:
        """Remove all items from the queue."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM retry_queue")
            return cursor.rowcount

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            item_id=row["item_id"],
            raw_tx=row["raw_tx"],
            epoch_id=row["epoch_id"],
            payload_type=row["payload_type"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            next_retry_at=row["next_retry_at"],
            metadata=json.loads(row["metadata"]),
        )


class RetryWorker:
    """Background thread that retries queued broadcasts.

    Args:
        queue:       RetryQueue instance.
        broadcaster: BroadcasterInterface for retrying broadcasts.
        interval:    Seconds between retry sweeps (default: 30).
        max_attempts: Items exceeding this are moved to dead letters (default: 10).
    """

    def __init__(
        self,
        queue: RetryQueue,
        broadcaster: Any,
        interval: float = 30.0,
        max_attempts: int = 10,
    ) -> None:
        self._queue = queue
        self._broadcaster = broadcaster
        self._interval = interval
        self._max_attempts = max_attempts
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background retry worker."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="aria-retry-worker"
        )
        self._thread.start()
        _log.info("Retry worker started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        """Stop the background retry worker."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    def process_once(self) -> int:
        """Process ready items once (synchronous). Returns count processed."""
        import asyncio

        items = self._queue.peek_ready(limit=20)
        processed = 0

        for item in items:
            if item.attempts >= self._max_attempts:
                continue
            try:
                result = asyncio.run(self._broadcaster.broadcast(item.raw_tx))
                if result.propagated:
                    self._queue.mark_success(item.item_id)
                    processed += 1
                else:
                    self._queue.mark_failed(item.item_id, result.message)
            except Exception as exc:
                self._queue.mark_failed(item.item_id, str(exc))

        return processed

    def _loop(self) -> None:
        while self._running:
            try:
                self.process_once()
            except Exception as exc:
                _log.error("Retry worker error: %s", exc)
            time.sleep(self._interval)
