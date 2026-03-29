"""Tests for aria.retry_queue — persistent retry queue for failed broadcasts."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.retry_queue import QueueItem, RetryQueue, RetryWorker


@pytest.fixture
def queue(tmp_path):
    """Provide a RetryQueue with a temporary database."""
    db_path = tmp_path / "retry_test.db"
    return RetryQueue(db_path=db_path)


class TestRetryQueue:
    """Tests for the RetryQueue class."""

    def test_enqueue(self, queue):
        item_id = queue.enqueue(raw_tx="0100abcd", epoch_id="ep-1")
        assert isinstance(item_id, int)
        assert item_id >= 1

    def test_enqueue_multiple(self, queue):
        id1 = queue.enqueue(raw_tx="tx1", epoch_id="ep-1")
        id2 = queue.enqueue(raw_tx="tx2", epoch_id="ep-2")
        assert id2 > id1

    def test_peek_ready(self, queue):
        queue.enqueue(raw_tx="tx-ready", epoch_id="ep-1")
        ready = queue.peek_ready(limit=5)
        assert len(ready) == 1
        assert ready[0].raw_tx == "tx-ready"
        assert ready[0].epoch_id == "ep-1"
        assert ready[0].attempts == 0

    def test_peek_ready_respects_next_retry_at(self, queue):
        queue.enqueue(raw_tx="tx1")
        # Mark failed → pushes next_retry_at into the future
        items = queue.peek_ready()
        queue.mark_failed(items[0].item_id, "fail", backoff_base=9999)
        ready = queue.peek_ready()
        assert len(ready) == 0  # not ready yet

    def test_mark_success(self, queue):
        item_id = queue.enqueue(raw_tx="tx-success")
        queue.mark_success(item_id)
        assert queue.peek_ready() == []
        counts = queue.count()
        assert counts["total"] == 0

    def test_mark_failed_increments_attempts(self, queue):
        item_id = queue.enqueue(raw_tx="tx-fail")
        queue.mark_failed(item_id, "network error", backoff_base=0.001)
        items = queue.all_items()
        assert len(items) == 1
        assert items[0].attempts == 1
        assert items[0].last_error == "network error"

    def test_dead_letters(self, queue):
        item_id = queue.enqueue(raw_tx="dead")
        for i in range(10):
            queue.mark_failed(item_id, f"fail-{i}", backoff_base=0.0)
        dead = queue.dead_letters(max_attempts=10)
        assert len(dead) == 1
        assert dead[0].attempts == 10

    def test_purge_dead_letters(self, queue):
        item_id = queue.enqueue(raw_tx="dead2")
        for i in range(12):
            queue.mark_failed(item_id, f"fail-{i}", backoff_base=0.0)
        purged = queue.purge_dead_letters(max_attempts=10)
        assert purged == 1
        assert queue.count()["total"] == 0

    def test_count(self, queue):
        queue.enqueue(raw_tx="tx1")
        queue.enqueue(raw_tx="tx2")
        counts = queue.count()
        assert counts["total"] == 2
        assert counts["ready"] == 2
        assert counts["waiting"] == 0

    def test_all_items(self, queue):
        queue.enqueue(raw_tx="a")
        queue.enqueue(raw_tx="b")
        items = queue.all_items()
        assert len(items) == 2
        assert items[0].raw_tx == "a"
        assert items[1].raw_tx == "b"

    def test_clear(self, queue):
        queue.enqueue(raw_tx="x")
        queue.enqueue(raw_tx="y")
        cleared = queue.clear()
        assert cleared == 2
        assert queue.count()["total"] == 0

    def test_metadata(self, queue):
        item_id = queue.enqueue(raw_tx="tx-meta", metadata={"key": "value"})
        items = queue.all_items()
        assert items[0].metadata == {"key": "value"}

    def test_queue_item_is_ready(self):
        item = QueueItem(
            item_id=1,
            raw_tx="tx",
            epoch_id="ep",
            payload_type="broadcast",
            attempts=0,
            last_error="",
            created_at=time.time(),
            next_retry_at=time.time() - 1,
            metadata={},
        )
        assert item.is_ready is True

    def test_queue_item_not_ready(self):
        item = QueueItem(
            item_id=1,
            raw_tx="tx",
            epoch_id="ep",
            payload_type="broadcast",
            attempts=0,
            last_error="",
            created_at=time.time(),
            next_retry_at=time.time() + 3600,
            metadata={},
        )
        assert item.is_ready is False


class TestRetryWorker:
    """Tests for the RetryWorker class."""

    def test_process_once_success(self, queue):
        queue.enqueue(raw_tx="tx-ok")

        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast = AsyncMock(
            return_value=MagicMock(propagated=True, txid="abc123", message="ok")
        )

        worker = RetryWorker(queue, mock_broadcaster)
        processed = worker.process_once()
        assert processed == 1
        assert queue.count()["total"] == 0

    def test_process_once_failure(self, queue):
        queue.enqueue(raw_tx="tx-nope")

        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast = AsyncMock(
            return_value=MagicMock(propagated=False, txid="", message="rejected")
        )

        worker = RetryWorker(queue, mock_broadcaster)
        processed = worker.process_once()
        assert processed == 0
        items = queue.all_items()
        assert items[0].attempts == 1
        assert items[0].last_error == "rejected"

    def test_process_once_exception(self, queue):
        queue.enqueue(raw_tx="tx-exc")

        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast = AsyncMock(side_effect=ConnectionError("offline"))

        worker = RetryWorker(queue, mock_broadcaster)
        processed = worker.process_once()
        assert processed == 0
        items = queue.all_items()
        assert items[0].attempts == 1
        assert "offline" in items[0].last_error

    def test_start_stop(self, queue):
        mock_broadcaster = MagicMock()
        worker = RetryWorker(queue, mock_broadcaster, interval=0.1)
        worker.start()
        assert worker._running is True
        worker.stop()
        assert worker._running is False

    def test_skips_dead_letter_items(self, queue):
        item_id = queue.enqueue(raw_tx="dead-item")
        for _ in range(10):
            queue.mark_failed(item_id, "fail", backoff_base=0.0)

        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast = AsyncMock()

        worker = RetryWorker(queue, mock_broadcaster, max_attempts=10)
        processed = worker.process_once()
        assert processed == 0
        mock_broadcaster.broadcast.assert_not_called()
