"""Tests for aria.scheduled_epochs — Auto epoch lifecycle."""

from __future__ import annotations

import time
import pytest

from aria.scheduled_epochs import (
    EpochScheduler,
    ScheduleConfig,
    parse_strategy,
)


class TestParseStrategy:
    def test_every_n_records(self):
        cfg = parse_strategy("every_100_records")
        assert cfg.max_records == 100

    def test_every_n_minutes(self):
        cfg = parse_strategy("every_5_minutes")
        assert cfg.max_minutes == 5

    def test_every_n_mb(self):
        cfg = parse_strategy("every_10mb")
        assert cfg.max_bytes == 10 * 1024 * 1024

    def test_hybrid(self):
        cfg = parse_strategy("records:50,minutes:10")
        assert cfg.max_records == 50
        assert cfg.max_minutes == 10

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_strategy("invalid_strategy")


class TestScheduleConfig:
    def test_defaults(self):
        cfg = ScheduleConfig()
        assert cfg.max_records == 0
        assert cfg.max_minutes == 0
        assert cfg.max_bytes == 0
        assert cfg.auto_reopen is True


class TestEpochScheduler:
    def test_notify_record_counts(self):
        close_called = []
        open_called = []

        def on_close():
            close_called.append(True)

        def on_open():
            open_called.append(True)

        cfg = ScheduleConfig(max_records=3, auto_reopen=False)
        scheduler = EpochScheduler(
            open_fn=on_open,
            close_fn=on_close,
            config=cfg,
        )
        scheduler.start()

        scheduler.notify_record(100)
        scheduler.notify_record(100)
        assert len(close_called) == 0

        scheduler.notify_record(100)
        assert len(close_called) == 1

        scheduler.stop()

    def test_notify_record_with_auto_reopen(self):
        close_count = [0]
        open_count = [0]

        def on_close():
            close_count[0] += 1

        def on_open():
            open_count[0] += 1

        cfg = ScheduleConfig(max_records=2, auto_reopen=True)
        scheduler = EpochScheduler(
            open_fn=on_open,
            close_fn=on_close,
            config=cfg,
        )
        scheduler.start()  # calls open_fn once

        scheduler.notify_record(10)
        scheduler.notify_record(10)
        assert close_count[0] == 1
        assert open_count[0] == 2  # start() + auto_reopen

        scheduler.notify_record(10)
        scheduler.notify_record(10)
        assert close_count[0] == 2
        assert open_count[0] == 3  # start() + 2 auto_reopens

        scheduler.stop()

    def test_bytes_threshold(self):
        close_called = []

        cfg = ScheduleConfig(max_records=999, max_bytes=100, auto_reopen=False)
        scheduler = EpochScheduler(
            open_fn=lambda: None,
            close_fn=lambda: close_called.append(True),
            config=cfg,
        )
        scheduler.start()

        scheduler.notify_record(50)
        assert len(close_called) == 0

        scheduler.notify_record(60)
        assert len(close_called) == 1

        scheduler.stop()

    def test_stop_is_safe(self):
        cfg = ScheduleConfig(max_records=100)
        scheduler = EpochScheduler(
            open_fn=lambda: None,
            close_fn=lambda: None,
            config=cfg,
        )
        # Stop without start should not raise
        scheduler.stop()

    def test_error_callback(self):
        errors = []

        def bad_close():
            raise RuntimeError("boom")

        cfg = ScheduleConfig(max_records=1, auto_reopen=False)
        scheduler = EpochScheduler(
            open_fn=lambda: None,
            close_fn=bad_close,
            config=cfg,
        )
        scheduler._config.on_error = lambda e: errors.append(str(e))
        scheduler.start()

        scheduler.notify_record(10)
        assert len(errors) == 1
        assert "boom" in errors[0]

        scheduler.stop()
