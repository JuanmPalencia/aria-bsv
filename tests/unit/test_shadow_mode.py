"""Tests for aria.shadow_mode — ShadowRunner."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aria.shadow_mode import ShadowResult, ShadowRunner


# ---------------------------------------------------------------------------
# Basic operation
# ---------------------------------------------------------------------------

class TestShadowRunnerBasic:
    def test_returns_live_result(self):
        live   = lambda x: f"live:{x}"
        shadow = lambda x: f"shadow:{x}"
        runner = ShadowRunner(live, shadow, async_shadow=False)
        assert runner.run("hello") == "live:hello"

    def test_shadow_also_called(self):
        live_calls   = []
        shadow_calls = []
        live   = lambda x: live_calls.append(x) or "live"
        shadow = lambda x: shadow_calls.append(x) or "shadow"
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("test")
        assert shadow_calls == ["test"]

    def test_shadow_error_does_not_affect_live(self):
        live   = lambda x: f"live:{x}"
        shadow = lambda x: (_ for _ in ()).throw(RuntimeError("boom"))
        runner = ShadowRunner(live, shadow, async_shadow=False)
        result = runner.run("hello")
        assert result == "live:hello"

    def test_shadow_results_collected(self):
        live   = lambda x: "live"
        shadow = lambda x: "shadow"
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("a")
        runner.run("b")
        assert len(runner.shadow_results) == 2

    def test_shadow_result_has_live_result(self):
        live   = lambda x: 42
        shadow = lambda x: 99
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("x")
        sr = runner.shadow_results[0]
        assert sr.live_result == 42
        assert sr.shadow_result == 99

    def test_shadow_error_stored(self):
        live   = lambda x: "ok"
        shadow = lambda x: 1 / 0
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("x")
        sr = runner.shadow_results[0]
        assert sr.shadow_error is not None
        assert "division by zero" in sr.shadow_error

    def test_live_latency_positive(self):
        live   = lambda x: time.sleep(0) or "ok"
        shadow = lambda x: "shadow"
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("x")
        sr = runner.shadow_results[0]
        assert sr.live_latency_ms >= 0

    def test_shadow_latency_positive(self):
        live   = lambda x: "ok"
        shadow = lambda x: time.sleep(0) or "shadow"
        runner = ShadowRunner(live, shadow, async_shadow=False)
        runner.run("x")
        sr = runner.shadow_results[0]
        assert sr.shadow_latency_ms >= 0


# ---------------------------------------------------------------------------
# error_rate
# ---------------------------------------------------------------------------

class TestErrorRate:
    def test_no_errors(self):
        runner = ShadowRunner(lambda x: "ok", lambda x: "shadow", async_shadow=False)
        runner.run("a")
        runner.run("b")
        assert runner.error_rate() == 0.0

    def test_all_errors(self):
        runner = ShadowRunner(lambda x: "ok", lambda x: 1/0, async_shadow=False)
        runner.run("a")
        runner.run("b")
        assert runner.error_rate() == 1.0

    def test_half_errors(self):
        call_count = [0]
        def shadow(x):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("fail")
            return "ok"
        runner = ShadowRunner(lambda x: "live", shadow, async_shadow=False)
        runner.run("a")
        runner.run("b")
        assert runner.error_rate() == pytest.approx(0.5)

    def test_empty(self):
        runner = ShadowRunner(lambda x: "ok", lambda x: "s", async_shadow=False)
        assert runner.error_rate() == 0.0


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestShadowRecording:
    def test_records_to_auditor(self):
        auditor = MagicMock()
        live   = lambda x: "ok"
        shadow = lambda x: "shadow-result"
        runner = ShadowRunner(
            live, shadow,
            auditor=auditor,
            shadow_model_id="v2",
            async_shadow=False,
        )
        runner.run("input")
        auditor.record.assert_called_once()
        args = auditor.record.call_args[0]
        assert args[0] == "v2"

    def test_records_to_aria(self):
        aria   = MagicMock()
        live   = lambda x: "ok"
        shadow = lambda x: "shadow"
        runner = ShadowRunner(live, shadow, aria=aria, shadow_model_id="v2", async_shadow=False)
        runner.run("input")
        aria.record.assert_called_once()

    def test_no_record_on_shadow_error(self):
        auditor = MagicMock()
        live   = lambda x: "ok"
        shadow = lambda x: 1/0
        runner = ShadowRunner(live, shadow, auditor=auditor, async_shadow=False)
        runner.run("input")
        auditor.record.assert_not_called()

    def test_metadata_has_shadow_mode(self):
        auditor = MagicMock()
        runner = ShadowRunner(
            lambda x: "live",
            lambda x: "shadow",
            auditor=auditor,
            async_shadow=False,
        )
        runner.run("x")
        kwargs = auditor.record.call_args[1]
        assert kwargs["metadata"]["mode"] == "shadow"


# ---------------------------------------------------------------------------
# Async shadow
# ---------------------------------------------------------------------------

class TestAsyncShadow:
    def test_async_returns_live_immediately(self):
        import threading
        event = threading.Event()

        def slow_shadow(x):
            event.wait(timeout=2.0)
            return "done"

        runner = ShadowRunner(lambda x: "live", slow_shadow, async_shadow=True)
        result = runner.run("test")
        assert result == "live"
        event.set()  # allow shadow to finish
