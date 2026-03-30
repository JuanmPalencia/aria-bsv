"""
Tests for pytest_aria.plugin.

Strategy: import ARIAPlugin directly, build lightweight mock config/report
objects, and call plugin hooks directly — no pytester required and no real
BSV connection needed.

ARIAQuick is patched at the import site inside plugin.py using
``unittest.mock.patch`` so the tests remain fully self-contained.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARIA_QUICK_PATH = "aria.quick.ARIAQuick"
_PLUGIN_IMPORT_PATH = "pytest_aria.plugin"


def _make_mock_aria() -> MagicMock:
    """Return a mock ARIAQuick instance with sensible defaults."""
    aria = MagicMock()
    aria.record = MagicMock(return_value="rec_0001")
    aria.close = MagicMock(
        return_value=MagicMock(
            epoch_id="ep_test_0001",
            merkle_root="sha256:" + "a" * 64,
            records_count=1,
        )
    )
    aria.start = MagicMock(return_value=aria)
    return aria


def _make_config(
    enabled: bool = True,
    record_passed: bool = True,
    record_failed: bool = True,
    record_skipped: bool = False,
    system_id: str = "pytest",
    db_path: str = "aria_tests.db",
) -> MagicMock:
    """Return a mock pytest.Config object pre-configured with ARIA ini values."""
    ini: dict[str, Any] = {
        "aria_enabled": enabled,
        "aria_record_passed": record_passed,
        "aria_record_failed": record_failed,
        "aria_record_skipped": record_skipped,
        "aria_system_id": system_id,
        "aria_db_path": db_path,
    }
    config = MagicMock()
    config.getini = MagicMock(side_effect=lambda key: ini.get(key))
    # CLI flag attribute: None means "not supplied via CLI"
    config.option.aria_enabled_flag = None
    config.pluginmanager.get_plugin = MagicMock(return_value=None)
    return config


def _make_report(
    outcome: str = "passed",
    nodeid: str = "tests/test_foo.py::test_bar",
    when: str = "call",
    duration: float = 0.01,
) -> MagicMock:
    """Return a mock pytest.TestReport."""
    report = MagicMock()
    report.outcome = outcome
    report.nodeid = nodeid
    report.when = when
    report.duration = duration
    return report


def _make_session(config: MagicMock | None = None) -> MagicMock:
    session = MagicMock()
    session.config = config or _make_config()
    return session


# ---------------------------------------------------------------------------
# Import the module under test (needs to happen after helpers are defined)
# ---------------------------------------------------------------------------

from pytest_aria.plugin import ARIAPlugin  # noqa: E402


# ===========================================================================
# 1 — Initialisation
# ===========================================================================


class TestInit:
    def test_plugin_initialises_aria_quick_when_enabled(self):
        """ARIAPlugin should create and start an ARIAQuick instance when enabled."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria) as MockCls:
            plugin = ARIAPlugin(_make_config(enabled=True))
        MockCls.assert_called_once()
        mock_aria.start.assert_called_once()
        assert plugin._enabled is True
        assert plugin._aria is mock_aria

    def test_plugin_does_not_init_aria_quick_when_disabled(self):
        """ARIAPlugin must NOT call ARIAQuick at all when aria_enabled=False."""
        with patch("pytest_aria.plugin.ARIAQuick") as MockCls:
            plugin = ARIAPlugin(_make_config(enabled=False))
        MockCls.assert_not_called()
        assert plugin._enabled is False
        assert plugin._aria is None

    def test_cli_flag_true_overrides_ini_false(self):
        """--aria-enabled CLI flag should override ini aria_enabled=False."""
        config = _make_config(enabled=False)
        config.option.aria_enabled_flag = True  # CLI override
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(config)
        assert plugin._enabled is True

    def test_cli_flag_false_overrides_ini_true(self):
        """--no-aria-enabled CLI flag should override ini aria_enabled=True."""
        config = _make_config(enabled=True)
        config.option.aria_enabled_flag = False  # CLI override
        with patch("pytest_aria.plugin.ARIAQuick") as MockCls:
            plugin = ARIAPlugin(config)
        MockCls.assert_not_called()
        assert plugin._enabled is False

    def test_system_id_passed_to_aria_quick(self):
        """system_id from ini should be forwarded to ARIAQuick constructor."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria) as MockCls:
            ARIAPlugin(_make_config(system_id="my-ci-system"))
        _, kwargs = MockCls.call_args
        assert kwargs.get("system_id") == "my-ci-system"

    def test_db_path_passed_to_aria_quick(self):
        """db_path from ini should be forwarded to ARIAQuick constructor."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria) as MockCls:
            ARIAPlugin(_make_config(db_path="/tmp/custom.db"))
        _, kwargs = MockCls.call_args
        assert kwargs.get("db_path") == "/tmp/custom.db"

    def test_import_error_disables_plugin_gracefully(self):
        """If aria-bsv is not installed, the plugin must disable without crashing."""
        with patch(
            "pytest_aria.plugin.ARIAQuick",
            side_effect=ImportError("No module named 'aria'"),
        ):
            plugin = ARIAPlugin(_make_config(enabled=True))
        assert plugin._enabled is False
        assert plugin._aria is None

    def test_record_flags_read_from_config(self):
        """Per-outcome recording flags should be read from config on init."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(
                _make_config(
                    record_passed=True,
                    record_failed=False,
                    record_skipped=True,
                )
            )
        assert plugin._record_passed is True
        assert plugin._record_failed is False
        assert plugin._record_skipped is True


# ===========================================================================
# 2 — pytest_runtest_logreport: phase filtering
# ===========================================================================


class TestPhaseFiltering:
    def _plugin(self, **ini_kwargs) -> tuple[ARIAPlugin, MagicMock]:
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config(**ini_kwargs))
        return plugin, mock_aria

    def test_setup_phase_is_ignored(self):
        plugin, mock_aria = self._plugin()
        plugin.pytest_runtest_logreport(_make_report(when="setup"))
        mock_aria.record.assert_not_called()

    def test_teardown_phase_is_ignored(self):
        plugin, mock_aria = self._plugin()
        plugin.pytest_runtest_logreport(_make_report(when="teardown"))
        mock_aria.record.assert_not_called()

    def test_call_phase_is_processed(self):
        plugin, mock_aria = self._plugin()
        plugin.pytest_runtest_logreport(_make_report(when="call", outcome="passed"))
        mock_aria.record.assert_called_once()


# ===========================================================================
# 3 — pytest_runtest_logreport: outcome filtering
# ===========================================================================


class TestOutcomeFiltering:
    def _plugin(self, **ini_kwargs) -> tuple[ARIAPlugin, MagicMock]:
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config(**ini_kwargs))
        return plugin, mock_aria

    def test_records_passed_when_flag_true(self):
        plugin, mock_aria = self._plugin(record_passed=True)
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))
        mock_aria.record.assert_called_once()

    def test_skips_passed_when_flag_false(self):
        plugin, mock_aria = self._plugin(record_passed=False)
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))
        mock_aria.record.assert_not_called()

    def test_records_failed_when_flag_true(self):
        plugin, mock_aria = self._plugin(record_failed=True)
        plugin.pytest_runtest_logreport(_make_report(outcome="failed"))
        mock_aria.record.assert_called_once()

    def test_skips_failed_when_flag_false(self):
        plugin, mock_aria = self._plugin(record_failed=False)
        plugin.pytest_runtest_logreport(_make_report(outcome="failed"))
        mock_aria.record.assert_not_called()

    def test_records_skipped_when_flag_true(self):
        plugin, mock_aria = self._plugin(record_skipped=True)
        plugin.pytest_runtest_logreport(_make_report(outcome="skipped"))
        mock_aria.record.assert_called_once()

    def test_skips_skipped_when_flag_false(self):
        plugin, mock_aria = self._plugin(record_skipped=False)
        plugin.pytest_runtest_logreport(_make_report(outcome="skipped"))
        mock_aria.record.assert_not_called()

    def test_no_record_when_plugin_disabled(self):
        """Even if a test runs, nothing should be recorded when plugin is off."""
        with patch("pytest_aria.plugin.ARIAQuick") as MockCls:
            plugin = ARIAPlugin(_make_config(enabled=False))
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))
        MockCls.return_value.record.assert_not_called()


# ===========================================================================
# 4 — pytest_runtest_logreport: payload correctness
# ===========================================================================


class TestRecordPayload:
    def _plugin_and_aria(self) -> tuple[ARIAPlugin, MagicMock]:
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config())
        return plugin, mock_aria

    def _get_call_kwargs(self, mock_aria: MagicMock) -> dict[str, Any]:
        assert mock_aria.record.called
        _, kwargs = mock_aria.record.call_args
        return kwargs

    def test_outcome_in_output_data(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(_make_report(outcome="failed"))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["output_data"]["outcome"] == "failed"

    def test_nodeid_in_input_data(self):
        plugin, mock_aria = self._plugin_and_aria()
        nodeid = "tests/unit/test_foo.py::test_something"
        plugin.pytest_runtest_logreport(_make_report(nodeid=nodeid))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["input_data"]["test"] == nodeid

    def test_model_id_is_nodeid(self):
        plugin, mock_aria = self._plugin_and_aria()
        nodeid = "tests/unit/test_foo.py::test_something"
        plugin.pytest_runtest_logreport(_make_report(nodeid=nodeid))
        _, kwargs = mock_aria.record.call_args
        assert kwargs["model_id"] == nodeid

    def test_confidence_1_for_passed(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["confidence"] == 1.0

    def test_confidence_0_for_failed(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(_make_report(outcome="failed"))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["confidence"] == 0.0

    def test_confidence_0_for_skipped(self):
        plugin, mock_aria = self._plugin_and_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin2 = ARIAPlugin(_make_config(record_skipped=True))
        plugin2.pytest_runtest_logreport(_make_report(outcome="skipped"))
        _, kwargs = mock_aria.record.call_args
        assert kwargs["confidence"] == 0.0

    def test_duration_ms_in_output_data(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(_make_report(duration=0.25))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["output_data"]["duration_ms"] == pytest.approx(250.0, abs=1.0)

    def test_module_parsed_from_nodeid(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(
            _make_report(nodeid="tests/unit/test_foo.py::test_something")
        )
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["input_data"]["module"] == "tests/unit/test_foo.py"

    def test_test_name_parsed_from_nodeid(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(
            _make_report(nodeid="tests/unit/test_foo.py::test_something")
        )
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["input_data"]["test_name"] == "test_something"

    def test_nodeid_without_separator_handled(self):
        """A bare nodeid with no '::' should not crash the parser."""
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_runtest_logreport(_make_report(nodeid="test_bare"))
        kwargs = self._get_call_kwargs(mock_aria)
        assert kwargs["input_data"]["module"] == "test_bare"
        assert kwargs["input_data"]["test_name"] == "test_bare"


# ===========================================================================
# 5 — pytest_sessionfinish
# ===========================================================================


class TestSessionFinish:
    def _plugin_and_aria(self, **ini_kwargs) -> tuple[ARIAPlugin, MagicMock]:
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config(**ini_kwargs))
        return plugin, mock_aria

    def test_sessionfinish_calls_close(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_sessionfinish(_make_session(), exitstatus=0)
        mock_aria.close.assert_called_once()

    def test_sessionfinish_disabled_does_not_crash(self):
        """pytest_sessionfinish with a disabled plugin must not raise."""
        with patch("pytest_aria.plugin.ARIAQuick"):
            plugin = ARIAPlugin(_make_config(enabled=False))
        # Should complete without any exception.
        plugin.pytest_sessionfinish(_make_session(), exitstatus=0)

    def test_sessionfinish_close_exception_does_not_crash(self):
        """If aria.close() raises, pytest_sessionfinish must still succeed."""
        plugin, mock_aria = self._plugin_and_aria()
        mock_aria.close.side_effect = RuntimeError("storage unavailable")
        # Must not propagate the exception.
        plugin.pytest_sessionfinish(_make_session(), exitstatus=1)

    def test_sessionfinish_with_zero_exit_status(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_sessionfinish(_make_session(), exitstatus=0)
        mock_aria.close.assert_called_once()

    def test_sessionfinish_with_nonzero_exit_status(self):
        plugin, mock_aria = self._plugin_and_aria()
        plugin.pytest_sessionfinish(_make_session(), exitstatus=1)
        mock_aria.close.assert_called_once()


# ===========================================================================
# 6 — Error resilience in record()
# ===========================================================================


class TestErrorResilience:
    def test_record_exception_does_not_crash_report_hook(self):
        """If aria.record() raises, pytest_runtest_logreport must not propagate."""
        mock_aria = _make_mock_aria()
        mock_aria.record.side_effect = RuntimeError("db locked")
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config())
        # Should complete without exception even though record() raised.
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))

    def test_records_written_counter_not_incremented_on_record_error(self):
        """If aria.record() raises, _records_written should NOT be incremented."""
        mock_aria = _make_mock_aria()
        mock_aria.record.side_effect = RuntimeError("db locked")
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config())
        plugin.pytest_runtest_logreport(_make_report(outcome="passed"))
        assert plugin._records_written == 0


# ===========================================================================
# 7 — Multiple reports
# ===========================================================================


class TestMultipleReports:
    def test_multiple_reports_call_record_n_times(self):
        """record() must be called once per qualifying test report."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config(record_passed=True, record_failed=True))

        reports = [
            _make_report(outcome="passed", nodeid=f"tests/test_foo.py::test_{i}")
            for i in range(5)
        ]
        for r in reports:
            plugin.pytest_runtest_logreport(r)

        assert mock_aria.record.call_count == 5

    def test_mixed_outcomes_counted_correctly(self):
        """Each pass/fail gets recorded; skipped is filtered (default flags)."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(
                _make_config(
                    record_passed=True,
                    record_failed=True,
                    record_skipped=False,
                )
            )
        reports = [
            _make_report(outcome="passed"),
            _make_report(outcome="failed"),
            _make_report(outcome="skipped"),
            _make_report(outcome="passed"),
        ]
        for r in reports:
            plugin.pytest_runtest_logreport(r)

        # 2 passed + 1 failed = 3; 1 skipped filtered out
        assert mock_aria.record.call_count == 3

    def test_records_written_counter_increments(self):
        """_records_written tracks the number of successful record() calls."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config())
        for _ in range(4):
            plugin.pytest_runtest_logreport(_make_report())
        assert plugin._records_written == 4

    def test_each_report_has_distinct_nodeid_in_payload(self):
        """Verify each record() call carries the correct nodeid."""
        mock_aria = _make_mock_aria()
        with patch("pytest_aria.plugin.ARIAQuick", return_value=mock_aria):
            plugin = ARIAPlugin(_make_config())

        nodeids = [
            "tests/test_a.py::test_one",
            "tests/test_b.py::test_two",
            "tests/test_c.py::test_three",
        ]
        for nid in nodeids:
            plugin.pytest_runtest_logreport(_make_report(nodeid=nid))

        recorded_nodeids = [
            c.kwargs["model_id"] for c in mock_aria.record.call_args_list
        ]
        assert recorded_nodeids == nodeids


# ===========================================================================
# 8 — aria-bsv import failure path
# ===========================================================================


class TestImportFailure:
    def test_import_error_leaves_plugin_disabled(self):
        with patch(
            "pytest_aria.plugin.ARIAQuick",
            side_effect=ImportError("aria not found"),
        ):
            plugin = ARIAPlugin(_make_config(enabled=True))
        assert plugin._enabled is False

    def test_no_record_after_import_failure(self):
        with patch(
            "pytest_aria.plugin.ARIAQuick",
            side_effect=ImportError("aria not found"),
        ):
            plugin = ARIAPlugin(_make_config(enabled=True))
        # Must not raise even though _aria is None.
        plugin.pytest_runtest_logreport(_make_report())
        assert plugin._records_written == 0

    def test_no_close_after_import_failure(self):
        with patch(
            "pytest_aria.plugin.ARIAQuick",
            side_effect=ImportError("aria not found"),
        ):
            plugin = ARIAPlugin(_make_config(enabled=True))
        # pytest_sessionfinish must not raise.
        plugin.pytest_sessionfinish(_make_session(), exitstatus=0)
