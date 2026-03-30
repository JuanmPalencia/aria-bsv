"""
pytest_aria.plugin — pytest plugin that anchors test run results to BSV via
the ARIA BRC-121 audit system.

Each test *call* phase is recorded as a single inference record inside a shared
ARIA epoch.  When the test session ends, the epoch is closed and the summary
(including the Merkle root of all test outcomes) is logged.

Configuration (pytest.ini / pyproject.toml ``[tool:pytest]`` / CLI):

    aria_enabled         = true   # master switch
    aria_system_id       = pytest # system_id passed to ARIAQuick
    aria_db_path         = aria_tests.db
    aria_record_passed   = true
    aria_record_failed   = true
    aria_record_skipped  = false

CLI flags:
    --aria-enabled / --no-aria-enabled
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

_log = logging.getLogger("pytest_aria")

# ---------------------------------------------------------------------------
# Optional ARIA dependency — imported at module level so tests can patch it
# ---------------------------------------------------------------------------

try:
    from aria.quick import ARIAQuick  # noqa: F401 (re-exported for patch target)
except ImportError:  # pragma: no cover
    ARIAQuick = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Option registration
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("aria", "ARIA BRC-121 audit options")

    # CLI flags ---------------------------------------------------------------
    group.addoption(
        "--aria-enabled",
        action="store_true",
        default=None,
        dest="aria_enabled_flag",
        help="Enable ARIA BRC-121 anchoring for this test run (overrides ini).",
    )
    group.addoption(
        "--no-aria-enabled",
        action="store_false",
        dest="aria_enabled_flag",
        help="Disable ARIA BRC-121 anchoring for this test run.",
    )

    # ini options -------------------------------------------------------------
    parser.addini(
        "aria_enabled",
        type="bool",
        default=True,
        help="Enable ARIA BRC-121 test run anchoring.",
    )
    parser.addini(
        "aria_system_id",
        default="pytest",
        help="system_id passed to ARIAQuick (identifies this system in the audit log).",
    )
    parser.addini(
        "aria_db_path",
        default="aria_tests.db",
        help="SQLite database path for the ARIA audit log.",
    )
    parser.addini(
        "aria_record_passed",
        type="bool",
        default=True,
        help="Record passing tests in the ARIA audit log.",
    )
    parser.addini(
        "aria_record_failed",
        type="bool",
        default=True,
        help="Record failing tests in the ARIA audit log.",
    )
    parser.addini(
        "aria_record_skipped",
        type="bool",
        default=False,
        help="Record skipped tests in the ARIA audit log.",
    )


# ---------------------------------------------------------------------------
# Configure hook — register the plugin instance
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register ARIAPlugin with the plugin manager."""
    plugin = ARIAPlugin(config)
    config.pluginmanager.register(plugin, "aria")


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class ARIAPlugin:
    """Captures pytest test results and records them via ARIA."""

    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._aria: Any = None
        self._enabled: bool = False
        self._record_passed: bool = True
        self._record_failed: bool = True
        self._record_skipped: bool = False
        self._records_written: int = 0
        self._session_start: float = time.time()

        self._setup(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup(self, config: pytest.Config) -> None:
        """Read configuration and initialise ARIAQuick (if enabled)."""
        # CLI flag takes priority over ini value
        cli_flag = getattr(config.option, "aria_enabled_flag", None)
        if cli_flag is not None:
            enabled = cli_flag
        else:
            enabled = config.getini("aria_enabled")

        if not enabled:
            _log.debug("ARIA plugin disabled via configuration.")
            return

        system_id: str = config.getini("aria_system_id") or "pytest"
        db_path: str = config.getini("aria_db_path") or "aria_tests.db"
        self._record_passed = bool(config.getini("aria_record_passed"))
        self._record_failed = bool(config.getini("aria_record_failed"))
        self._record_skipped = bool(config.getini("aria_record_skipped"))

        try:
            if ARIAQuick is None:  # import failed at module level
                raise ImportError("aria-bsv is not installed")
            self._aria = ARIAQuick(
                system_id=system_id,
                db_path=db_path,
                watchdog=False,
                compliance=False,  # keep overhead minimal during test runs
            )
            self._aria.start()
            self._enabled = True
            _log.info(
                "ARIA plugin active — system_id=%r  db=%r", system_id, db_path
            )
        except ImportError:
            _log.warning(
                "pytest-aria: aria-bsv is not installed. "
                "Test run anchoring is disabled. "
                "Install with: pip install aria-bsv"
            )
        except Exception as exc:  # pragma: no cover
            _log.warning("pytest-aria: failed to initialise ARIAQuick: %s", exc)

    # ------------------------------------------------------------------
    # pytest hooks
    # ------------------------------------------------------------------

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Called after each test phase (setup / call / teardown).

        We only act on the ``call`` phase, which represents the actual
        execution of the test body.
        """
        if not self._enabled or self._aria is None:
            return

        # Only record the test-body phase, not setup/teardown.
        if report.when != "call":
            return

        outcome: str = report.outcome  # "passed" / "failed" / "skipped"

        # Apply per-outcome recording filters.
        if outcome == "passed" and not self._record_passed:
            return
        if outcome == "failed" and not self._record_failed:
            return
        if outcome == "skipped" and not self._record_skipped:
            return

        nodeid: str = report.nodeid
        duration_ms: float = round(getattr(report, "duration", 0.0) * 1000, 3)

        # Parse module and test name from the nodeid for richer input_data.
        parts = nodeid.rsplit("::", 1)
        module = parts[0] if len(parts) == 2 else nodeid
        test_name = parts[1] if len(parts) == 2 else nodeid

        input_data: dict[str, Any] = {
            "test": nodeid,
            "module": module,
            "test_name": test_name,
        }
        output_data: dict[str, Any] = {
            "outcome": outcome,
            "duration_ms": duration_ms,
        }

        # confidence: 1.0 for passed, 0.0 for everything else.
        confidence: float = 1.0 if outcome == "passed" else 0.0

        try:
            self._aria.record(
                model_id=nodeid,
                input_data=input_data,
                output_data=output_data,
                confidence=confidence,
                latency_ms=duration_ms,
            )
            self._records_written += 1
        except Exception as exc:
            _log.warning("pytest-aria: record() failed for %r: %s", nodeid, exc)

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: int
    ) -> None:
        """Called once after the entire test session completes."""
        if not self._enabled or self._aria is None:
            return

        elapsed_s = time.time() - self._session_start

        try:
            summary = self._aria.close()
            _log.info(
                "ARIA epoch closed — epoch_id=%s  records=%d  "
                "merkle_root=%.16s...  elapsed_session=%.1fs",
                summary.epoch_id,
                self._records_written,
                summary.merkle_root or "(none)",
                elapsed_s,
            )
            # Also emit to the terminal via the reporting system so it is
            # visible in normal test output without requiring --log-cli-level.
            try:
                reporter = session.config.pluginmanager.get_plugin(
                    "terminalreporter"
                )
                if reporter is not None:
                    reporter.write_sep(
                        "-",
                        f"ARIA BRC-121: {self._records_written} tests anchored "
                        f"(epoch {summary.epoch_id[:16]}...)",
                    )
            except Exception:  # pragma: no cover
                pass
        except Exception as exc:
            _log.warning("pytest-aria: close() failed: %s", exc)
