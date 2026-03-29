"""Tests for aria.mcp_server — MCP tool functions.

Since the ``mcp`` package is not installed in the test environment, the entire
``mcp`` package is mocked in sys.modules before the module is imported.  Each
tool function is tested by calling it directly as a plain Python function —
no MCP protocol transport is involved.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake mcp module factory
# ---------------------------------------------------------------------------


def _make_mcp_module() -> ModuleType:
    """Build a minimal fake ``mcp`` hierarchy that satisfies the import."""
    mcp_mod = ModuleType("mcp")
    server_mod = ModuleType("mcp.server")
    fastmcp_mod = ModuleType("mcp.server.fastmcp")

    class _FakeFastMCP:
        """Minimal FastMCP stand-in. tool() is a no-op decorator."""

        def __init__(self, name: str, description: str = "") -> None:
            self.name = name
            self.description = description

        def tool(self):
            """Return a transparent pass-through decorator."""
            def decorator(fn):
                return fn
            return decorator

        def run(self) -> None:
            pass

    fastmcp_mod.FastMCP = _FakeFastMCP
    mcp_mod.server = server_mod
    server_mod.fastmcp = fastmcp_mod

    return mcp_mod, server_mod, fastmcp_mod


def _install_fake_mcp() -> None:
    """Insert all fake mcp sub-modules into sys.modules."""
    mcp_mod, server_mod, fastmcp_mod = _make_mcp_module()
    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.fastmcp"] = fastmcp_mod


# ---------------------------------------------------------------------------
# Helper: load (or reload) aria.mcp_server with mocked mcp + ARIAQuick
# ---------------------------------------------------------------------------


def _load_module(mock_aria_quick: MagicMock | None = None):
    """Return a freshly reloaded aria.mcp_server with fakes in place.

    ``mock_aria_quick`` is installed as the ``ARIAQuick`` class used by the
    module so that no real SQLite or BSV logic is executed.
    """
    _install_fake_mcp()

    # Ensure module is importable (or already cached) — then reload.
    import aria.mcp_server  # noqa: F401 — ensure it is in sys.modules first

    if mock_aria_quick is not None:
        with patch("aria.mcp_server.ARIAQuick", mock_aria_quick, create=True):
            with patch("aria.mcp_server._aria_instance", None, create=True):
                mod = importlib.reload(aria.mcp_server)
                mod._aria_instance = None  # reset lazy singleton
    else:
        mod = importlib.reload(aria.mcp_server)
        mod._aria_instance = None

    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level _aria_instance before each test."""
    _install_fake_mcp()
    import aria.mcp_server as m
    m._aria_instance = None
    yield
    m._aria_instance = None


def _make_mock_record(
    record_id: str = "rec_epoch1_000000",
    epoch_id: str = "epoch-abc",
    model_id: str = "test-model",
    record_hash: str = "sha256:" + "a" * 64,
) -> MagicMock:
    rec = MagicMock()
    rec.record_id = record_id
    rec.epoch_id = epoch_id
    rec.model_id = model_id
    rec.hash.return_value = record_hash
    return rec


def _make_mock_aria(
    system_id: str = "test-system",
    db_path: str = "test.db",
    current_epoch_id: str = "epoch-abc",
    record_return: str = "rec_epoch_abc_000000",
    record_obj: MagicMock | None = None,
    close_summary: MagicMock | None = None,
) -> MagicMock:
    aria = MagicMock()
    aria.system_id = system_id
    aria._db_path = db_path
    aria.current_epoch_id = current_epoch_id
    aria.record.return_value = record_return

    # Default record object returned by storage.get_record
    stored_record = record_obj or _make_mock_record(
        record_id=record_return,
        epoch_id=current_epoch_id,
    )
    aria.storage.get_record.return_value = stored_record
    aria.storage.list_records_by_epoch.return_value = [stored_record]

    # Default close summary
    if close_summary is None:
        cs = MagicMock()
        cs.epoch_id = current_epoch_id
        cs.merkle_root = "sha256:" + "b" * 64
        cs.records_count = 1
        cs.anchored = False
        close_summary = cs
    aria.close.return_value = close_summary

    return aria


# ---------------------------------------------------------------------------
# Tests: aria_record
# ---------------------------------------------------------------------------


class TestAriaRecord:
    def _mod_with_aria(self, mock_aria=None):
        if mock_aria is None:
            mock_aria = _make_mock_aria()
        mod = _load_module()
        mod._aria_instance = mock_aria
        return mod

    def test_returns_record_id(self):
        aria = _make_mock_aria(record_return="rec_ep1_000000")
        mod = self._mod_with_aria(aria)
        result = mod.aria_record("gpt-4", "hello", "world")
        assert result["record_id"] == "rec_ep1_000000"

    def test_returns_epoch_id(self):
        aria = _make_mock_aria(current_epoch_id="epoch-xyz")
        mod = self._mod_with_aria(aria)
        result = mod.aria_record("gpt-4", "hi", "there")
        assert result["epoch_id"] == "epoch-xyz"

    def test_returns_record_hash_key(self):
        mod = self._mod_with_aria()
        result = mod.aria_record("gpt-4", "in", "out")
        assert "record_hash" in result

    def test_record_hash_is_sha256_prefixed(self):
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = _make_mock_record(
            record_hash="sha256:" + "c" * 64
        )
        mod = self._mod_with_aria(aria)
        result = mod.aria_record("gpt-4", "in", "out")
        assert result["record_hash"].startswith("sha256:")

    def test_calls_aria_record_with_model_id(self):
        aria = _make_mock_aria()
        mod = self._mod_with_aria(aria)
        mod.aria_record("my-model", "prompt", "response", confidence=0.9)
        call_kwargs = aria.record.call_args[1]
        assert call_kwargs["model_id"] == "my-model"

    def test_calls_aria_record_with_confidence(self):
        aria = _make_mock_aria()
        mod = self._mod_with_aria(aria)
        mod.aria_record("m", "i", "o", confidence=0.75)
        call_kwargs = aria.record.call_args[1]
        assert call_kwargs["confidence"] == 0.75

    def test_error_returns_error_dict(self):
        aria = _make_mock_aria()
        aria.record.side_effect = RuntimeError("db locked")
        mod = self._mod_with_aria(aria)
        result = mod.aria_record("m", "i", "o")
        assert "error" in result
        assert "db locked" in result["error"]

    def test_bsv_wif_not_in_output(self):
        """BSV key must never appear in tool output."""
        aria = _make_mock_aria()
        aria._bsv_wif = "cNqKfake123456WIF"
        mod = self._mod_with_aria(aria)
        result = mod.aria_record("m", "i", "o")
        result_str = str(result)
        assert "cNqKfake123456WIF" not in result_str


# ---------------------------------------------------------------------------
# Tests: aria_status
# ---------------------------------------------------------------------------


class TestAriaStatus:
    def _mod_with_aria(self, mock_aria=None):
        if mock_aria is None:
            mock_aria = _make_mock_aria()
        mod = _load_module()
        mod._aria_instance = mock_aria
        return mod

    def test_returns_system_id(self):
        aria = _make_mock_aria(system_id="prod-system")
        mod = self._mod_with_aria(aria)
        result = mod.aria_status()
        assert result["system_id"] == "prod-system"

    def test_returns_epoch_id(self):
        aria = _make_mock_aria(current_epoch_id="epoch-001")
        mod = self._mod_with_aria(aria)
        result = mod.aria_status()
        assert result["epoch_id"] == "epoch-001"

    def test_returns_record_count(self):
        aria = _make_mock_aria()
        aria.storage.list_records_by_epoch.return_value = [MagicMock(), MagicMock()]
        mod = self._mod_with_aria(aria)
        result = mod.aria_status()
        assert result["record_count"] == 2

    def test_returns_db_path(self):
        aria = _make_mock_aria(db_path="/tmp/aria.db")
        mod = self._mod_with_aria(aria)
        result = mod.aria_status()
        assert result["db_path"] == "/tmp/aria.db"

    def test_error_returns_error_dict(self):
        aria = _make_mock_aria()
        aria.system_id = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        aria.system_id = MagicMock(side_effect=RuntimeError("fail"))
        # Simulate error by making _get_aria raise
        mod = _load_module()
        mod._aria_instance = None

        def _bad_aria(*a, **kw):
            raise RuntimeError("init failed")

        with patch("aria.mcp_server._get_aria", _bad_aria):
            result = mod.aria_status()
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: aria_verify_local
# ---------------------------------------------------------------------------


class TestAriaVerifyLocal:
    def _mod_with_aria(self, mock_aria=None):
        if mock_aria is None:
            mock_aria = _make_mock_aria()
        mod = _load_module()
        mod._aria_instance = mock_aria
        return mod

    def test_returns_record_id(self):
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = _make_mock_record(
            record_id="rec_ep_000001"
        )
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("rec_ep_000001")
        assert result["record_id"] == "rec_ep_000001"

    def test_returns_record_hash(self):
        the_hash = "sha256:" + "d" * 64
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = _make_mock_record(record_hash=the_hash)
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("any_id")
        assert result["record_hash"] == the_hash

    def test_returns_epoch_id(self):
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = _make_mock_record(epoch_id="ep-999")
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("any_id")
        assert result["epoch_id"] == "ep-999"

    def test_returns_model_id(self):
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = _make_mock_record(model_id="claude-3")
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("any_id")
        assert result["model_id"] == "claude-3"

    def test_not_found_returns_error_dict(self):
        aria = _make_mock_aria()
        aria.storage.get_record.return_value = None
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("nonexistent-id")
        assert "error" in result
        assert "nonexistent-id" in result["error"]

    def test_storage_exception_returns_error_dict(self):
        aria = _make_mock_aria()
        aria.storage.get_record.side_effect = RuntimeError("storage error")
        mod = self._mod_with_aria(aria)
        result = mod.aria_verify_local("any_id")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: aria_close_epoch
# ---------------------------------------------------------------------------


class TestAriaCloseEpoch:
    def _mod_with_aria(self, mock_aria=None):
        if mock_aria is None:
            mock_aria = _make_mock_aria()
        mod = _load_module()
        mod._aria_instance = mock_aria
        return mod

    def test_returns_merkle_root(self):
        root = "sha256:" + "e" * 64
        cs = MagicMock()
        cs.epoch_id = "ep-1"
        cs.merkle_root = root
        cs.records_count = 3
        cs.anchored = False
        aria = _make_mock_aria(close_summary=cs)
        mod = self._mod_with_aria(aria)
        result = mod.aria_close_epoch()
        assert result["merkle_root"] == root

    def test_returns_epoch_id(self):
        cs = MagicMock()
        cs.epoch_id = "ep-closed-42"
        cs.merkle_root = "sha256:" + "f" * 64
        cs.records_count = 0
        cs.anchored = False
        aria = _make_mock_aria(close_summary=cs)
        mod = self._mod_with_aria(aria)
        result = mod.aria_close_epoch()
        assert result["epoch_id"] == "ep-closed-42"

    def test_returns_records_count(self):
        cs = MagicMock()
        cs.epoch_id = "ep-1"
        cs.merkle_root = "sha256:" + "a" * 64
        cs.records_count = 7
        cs.anchored = True
        aria = _make_mock_aria(close_summary=cs)
        mod = self._mod_with_aria(aria)
        result = mod.aria_close_epoch()
        assert result["records_count"] == 7

    def test_returns_anchored_flag(self):
        cs = MagicMock()
        cs.epoch_id = "ep-1"
        cs.merkle_root = "sha256:" + "a" * 64
        cs.records_count = 1
        cs.anchored = True
        aria = _make_mock_aria(close_summary=cs)
        mod = self._mod_with_aria(aria)
        result = mod.aria_close_epoch()
        assert result["anchored"] is True

    def test_close_error_returns_error_dict(self):
        aria = _make_mock_aria()
        aria.close.side_effect = RuntimeError("epoch already closed")
        mod = self._mod_with_aria(aria)
        result = mod.aria_close_epoch()
        assert "error" in result
        assert "epoch already closed" in result["error"]


# ---------------------------------------------------------------------------
# Tests: aria_hash_text
# ---------------------------------------------------------------------------


class TestAriaHashText:
    def _mod(self):
        mod = _load_module()
        return mod

    def test_returns_sha256_prefixed_hash(self):
        mod = self._mod()
        result = mod.aria_hash_text("hello world")
        assert result["hash"].startswith("sha256:")

    def test_hash_matches_sha256(self):
        mod = self._mod()
        text = "The quick brown fox"
        expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = mod.aria_hash_text(text)
        assert result["hash"] == expected

    def test_returns_length_key(self):
        mod = self._mod()
        result = mod.aria_hash_text("test")
        assert "length" in result

    def test_length_matches_input_character_count(self):
        mod = self._mod()
        text = "abc123"
        result = mod.aria_hash_text(text)
        assert result["length"] == len(text)

    def test_empty_string_hash(self):
        mod = self._mod()
        expected = "sha256:" + hashlib.sha256(b"").hexdigest()
        result = mod.aria_hash_text("")
        assert result["hash"] == expected
        assert result["length"] == 0

    def test_unicode_text(self):
        mod = self._mod()
        text = "こんにちは"
        expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = mod.aria_hash_text(text)
        assert result["hash"] == expected

    def test_hash_length_is_64_hex_chars_after_prefix(self):
        mod = self._mod()
        result = mod.aria_hash_text("any text")
        hex_part = result["hash"][len("sha256:"):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_different_inputs_produce_different_hashes(self):
        mod = self._mod()
        r1 = mod.aria_hash_text("input A")
        r2 = mod.aria_hash_text("input B")
        assert r1["hash"] != r2["hash"]


# ---------------------------------------------------------------------------
# Tests: environment variable configuration
# ---------------------------------------------------------------------------


class TestEnvVarConfiguration:
    def test_system_id_from_env(self):
        """ARIA_SYSTEM_ID env var should be used when initialising ARIAQuick."""
        _install_fake_mcp()
        import aria.mcp_server as m

        captured: list[dict] = []

        class _CapturingQuick:
            def __init__(self, system_id, db_path=None, bsv_wif=None, **kw):
                captured.append({"system_id": system_id, "db_path": db_path})
                self.system_id = system_id
                self._db_path = db_path
                self.current_epoch_id = None
                self.storage = MagicMock()
                self.storage.list_records_by_epoch.return_value = []

            def start(self):
                return self

            def record(self, *a, **kw):
                return "rec_test"

        with patch.dict(
            "os.environ", {"ARIA_SYSTEM_ID": "env-system", "ARIA_DB_PATH": "env.db"}
        ):
            with patch("aria.mcp_server.ARIAQuick", _CapturingQuick):
                m._aria_instance = None
                m.aria_status()

        assert len(captured) >= 1
        assert captured[0]["system_id"] == "env-system"

    def test_db_path_from_env(self):
        """ARIA_DB_PATH env var should be used when initialising ARIAQuick."""
        _install_fake_mcp()
        import aria.mcp_server as m

        captured: list[dict] = []

        class _CapturingQuick:
            def __init__(self, system_id, db_path=None, bsv_wif=None, **kw):
                captured.append({"db_path": db_path})
                self.system_id = system_id
                self._db_path = db_path
                self.current_epoch_id = None
                self.storage = MagicMock()
                self.storage.list_records_by_epoch.return_value = []

            def start(self):
                return self

        with patch.dict(
            "os.environ",
            {"ARIA_SYSTEM_ID": "s", "ARIA_DB_PATH": "/custom/path.db"},
            clear=False,
        ):
            with patch("aria.mcp_server.ARIAQuick", _CapturingQuick):
                m._aria_instance = None
                m.aria_status()

        assert captured[0]["db_path"] == "/custom/path.db"

    def test_bsv_wif_not_logged(self, caplog):
        """BSV WIF key must never appear in any log output."""
        import logging

        _install_fake_mcp()
        import aria.mcp_server as m

        fake_wif = "cTestWIF999SecretKey"

        class _CapturingQuick:
            def __init__(self, system_id, db_path=None, bsv_wif=None, **kw):
                self.system_id = system_id
                self._db_path = db_path
                self.current_epoch_id = "ep1"
                self.storage = MagicMock()
                self.storage.list_records_by_epoch.return_value = []

            def start(self):
                return self

        with caplog.at_level(logging.DEBUG, logger="aria.mcp_server"):
            with patch.dict("os.environ", {"ARIA_BSV_WIF": fake_wif}, clear=False):
                with patch("aria.mcp_server.ARIAQuick", _CapturingQuick):
                    m._aria_instance = None
                    m.aria_status()

        # WIF key must not appear in any log record
        for record in caplog.records:
            assert fake_wif not in record.getMessage()
