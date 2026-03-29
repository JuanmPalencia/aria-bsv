"""Tests for aria.config_file — project configuration loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from aria.config_file import (
    ARIAProjectConfig,
    _apply_env_overrides,
    _find_config_file,
    _parse_toml,
    generate_config_template,
    load_config,
)
from aria.core.errors import ARIAConfigError


class TestARIAProjectConfig:
    """Tests for the ARIAProjectConfig dataclass."""

    def test_defaults(self):
        cfg = ARIAProjectConfig()
        assert cfg.system_id == ""
        assert cfg.network == "testnet"
        assert cfg.batch_ms == 5_000
        assert cfg.batch_size == 500
        assert cfg.offline is False
        assert cfg.pii_fields == []

    def test_to_dict(self):
        cfg = ARIAProjectConfig(system_id="test", network="mainnet")
        d = cfg.to_dict()
        assert d["system_id"] == "test"
        assert d["network"] == "mainnet"

    def test_to_audit_config_requires_system_id(self):
        cfg = ARIAProjectConfig()
        with pytest.raises(ARIAConfigError, match="system_id is required"):
            cfg.to_audit_config()

    def test_to_audit_config_auto_key(self):
        cfg = ARIAProjectConfig(system_id="test-system")
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("ARIA_")}
        with pytest.MonkeyPatch.context() as mp:
            for k in list(os.environ):
                if k.startswith("ARIA_"):
                    mp.delenv(k, raising=False)
            audit_cfg = cfg.to_audit_config()
            assert audit_cfg.system_id == "test-system"
            assert audit_cfg.bsv_key  # auto-generated


class TestFindConfigFile:
    """Tests for _find_config_file."""

    def test_finds_aria_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "aria.toml"
            toml_path.write_text('[aria]\nsystem_id = "found"\n')
            result = _find_config_file(start=Path(tmpdir))
            assert result is not None
            assert result.name == "aria.toml"

    def test_finds_pyproject_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "pyproject.toml"
            toml_path.write_text('[tool.aria]\nsystem_id = "found"\n')
            result = _find_config_file(start=Path(tmpdir))
            assert result is not None
            assert result.name == "pyproject.toml"

    def test_aria_toml_takes_precedence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "aria.toml").write_text('[aria]\nsystem_id = "aria"\n')
            (Path(tmpdir) / "pyproject.toml").write_text('[tool.aria]\nsystem_id = "pyproject"\n')
            result = _find_config_file(start=Path(tmpdir))
            assert result is not None
            assert result.name == "aria.toml"

    def test_returns_none_when_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _find_config_file(start=Path(tmpdir))
            assert result is None

    def test_walks_up_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            (parent / "aria.toml").write_text('[aria]\nsystem_id = "parent"\n')
            child = parent / "subdir" / "deep"
            child.mkdir(parents=True)
            result = _find_config_file(start=child)
            assert result is not None
            assert result.name == "aria.toml"


class TestParseToml:
    """Tests for _parse_toml."""

    def test_parse_aria_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aria.toml"
            path.write_text('[aria]\nsystem_id = "test"\nnetwork = "mainnet"\n')
            data = _parse_toml(path)
            assert data["system_id"] == "test"
            assert data["network"] == "mainnet"

    def test_parse_pyproject_toml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pyproject.toml"
            path.write_text('[tool.aria]\nsystem_id = "pyp"\n')
            data = _parse_toml(path)
            assert data["system_id"] == "pyp"


class TestApplyEnvOverrides:
    """Tests for _apply_env_overrides."""

    def test_env_override_system_id(self, monkeypatch):
        monkeypatch.setenv("ARIA_SYSTEM_ID", "from-env")
        cfg = ARIAProjectConfig()
        cfg = _apply_env_overrides(cfg)
        assert cfg.system_id == "from-env"

    def test_env_override_batch_ms(self, monkeypatch):
        monkeypatch.setenv("ARIA_BATCH_MS", "3000")
        cfg = ARIAProjectConfig()
        cfg = _apply_env_overrides(cfg)
        assert cfg.batch_ms == 3000

    def test_env_override_network(self, monkeypatch):
        monkeypatch.setenv("ARIA_MODE", "mainnet")
        cfg = ARIAProjectConfig()
        cfg = _apply_env_overrides(cfg)
        assert cfg.network == "mainnet"

    def test_env_override_offline(self, monkeypatch):
        monkeypatch.setenv("ARIA_OFFLINE", "true")
        cfg = ARIAProjectConfig()
        cfg = _apply_env_overrides(cfg)
        assert cfg.offline is True


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aria.toml"
            path.write_text('[aria]\nsystem_id = "explicit"\n')
            cfg = load_config(path=str(path))
            assert cfg.system_id == "explicit"
            assert cfg.source == "aria.toml"

    def test_load_missing_file_raises(self):
        with pytest.raises(ARIAConfigError, match="Config file not found"):
            load_config(path="/nonexistent/aria.toml")

    def test_auto_discover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aria.toml"
            path.write_text('[aria]\nsystem_id = "auto"\n')
            cfg = load_config(start_dir=Path(tmpdir))
            assert cfg.system_id == "auto"

    def test_env_overrides_file(self, monkeypatch):
        monkeypatch.setenv("ARIA_SYSTEM_ID", "env-wins")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aria.toml"
            path.write_text('[aria]\nsystem_id = "file"\n')
            cfg = load_config(path=str(path))
            assert cfg.system_id == "env-wins"

    def test_no_config_file_uses_env(self, monkeypatch):
        monkeypatch.setenv("ARIA_SYSTEM_ID", "env-only")
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_config(start_dir=Path(tmpdir))
            assert cfg.system_id == "env-only"
            assert cfg.source == "env"


class TestGenerateConfigTemplate:
    """Tests for generate_config_template."""

    def test_template_contains_system_id(self):
        tpl = generate_config_template("my-app", "mainnet")
        assert "my-app" in tpl
        assert "mainnet" in tpl

    def test_template_is_valid_toml(self):
        tpl = generate_config_template("test", "testnet")
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        data = tomllib.loads(tpl)
        assert data["aria"]["system_id"] == "test"

    def test_default_args(self):
        tpl = generate_config_template()
        assert "my-ai-system" in tpl
        assert "testnet" in tpl
