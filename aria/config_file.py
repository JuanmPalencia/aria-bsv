"""
aria.config_file — Declarative project configuration via ``aria.toml``.

Supports loading from ``aria.toml``, ``pyproject.toml [tool.aria]``,
or environment variables (env vars take highest precedence).

Usage::

    from aria.config_file import load_config, ARIAProjectConfig

    config = load_config()           # auto-discovers aria.toml or pyproject.toml
    auditor = config.to_audit_config()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.errors import ARIAConfigError

_log = logging.getLogger(__name__)

# Search order for config files (cwd → parent → parent… up to 5 levels)
_CONFIG_FILENAMES = ("aria.toml", "pyproject.toml")
_MAX_SEARCH_DEPTH = 5


@dataclass
class ARIAProjectConfig:
    """Parsed ARIA project configuration.

    All fields are optional — sensible defaults are applied.
    Environment variables override file values.
    """

    system_id: str = ""
    network: str = "testnet"
    storage: str = ""
    arc_url: str = ""
    arc_api_key: str = ""
    bsv_key: str = ""
    brc100_url: str = ""
    batch_ms: int = 5_000
    batch_size: int = 500
    pii_fields: list[str] = field(default_factory=list)
    passphrase: str = ""
    offline: bool = False
    auto_epoch: str = ""

    # Metadata
    config_path: str = ""
    source: str = "defaults"

    def to_audit_config(self):
        """Convert to an AuditConfig suitable for InferenceAuditor."""
        from .auditor import AuditConfig

        system_id = self.system_id
        if not system_id:
            raise ARIAConfigError(
                "system_id is required. Set it in aria.toml or ARIA_SYSTEM_ID env var."
            )

        bsv_key = self.bsv_key or os.environ.get("ARIA_BSV_KEY")
        brc100_url = self.brc100_url or None

        if not bsv_key and not brc100_url:
            from .auto_config import get_or_create_wif
            bsv_key, _, _ = get_or_create_wif(
                network=self.network, passphrase=self.passphrase or None
            )

        storage = self.storage or f"sqlite:///aria_{system_id}.db"
        arc_url = self.arc_url or "https://arc.taal.com"

        return AuditConfig(
            system_id=system_id,
            bsv_key=bsv_key,
            brc100_url=brc100_url,
            storage=storage,
            batch_ms=self.batch_ms,
            batch_size=self.batch_size,
            arc_url=arc_url,
            arc_api_key=self.arc_api_key or None,
            network=self.network,
            pii_fields=self.pii_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "network": self.network,
            "storage": self.storage,
            "arc_url": self.arc_url,
            "batch_ms": self.batch_ms,
            "batch_size": self.batch_size,
            "pii_fields": self.pii_fields,
            "offline": self.offline,
            "auto_epoch": self.auto_epoch,
            "config_path": self.config_path,
            "source": self.source,
        }


def _find_config_file(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a config file."""
    current = (start or Path.cwd()).resolve()
    for _ in range(_MAX_SEARCH_DEPTH):
        for name in _CONFIG_FILENAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _parse_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file and extract the ARIA config section."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)

    if path.name == "pyproject.toml":
        return data.get("tool", {}).get("aria", {})
    return data.get("aria", data)


def _apply_env_overrides(cfg: ARIAProjectConfig) -> ARIAProjectConfig:
    """Override config values from environment variables."""
    env_map = {
        "ARIA_SYSTEM_ID": "system_id",
        "ARIA_MODE": "network",
        "ARIA_DB": "storage",
        "ARIA_ARC_URL": "arc_url",
        "ARIA_ARC_KEY": "arc_api_key",
        "ARIA_BSV_KEY": "bsv_key",
        "ARIA_BRC100_URL": "brc100_url",
        "ARIA_PASSPHRASE": "passphrase",
        "ARIA_BATCH_MS": "batch_ms",
        "ARIA_BATCH_SIZE": "batch_size",
    }
    for env_var, attr in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            current = getattr(cfg, attr)
            if isinstance(current, int):
                val = int(val)
            elif isinstance(current, bool):
                val = val.lower() in ("1", "true", "yes")
            setattr(cfg, attr, val)

    offline = os.environ.get("ARIA_OFFLINE")
    if offline is not None:
        cfg.offline = offline.lower() in ("1", "true", "yes")

    return cfg


def load_config(
    path: str | Path | None = None,
    start_dir: Path | None = None,
) -> ARIAProjectConfig:
    """Load ARIA configuration from file + environment.

    Search order:
      1. Explicit *path* argument
      2. ``aria.toml`` in cwd or parent directories
      3. ``pyproject.toml`` → ``[tool.aria]`` section
      4. Environment variables only (no file)

    Environment variables always override file values.

    Returns:
        Populated ARIAProjectConfig.
    """
    cfg = ARIAProjectConfig()

    config_path: Path | None = None
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ARIAConfigError(f"Config file not found: {config_path}")
    else:
        config_path = _find_config_file(start_dir)

    if config_path is not None:
        data = _parse_toml(config_path)
        cfg.config_path = str(config_path)
        cfg.source = "aria.toml" if config_path.name == "aria.toml" else "pyproject.toml"

        cfg.system_id = data.get("system_id", cfg.system_id)
        cfg.network = data.get("network", cfg.network)
        cfg.storage = data.get("storage", cfg.storage)
        cfg.arc_url = data.get("arc_url", cfg.arc_url)
        cfg.arc_api_key = data.get("arc_api_key", cfg.arc_api_key)
        cfg.bsv_key = data.get("bsv_key", cfg.bsv_key)
        cfg.brc100_url = data.get("brc100_url", cfg.brc100_url)
        cfg.batch_ms = data.get("batch_ms", cfg.batch_ms)
        cfg.batch_size = data.get("batch_size", cfg.batch_size)
        cfg.pii_fields = data.get("pii_fields", cfg.pii_fields)
        cfg.passphrase = data.get("passphrase", cfg.passphrase)
        cfg.offline = data.get("offline", cfg.offline)
        cfg.auto_epoch = data.get("auto_epoch", cfg.auto_epoch)

        _log.info("Loaded config from %s", config_path)
    else:
        cfg.source = "env"
        _log.debug("No config file found, using env vars + defaults.")

    cfg = _apply_env_overrides(cfg)
    return cfg


def generate_config_template(
    system_id: str = "my-ai-system",
    network: str = "testnet",
) -> str:
    """Generate a sample aria.toml template string."""
    return f"""# ARIA Configuration
# Generated by: aria init

[aria]
system_id = "{system_id}"
network = "{network}"

# Storage (SQLite by default, PostgreSQL for production)
# storage = "sqlite:///aria_{system_id}.db"
# storage = "postgresql://user:pass@localhost/aria"

# BSV broadcasting
# arc_url = "https://arc.taal.com"
# arc_api_key = ""

# Epoch auto-rotation (optional)
# auto_epoch = "every_100_records"

# PII fields to redact before hashing
# pii_fields = ["name", "email", "ssn"]

# Offline mode — records stored locally, sync later with 'aria sync'
# offline = false
"""
