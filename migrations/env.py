"""Alembic migration environment for ARIA.

Database URL is resolved in order:
  1. ARIA_DB_URL environment variable
  2. ``-x db_url=...`` CLI option passed to alembic
  3. ``sqlalchemy.url`` from alembic.ini (fallback / placeholder)
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Alembic Config — gives access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import ARIA schema metadata so Alembic can auto-generate migrations
# ---------------------------------------------------------------------------
from aria.storage._schema import _Base  # noqa: E402

target_metadata = _Base.metadata

# ---------------------------------------------------------------------------
# URL resolution — env var takes priority over alembic.ini placeholder
# ---------------------------------------------------------------------------

def _get_url() -> str:
    # 1. Environment variable
    env_url = os.environ.get("ARIA_DB_URL")
    if env_url:
        return env_url
    # 2. -x db_url=... CLI option
    x_url = context.get_x_argument(as_dictionary=True).get("db_url")
    if x_url:
        return x_url
    # 3. alembic.ini value
    return config.get_main_option("sqlalchemy.url", "")


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations without a live connection (generates SQL script)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _get_url()
    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
