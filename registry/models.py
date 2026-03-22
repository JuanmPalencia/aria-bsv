"""SQLAlchemy ORM models for the ARIA Registry database."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class _Base(DeclarativeBase):
    pass


class SystemModel(_Base):
    __tablename__ = "registry_systems"

    system_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    system_name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(64), default="1.0.0")
    description: Mapped[str] = mapped_column(String(2048), default="")
    operator_name: Mapped[str] = mapped_column(String(256), nullable=False)
    operator_contact: Mapped[str] = mapped_column(String(256), default="")
    eu_ai_act_risk_level: Mapped[str] = mapped_column(String(16), default="minimal")
    eu_ai_act_article: Mapped[str] = mapped_column(String(128), default="")
    deployment_context: Mapped[str] = mapped_column(String(512), default="")
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class EpochModel(_Base):
    __tablename__ = "registry_epochs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    system_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    epoch_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    open_txid: Mapped[str] = mapped_column(String(64), nullable=False)
    close_txid: Mapped[str] = mapped_column(String(64), default="")
    records_count: Mapped[int] = mapped_column(Integer, default=0)
    merkle_root: Mapped[str] = mapped_column(String(128), default="")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
