"""Pydantic request/response schemas for the ARIA Registry API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EuRiskLevel = Literal["prohibited", "high", "limited", "minimal"]


class SystemCreate(BaseModel):
    """Fields required to register an AI system."""

    system_id: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-z0-9_-]+$")
    system_name: str = Field(..., min_length=1, max_length=256)
    version: str = Field(default="1.0.0", max_length=64)
    description: str = Field(default="", max_length=2048)
    operator_name: str = Field(..., min_length=1, max_length=256)
    operator_contact: str = Field(default="", max_length=256)
    # EU AI Act compliance fields
    eu_ai_act_risk_level: EuRiskLevel = "minimal"
    eu_ai_act_article: str = Field(default="", max_length=128)
    deployment_context: str = Field(default="", max_length=512)


class SystemRead(SystemCreate):
    """System registration record — returned by GET endpoints."""

    registered_at: datetime
    total_epochs: int = 0
    total_records: int = 0
    last_epoch_at: datetime | None = None

    model_config = {"from_attributes": True}


class EpochCreate(BaseModel):
    """Payload to record a closed ARIA epoch under a registered system."""

    epoch_id: str = Field(..., min_length=1, max_length=128)
    open_txid: str = Field(..., min_length=64, max_length=64)
    close_txid: str = Field(default="", max_length=64)
    records_count: int = Field(default=0, ge=0)
    merkle_root: str = Field(default="", max_length=128)
    opened_at: int = Field(..., description="Unix timestamp in seconds")
    closed_at: int = Field(default=0, description="Unix timestamp in seconds; 0 if still open")


class EpochRecord(BaseModel):
    """A single epoch entry in a system's history."""

    epoch_id: str
    open_txid: str
    close_txid: str
    records_count: int
    merkle_root: str
    opened_at: datetime
    closed_at: datetime | None

    model_config = {"from_attributes": True}
