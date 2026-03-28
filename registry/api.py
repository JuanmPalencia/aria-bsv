"""ARIA Registry — FastAPI application."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import EpochCreate, EpochRecord, SystemCreate, SystemRead
from .storage import RegistryStorage

app = FastAPI(
    title="ARIA Registry",
    description="Public directory of AI systems audited via BRC-121.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage dependency
# ---------------------------------------------------------------------------

_storage: RegistryStorage | None = None


def get_storage() -> RegistryStorage:
    global _storage
    if _storage is None:
        _storage = RegistryStorage()
    return _storage


StorageDep = Annotated[RegistryStorage, Depends(get_storage)]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/systems", response_model=SystemRead, status_code=status.HTTP_201_CREATED)
def register_system(
    data: SystemCreate,
    store: StorageDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> SystemRead:
    """Register a new AI system.

    The caller supplies their own secret ``X-API-Key`` header — it is stored
    as a SHA-256 hash and required for all subsequent write operations on this
    system.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-API-Key header required",
        )
    if store.get_system(data.system_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"system_id {data.system_id!r} is already registered",
        )
    return store.create_system(data, x_api_key)


@app.get("/systems", response_model=list[SystemRead])
def list_systems(store: StorageDep) -> list[SystemRead]:
    """List all registered systems (public)."""
    return store.list_systems()


@app.get("/systems/{system_id}", response_model=SystemRead)
def get_system(system_id: str, store: StorageDep) -> SystemRead:
    """Get a single registered system with aggregate statistics (public)."""
    row = store.get_system(system_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="system not found")
    return row


@app.get("/systems/{system_id}/history", response_model=list[EpochRecord])
def get_epoch_history(system_id: str, store: StorageDep) -> list[EpochRecord]:
    """Return the epoch history for a system, newest first (public)."""
    if store.get_system(system_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="system not found")
    return store.list_epochs(system_id)


@app.post(
    "/systems/{system_id}/epochs",
    response_model=EpochRecord,
    status_code=status.HTTP_201_CREATED,
)
def record_epoch(
    system_id: str,
    data: EpochCreate,
    store: StorageDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> EpochRecord:
    """Record a closed ARIA epoch for an existing system (requires API key)."""
    if store.get_system(system_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="system not found")
    if not x_api_key or not store.verify_api_key(system_id, x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return store.record_epoch(system_id, data)
