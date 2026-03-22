"""Registry database operations — thread-safe SQLite/PostgreSQL backend."""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from .models import EpochModel, SystemModel, _Base
from .schemas import EpochCreate, EpochRecord, SystemCreate, SystemRead


class RegistryStorage:
    """Persistent store for registered systems and their epoch histories.

    Args:
        dsn: SQLAlchemy database URL.  Defaults to ``sqlite:///registry.db``.
             Use ``"sqlite://"`` for an in-memory database (tests).
    """

    def __init__(self, dsn: str = "sqlite:///registry.db") -> None:
        connect_args: dict = {"check_same_thread": False}
        kwargs: dict = {}
        if dsn in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        self._engine = create_engine(dsn, connect_args=connect_args, **kwargs)
        _Base.metadata.create_all(self._engine)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Systems
    # ------------------------------------------------------------------

    def create_system(self, data: SystemCreate, api_key: str) -> SystemRead:
        """Register a new system.  Raises ValueError if system_id already exists."""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        row = SystemModel(
            system_id=data.system_id,
            system_name=data.system_name,
            version=data.version,
            description=data.description,
            operator_name=data.operator_name,
            operator_contact=data.operator_contact,
            eu_ai_act_risk_level=data.eu_ai_act_risk_level,
            eu_ai_act_article=data.eu_ai_act_article,
            deployment_context=data.deployment_context,
            registered_at=datetime.now(tz=timezone.utc),
            api_key_hash=api_key_hash,
        )
        with self._lock, Session(self._engine) as s:
            s.add(row)
            s.commit()
            # snapshot attributes before session closes
            _ = row.system_id
        return self._enrich(row)

    def get_system(self, system_id: str) -> SystemRead | None:
        with Session(self._engine) as s:
            row = s.get(SystemModel, system_id)
        if row is None:
            return None
        return self._enrich(row)

    def list_systems(self) -> list[SystemRead]:
        with Session(self._engine) as s:
            rows = list(s.execute(select(SystemModel)).scalars())
        return [self._enrich(r) for r in rows]

    def verify_api_key(self, system_id: str, api_key: str) -> bool:
        with Session(self._engine) as s:
            row = s.get(SystemModel, system_id)
        if row is None:
            return False
        return row.api_key_hash == hashlib.sha256(api_key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Epochs
    # ------------------------------------------------------------------

    def record_epoch(self, system_id: str, data: EpochCreate) -> EpochRecord:
        opened_at = datetime.fromtimestamp(data.opened_at, tz=timezone.utc)
        closed_at = (
            datetime.fromtimestamp(data.closed_at, tz=timezone.utc) if data.closed_at else None
        )
        row = EpochModel(
            system_id=system_id,
            epoch_id=data.epoch_id,
            open_txid=data.open_txid,
            close_txid=data.close_txid,
            records_count=data.records_count,
            merkle_root=data.merkle_root,
            opened_at=opened_at,
            closed_at=closed_at,
        )
        with self._lock, Session(self._engine) as s:
            s.add(row)
            s.commit()
            # Load all attributes while session is open.
            epoch_id = row.epoch_id
            open_txid = row.open_txid
            close_txid = row.close_txid
            records_count = row.records_count
            merkle_root = row.merkle_root
        return EpochRecord(
            epoch_id=epoch_id,
            open_txid=open_txid,
            close_txid=close_txid,
            records_count=records_count,
            merkle_root=merkle_root,
            opened_at=opened_at,
            closed_at=closed_at,
        )

    def list_epochs(self, system_id: str) -> list[EpochRecord]:
        with Session(self._engine) as s:
            rows = list(
                s.execute(
                    select(EpochModel)
                    .where(EpochModel.system_id == system_id)
                    .order_by(EpochModel.opened_at.desc())
                ).scalars()
            )
        return [
            EpochRecord(
                epoch_id=r.epoch_id,
                open_txid=r.open_txid,
                close_txid=r.close_txid,
                records_count=r.records_count,
                merkle_root=r.merkle_root,
                opened_at=r.opened_at,
                closed_at=r.closed_at,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich(self, row: SystemModel) -> SystemRead:
        """Build a SystemRead by querying aggregate stats for a system row."""
        with Session(self._engine) as s:
            total_epochs: int = s.scalar(
                select(func.count(EpochModel.id)).where(EpochModel.system_id == row.system_id)
            ) or 0
            total_records: int = s.scalar(
                select(func.sum(EpochModel.records_count)).where(
                    EpochModel.system_id == row.system_id
                )
            ) or 0
            last_epoch_at: datetime | None = s.scalar(
                select(func.max(EpochModel.opened_at)).where(
                    EpochModel.system_id == row.system_id
                )
            )
        return SystemRead(
            system_id=row.system_id,
            system_name=row.system_name,
            version=row.version,
            description=row.description,
            operator_name=row.operator_name,
            operator_contact=row.operator_contact,
            eu_ai_act_risk_level=row.eu_ai_act_risk_level,  # type: ignore[arg-type]
            eu_ai_act_article=row.eu_ai_act_article,
            deployment_context=row.deployment_context,
            registered_at=row.registered_at,
            total_epochs=total_epochs,
            total_records=total_records,
            last_epoch_at=last_epoch_at,
        )
