"""
aria.lineage — Data lineage tracking for AI model versions and epochs.

Records which datasets, model checkpoints, and hyperparameters produced
each epoch. Enables full traceability: "which training run produced this
inference epoch, and what data was it trained on?"

Usage::

    from aria.lineage import LineageTracker, ModelVersion, DatasetRef

    tracker = LineageTracker(storage)

    # Register a model version
    mv = ModelVersion(
        model_id="sentiment-v2",
        version="2.1.0",
        training_epochs=5,
        datasets=[DatasetRef("sst2", "1.0", 67349)],
        hyperparams={"lr": 2e-5, "batch_size": 32},
    )
    tracker.register_model(mv)

    # Link an inference epoch to the model version
    tracker.link_epoch("epoch-prod-123", "sentiment-v2", "2.1.0")

    # Retrieve full lineage
    lineage = tracker.get_lineage("epoch-prod-123")
    print(lineage)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DatasetRef:
    """Reference to a training dataset."""
    name:       str
    version:    str = ""
    num_rows:   int = 0
    source_uri: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetRef":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    """Metadata about a trained model version."""
    model_id:         str
    version:          str
    training_epochs:  int = 0
    datasets:         list[DatasetRef] = field(default_factory=list)
    hyperparams:      dict[str, Any] = field(default_factory=dict)
    base_model:       str = ""          # e.g. "bert-base-uncased"
    framework:        str = ""          # e.g. "pytorch", "tensorflow"
    created_at:       str = ""          # ISO 8601

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datasets"] = [ds.to_dict() for ds in self.datasets]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        datasets = [DatasetRef.from_dict(ds) for ds in d.pop("datasets", [])]
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj = cls(**fields)
        obj.datasets = datasets
        return obj


@dataclass
class EpochLineage:
    """Full lineage record linking an inference epoch to its model version."""
    epoch_id:       str
    model_id:       str
    model_version:  str
    model_meta:     ModelVersion | None = None
    linked_at:      str = ""

    def __post_init__(self):
        if not self.linked_at:
            self.linked_at = datetime.now(timezone.utc).isoformat()

    def __str__(self) -> str:
        lines = [
            f"EpochLineage: epoch={self.epoch_id}",
            f"  model={self.model_id}  version={self.model_version}",
        ]
        if self.model_meta:
            m = self.model_meta
            if m.base_model:
                lines.append(f"  base={m.base_model}")
            if m.datasets:
                ds_names = ", ".join(d.name for d in m.datasets)
                lines.append(f"  datasets=[{ds_names}]")
            if m.hyperparams:
                lines.append(f"  hyperparams={m.hyperparams}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LineageTracker
# ---------------------------------------------------------------------------

class LineageTracker:
    """Tracks model version lineage for inference epochs.

    Uses an in-memory registry supplemented by optional metadata stored in
    epoch metadata via the storage backend.

    Args:
        storage:  StorageInterface (used for epoch metadata enrichment).
    """

    def __init__(self, storage: "StorageInterface | None" = None) -> None:
        self._storage = storage
        self._models:  dict[str, dict[str, ModelVersion]] = {}  # model_id → version → mv
        self._links:   dict[str, EpochLineage] = {}             # epoch_id → lineage

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(self, model_version: ModelVersion) -> None:
        """Register a model version in the registry."""
        mid = model_version.model_id
        if mid not in self._models:
            self._models[mid] = {}
        self._models[mid][model_version.version] = model_version
        _log.debug("Registered model %s v%s", mid, model_version.version)

    def get_model(self, model_id: str, version: str) -> ModelVersion | None:
        """Look up a registered model version."""
        return self._models.get(model_id, {}).get(version)

    def list_versions(self, model_id: str) -> list[str]:
        """List all registered versions for a model."""
        return list(self._models.get(model_id, {}).keys())

    # ------------------------------------------------------------------
    # Epoch linking
    # ------------------------------------------------------------------

    def link_epoch(self, epoch_id: str, model_id: str, version: str) -> EpochLineage:
        """Link an inference epoch to a specific model version."""
        meta = self.get_model(model_id, version)
        lineage = EpochLineage(
            epoch_id=epoch_id,
            model_id=model_id,
            model_version=version,
            model_meta=meta,
        )
        self._links[epoch_id] = lineage
        return lineage

    def get_lineage(self, epoch_id: str) -> EpochLineage | None:
        """Retrieve the lineage for an epoch."""
        return self._links.get(epoch_id)

    def epochs_for_model(self, model_id: str, version: str | None = None) -> list[str]:
        """Return all epoch IDs linked to a model (optionally filtered by version)."""
        result = []
        for eid, lineage in self._links.items():
            if lineage.model_id == model_id:
                if version is None or lineage.model_version == version:
                    result.append(eid)
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def export_lineage(self, epoch_id: str) -> dict:
        """Export lineage as a plain dict (JSON-serialisable)."""
        lineage = self.get_lineage(epoch_id)
        if lineage is None:
            return {}
        result = {
            "epoch_id":      lineage.epoch_id,
            "model_id":      lineage.model_id,
            "model_version": lineage.model_version,
            "linked_at":     lineage.linked_at,
        }
        if lineage.model_meta:
            result["model_meta"] = lineage.model_meta.to_dict()
        return result

    def import_lineage(self, data: dict) -> EpochLineage:
        """Restore a lineage from a dict (as produced by export_lineage)."""
        meta = None
        if "model_meta" in data:
            meta = ModelVersion.from_dict(data["model_meta"])
        lineage = EpochLineage(
            epoch_id=data["epoch_id"],
            model_id=data["model_id"],
            model_version=data["model_version"],
            model_meta=meta,
            linked_at=data.get("linked_at", ""),
        )
        self._links[lineage.epoch_id] = lineage
        return lineage
