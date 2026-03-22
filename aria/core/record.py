"""
aria/core/record.py

AuditRecord — the canonical type for a single AI inference event.

An AuditRecord captures everything needed to include one inference in an
ARIA epoch's Merkle tree. Raw inputs and outputs never appear here; only
their SHA-256 hashes (computed by the caller before constructing the record).

Schema defined in BRC-120 §3.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from aria.core.errors import ARIASerializationError
from aria.core.hasher import hash_object

ARIA_VERSION = "1.0"
_HASH_PREFIX = "sha256:"


def _validate_hash(value: str, field_name: str) -> None:
    """Raise ARIASerializationError if *value* is not a valid ARIA hash string."""
    if not isinstance(value, str) or not value.startswith(_HASH_PREFIX):
        raise ARIASerializationError(
            f"AuditRecord.{field_name} must start with 'sha256:', got {value!r}"
        )
    hex_part = value[len(_HASH_PREFIX):]
    if len(hex_part) != 64:
        raise ARIASerializationError(
            f"AuditRecord.{field_name} must be 'sha256:' followed by "
            f"64 hex characters, got {len(hex_part)} characters"
        )
    try:
        bytes.fromhex(hex_part)
    except ValueError as exc:
        raise ARIASerializationError(
            f"AuditRecord.{field_name} contains invalid hex: {exc}"
        ) from exc


@dataclass
class AuditRecord:
    """
    A single AI inference event within an ARIA epoch.

    Instances should be treated as immutable after creation. The *record_id*
    and *aria_version* fields are computed automatically and cannot be
    provided by the caller.

    Args:
        epoch_id:    ID of the epoch this record belongs to.
        model_id:    Identifier of the model that performed the inference.
                     Must match a key in the parent epoch's *model_hashes*.
        input_hash:  SHA-256 hash of the canonical, PII-stripped input.
                     Format: "sha256:{64 hex chars}".
        output_hash: SHA-256 hash of the canonical output.
                     Format: "sha256:{64 hex chars}".
        sequence:    Zero-based position of this record within the epoch.
        confidence:  Model confidence score in [0.0, 1.0], or None.
        latency_ms:  Inference duration in milliseconds (>= 0).
        metadata:    Optional free-form dict for domain-specific fields.
    """

    epoch_id:    str
    model_id:    str
    input_hash:  str
    output_hash: str
    sequence:    int
    confidence:  float | None        = None
    latency_ms:  int                 = 0
    metadata:    dict[str, Any]      = field(default_factory=dict)

    # Auto-computed — not accepted as constructor arguments.
    aria_version: str = field(default=ARIA_VERSION, init=False)
    record_id:    str = field(default="", init=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "record_id", f"rec_{self.epoch_id}_{self.sequence:06d}")

    def _validate(self) -> None:
        if not self.epoch_id:
            raise ARIASerializationError("AuditRecord.epoch_id cannot be empty")
        if not self.model_id:
            raise ARIASerializationError("AuditRecord.model_id cannot be empty")

        _validate_hash(self.input_hash, "input_hash")
        _validate_hash(self.output_hash, "output_hash")

        if self.sequence < 0:
            raise ARIASerializationError(
                f"AuditRecord.sequence must be >= 0, got {self.sequence}"
            )
        if self.latency_ms < 0:
            raise ARIASerializationError(
                f"AuditRecord.latency_ms must be >= 0, got {self.latency_ms}"
            )
        if self.confidence is not None:
            if math.isnan(self.confidence) or math.isinf(self.confidence):
                raise ARIASerializationError(
                    "AuditRecord.confidence must not be NaN or Infinity"
                )
            if not (0.0 <= self.confidence <= 1.0):
                raise ARIASerializationError(
                    f"AuditRecord.confidence must be in [0.0, 1.0], got {self.confidence}"
                )

    def to_canonical_dict(self) -> dict[str, Any]:
        """
        Return a dict representation of this record suitable for hashing.

        Keys are in alphabetical order (canonical_json sorts them anyway,
        but explicit ordering here aids readability and review).
        """
        return {
            "aria_version": self.aria_version,
            "confidence":   self.confidence,
            "epoch_id":     self.epoch_id,
            "input_hash":   self.input_hash,
            "latency_ms":   self.latency_ms,
            "metadata":     self.metadata,
            "model_id":     self.model_id,
            "output_hash":  self.output_hash,
            "record_id":    self.record_id,
            "sequence":     self.sequence,
        }

    def hash(self) -> str:
        """
        Return the SHA-256 hash of this record's canonical representation.

        This value is used as the leaf in the epoch's Merkle tree.

        Returns:
            str: "sha256:{64 hex chars}"
        """
        return hash_object(self.to_canonical_dict())
