"""aria.dataset — Dataset anchoring for BSV.

Allows users to commit the hash of any dataset to BSV via an OP_RETURN
transaction, producing a tamper-evident, timestamped record of the dataset's
contents.

Supported dataset types (all serialised to bytes before hashing):
    - Arbitrary bytes or file paths (``anchor_bytes``)
    - JSON-serialisable objects  (``anchor_json``)
    - CSV files / text content   (``anchor_text``)

The on-chain payload follows the BRC-121 convention:

.. code-block:: json

    {
        "type": "DATASET_ANCHOR",
        "brc121_version": "1.0",
        "system_id": "<system>",
        "dataset_id": "<uuid4>",
        "content_hash": "sha256:<64hex>",
        "schema_hash": "sha256:<64hex>|null",
        "row_count": <int>|null,
        "column_names": [...]|null,
        "media_type": "<mime>",
        "anchored_at": "<ISO-8601 UTC>",
        "nonce": "<32hex>"
    }

The ``content_hash`` is SHA-256 of the raw bytes of the dataset.
The ``schema_hash`` is SHA-256 of the sorted canonical JSON of column names
when column information is available.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from aria.core.errors import ARIAError
from aria.core.hasher import canonical_json, hash_object

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BRC121_VERSION = "1.0"
_ANCHOR_TYPE = "DATASET_ANCHOR"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetAnchor:
    """Immutable record describing a dataset anchoring operation.

    Attributes:
        dataset_id:   UUID identifying this anchor (unique per call).
        system_id:    ARIA system that performed the anchoring.
        content_hash: ``sha256:<64hex>`` of the raw dataset bytes.
        schema_hash:  ``sha256:<64hex>`` of column schema, or ``None``.
        row_count:    Number of rows, or ``None`` if not available.
        column_names: Sorted list of column names, or ``None``.
        media_type:   MIME type of the dataset (e.g. ``"text/csv"``).
        anchored_at:  ISO-8601 UTC timestamp of the anchoring.
        txid:         BSV transaction ID, or empty string if broadcast failed.
        payload:      Full on-chain JSON payload (dict).
    """

    dataset_id: str
    system_id: str
    content_hash: str
    schema_hash: str | None
    row_count: int | None
    column_names: list[str] | None
    media_type: str
    anchored_at: str
    txid: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Core hashing helpers
# ---------------------------------------------------------------------------


def hash_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` of *data*."""
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def hash_columns(column_names: Sequence[str]) -> str:
    """Return ``sha256:<hex>`` of the sorted canonical JSON of column names.

    Sorting ensures that column order does not affect the schema hash.
    """
    sorted_cols = sorted(column_names)
    digest = hashlib.sha256(canonical_json(sorted_cols)).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# DatasetAnchor builder
# ---------------------------------------------------------------------------


def _build_payload(
    system_id: str,
    content_hash: str,
    schema_hash: str | None,
    row_count: int | None,
    column_names: list[str] | None,
    media_type: str,
    anchored_at: str,
    dataset_id: str,
    nonce: str,
) -> dict[str, Any]:
    return {
        "type": _ANCHOR_TYPE,
        "brc121_version": _BRC121_VERSION,
        "system_id": system_id,
        "dataset_id": dataset_id,
        "content_hash": content_hash,
        "schema_hash": schema_hash,
        "row_count": row_count,
        "column_names": column_names,
        "media_type": media_type,
        "anchored_at": anchored_at,
        "nonce": nonce,
    }


# ---------------------------------------------------------------------------
# DatasetAnchorer
# ---------------------------------------------------------------------------


class DatasetAnchorer:
    """Anchors dataset hashes to BSV.

    Args:
        system_id:  ARIA system identifier.
        wallet:     Wallet implementation with
                    :meth:`~aria.wallet.base.WalletInterface.sign_and_broadcast`.
                    Pass ``None`` to compute hashes only (no broadcast).

    Example::

        anchorer = DatasetAnchorer("my-system", wallet)

        # Anchor a CSV file
        anchor = await anchorer.anchor_file("train.csv", media_type="text/csv")
        print(anchor.txid, anchor.content_hash)
    """

    def __init__(self, system_id: str, wallet: Any | None = None) -> None:
        self._system_id = system_id
        self._wallet = wallet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def anchor_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        row_count: int | None = None,
        column_names: Sequence[str] | None = None,
    ) -> DatasetAnchor:
        """Anchor raw *data* bytes to BSV.

        Args:
            data:         Raw bytes to hash and anchor.
            media_type:   MIME type (default: ``application/octet-stream``).
            row_count:    Optional row count metadata.
            column_names: Optional column names for schema hash.

        Returns:
            :class:`DatasetAnchor` with all metadata and the broadcast txid.
        """
        return await self._anchor(
            data=data,
            media_type=media_type,
            row_count=row_count,
            column_names=list(column_names) if column_names else None,
        )

    async def anchor_file(
        self,
        path: str | Path,
        *,
        media_type: str | None = None,
        row_count: int | None = None,
        column_names: Sequence[str] | None = None,
    ) -> DatasetAnchor:
        """Anchor the contents of a file to BSV.

        Args:
            path:         Path to the file.
            media_type:   MIME type. If omitted, guessed from extension.
            row_count:    Optional row count metadata.
            column_names: Optional column names for schema hash.

        Returns:
            :class:`DatasetAnchor` describing the anchored file.

        Raises:
            ARIAError: If the file cannot be read.
        """
        p = Path(path)
        if not p.exists():
            raise ARIAError(f"Dataset file not found: {path}")
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise ARIAError(f"Cannot read dataset file {path}: {exc}") from exc

        mt = media_type or _guess_media_type(p.suffix)
        return await self.anchor_bytes(
            data,
            media_type=mt,
            row_count=row_count,
            column_names=column_names,
        )

    async def anchor_json(
        self,
        obj: Any,
        *,
        row_count: int | None = None,
        column_names: Sequence[str] | None = None,
    ) -> DatasetAnchor:
        """Anchor a JSON-serialisable object to BSV.

        The object is serialised to canonical JSON (keys sorted recursively)
        before hashing so that dict key order does not affect the hash.

        Args:
            obj:          JSON-serialisable object.
            row_count:    Optional row count metadata.
            column_names: Optional column names for schema hash.

        Returns:
            :class:`DatasetAnchor` describing the anchored object.
        """
        serialised = canonical_json(obj)
        return await self.anchor_bytes(
            serialised,
            media_type="application/json",
            row_count=row_count,
            column_names=column_names,
        )

    async def anchor_text(
        self,
        text: str,
        *,
        encoding: str = "utf-8",
        media_type: str = "text/plain",
        row_count: int | None = None,
        column_names: Sequence[str] | None = None,
    ) -> DatasetAnchor:
        """Anchor a text string or CSV content to BSV.

        Args:
            text:     Text content to anchor.
            encoding: Character encoding (default: ``utf-8``).
            media_type: MIME type (default: ``text/plain``).
            row_count:  Optional row count metadata.
            column_names: Optional column names for schema hash.

        Returns:
            :class:`DatasetAnchor` describing the anchored text.
        """
        data = text.encode(encoding)
        return await self.anchor_bytes(
            data,
            media_type=media_type,
            row_count=row_count,
            column_names=column_names,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _anchor(
        self,
        data: bytes,
        media_type: str,
        row_count: int | None,
        column_names: list[str] | None,
    ) -> DatasetAnchor:
        content_hash = hash_bytes(data)
        schema_hash = hash_columns(column_names) if column_names else None
        anchored_at = datetime.now(timezone.utc).isoformat()
        dataset_id = str(uuid.uuid4())
        nonce = secrets.token_hex(16)

        payload = _build_payload(
            system_id=self._system_id,
            content_hash=content_hash,
            schema_hash=schema_hash,
            row_count=row_count,
            column_names=column_names,
            media_type=media_type,
            anchored_at=anchored_at,
            dataset_id=dataset_id,
            nonce=nonce,
        )

        txid = ""
        if self._wallet is not None:
            try:
                txid = await self._wallet.sign_and_broadcast(payload)
            except Exception as exc:
                raise ARIAError(f"Failed to anchor dataset to BSV: {exc}") from exc

        return DatasetAnchor(
            dataset_id=dataset_id,
            system_id=self._system_id,
            content_hash=content_hash,
            schema_hash=schema_hash,
            row_count=row_count,
            column_names=column_names,
            media_type=media_type,
            anchored_at=anchored_at,
            txid=txid,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_dataset_anchor(data: bytes, anchor: DatasetAnchor) -> bool:
    """Verify that *data* matches the hash stored in *anchor*.

    Args:
        data:   The raw bytes claimed to correspond to this anchor.
        anchor: The :class:`DatasetAnchor` to verify against.

    Returns:
        ``True`` if ``SHA-256(data) == anchor.content_hash``.
    """
    return hash_bytes(data) == anchor.content_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIME_MAP: dict[str, str] = {
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".parquet": "application/vnd.apache.parquet",
    ".arrow": "application/vnd.apache.arrow.file",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".zip": "application/zip",
    ".gz": "application/gzip",
}


def _guess_media_type(suffix: str) -> str:
    return _MIME_MAP.get(suffix.lower(), "application/octet-stream")
