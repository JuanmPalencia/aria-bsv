"""
aria/core/hasher.py

Canonical JSON serialisation and SHA-256 hashing.

This is the ONLY module that produces hashes in ARIA. All other modules
that need to hash something must call functions from here. Never call
hashlib directly outside this module.

Canonical JSON rules (ARIA spec §4.6 / BRC-120):
  1. No whitespace (no spaces, no line breaks).
  2. Object keys sorted alphabetically at every nesting level.
  3. Floats: Python's default repr — shortest round-trip, max 17 sig digits.
  4. null → JSON null.
  5. NaN and Infinity are FORBIDDEN — raises ARIASerializationError.
  6. Arrays: order preserved (arrays are never sorted).
  7. Encoding: UTF-8.

All hash strings returned by this module use the format "sha256:{64 hex chars}".
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from aria.core.errors import ARIASerializationError

_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB — for streaming file hashes


def canonical_json(obj: Any) -> bytes:
    """
    Serialise *obj* to canonical JSON bytes (UTF-8).

    The output is deterministic: the same logical object always produces
    identical bytes regardless of the order keys were inserted in Python
    dicts, or the platform running the code.

    Args:
        obj: Any JSON-serialisable Python object.

    Returns:
        UTF-8 encoded bytes of the canonical JSON representation.

    Raises:
        ARIASerializationError: if *obj* contains NaN, Infinity, or any
            type that cannot be serialised to JSON (e.g. set, bytes).
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ARIASerializationError(
            f"Object contains a non-serialisable value: {exc}. "
            "NaN and Infinity are not allowed in ARIA audit records."
        ) from exc
    except TypeError as exc:
        raise ARIASerializationError(
            f"Object contains a non-serialisable type: {exc}"
        ) from exc


def hash_object(obj: Any) -> str:
    """
    Return the SHA-256 hex digest of *canonical_json(obj)*.

    Args:
        obj: Any JSON-serialisable Python object.

    Returns:
        str in the format "sha256:{64 lowercase hex chars}".

    Raises:
        ARIASerializationError: if *obj* is not canonically serialisable.
    """
    data = canonical_json(obj)
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def hash_file(path: str | Path) -> str:
    """
    Return the SHA-256 hex digest of the raw binary contents of a file.

    Used to compute *model_hash* from a serialised model file (e.g. a
    joblib-serialised scikit-learn model or an ONNX file). Reads in
    8 MB chunks so large model files do not exhaust memory.

    Args:
        path: Path to the file to hash.

    Returns:
        str in the format "sha256:{64 lowercase hex chars}".

    Raises:
        FileNotFoundError: if *path* does not exist.
        ARIASerializationError: if the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK_SIZE):
                hasher.update(chunk)
    except OSError as exc:
        raise ARIASerializationError(
            f"Cannot read file for hashing: {path}: {exc}"
        ) from exc

    return f"sha256:{hasher.hexdigest()}"
