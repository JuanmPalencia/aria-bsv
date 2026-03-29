"""
aria.mcp_server — MCP (Model Context Protocol) server for ARIA auditing.

Exposes ARIA inference auditing as MCP tools so any LLM (Claude, etc.) can
self-audit its own inference calls via the standard MCP protocol.

Tools provided
--------------
aria_record         Record one inference to the current epoch.
aria_status         Return current auditor statistics.
aria_verify_local   Retrieve a stored record receipt by ID.
aria_close_epoch    Close the current epoch and return the Merkle root.
aria_hash_text      Hash a text string with SHA-256 (no recording).

Usage (standalone server)::

    python -m aria.mcp_server

Usage (programmatic)::

    from aria.mcp_server import mcp
    mcp.run()

Environment variables
---------------------
ARIA_SYSTEM_ID   System identifier (default: "aria-mcp").
ARIA_DB_PATH     SQLite database path (default: "aria_mcp.db").
ARIA_BSV_WIF     WIF private key for BSV on-chain anchoring (optional).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from aria.quick import ARIAQuick  # imported at module level so tests can patch it

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard MCP import
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _mcp_import_error:
    raise ImportError(
        "The 'mcp' package is required to use aria.mcp_server. "
        "Install it with: pip install 'mcp>=1.0'"
    ) from _mcp_import_error

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "aria-bsv",
    description=(
        "ARIA — Auditable Real-time Inference Architecture. "
        "Provides cryptographic accountability for AI inference calls on BSV blockchain. "
        "Use these tools to record, verify, and close audit epochs for your LLM outputs."
    ),
)

# ---------------------------------------------------------------------------
# Lazy ARIAQuick initialisation
# ---------------------------------------------------------------------------

_aria_instance: Any = None  # ARIAQuick | None


def _get_aria():
    """Return the module-level ARIAQuick instance, initialising it on first call."""
    global _aria_instance
    if _aria_instance is None:
        system_id = os.environ.get("ARIA_SYSTEM_ID", "aria-mcp")
        db_path = os.environ.get("ARIA_DB_PATH", "aria_mcp.db")
        bsv_wif = os.environ.get("ARIA_BSV_WIF")  # may be None — local-only mode

        _aria_instance = ARIAQuick(
            system_id=system_id,
            db_path=db_path,
            bsv_wif=bsv_wif,
            watchdog=False,
            compliance=False,  # keep latency low for interactive MCP calls
        )
        _aria_instance.start()
        _log.info(
            "ARIAQuick initialised: system_id=%r db_path=%r bsv=%s",
            system_id,
            db_path,
            "configured" if bsv_wif else "local-only",
        )
    return _aria_instance


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def aria_record(
    model_id: str,
    input_text: str,
    output_text: str,
    confidence: float = 1.0,
) -> dict:
    """Record one AI inference to the current ARIA audit epoch.

    This tool hashes the input and output texts before storing them —
    raw text never leaves your environment.  The returned record_id can
    be used later with aria_verify_local to retrieve a receipt.

    Args:
        model_id:    Identifier for the model (e.g. "claude-opus-4-6").
        input_text:  The prompt or input sent to the model.
        output_text: The model's response.
        confidence:  Model confidence score in [0.0, 1.0] (default 1.0).

    Returns:
        dict with keys: record_id, epoch_id, record_hash.
    """
    try:
        aria = _get_aria()
        record_id = aria.record(
            model_id=model_id,
            input_data={"text": input_text},
            output_data={"text": output_text},
            confidence=confidence,
        )
        epoch_id = aria.current_epoch_id or ""

        # Compute the record hash from the record stored in the DB
        record_hash = ""
        try:
            record = aria.storage.get_record(record_id)
            if record is not None:
                record_hash = record.hash()
        except Exception as exc:  # noqa: BLE001
            _log.debug("aria_record: could not compute record_hash: %s", exc)

        return {
            "record_id": record_id,
            "epoch_id": epoch_id,
            "record_hash": record_hash,
        }
    except Exception as exc:  # noqa: BLE001
        _log.error("aria_record failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def aria_status() -> dict:
    """Return current ARIA auditor statistics.

    Returns:
        dict with keys: system_id, epoch_id, record_count, db_path.
    """
    try:
        aria = _get_aria()
        epoch_id = aria.current_epoch_id or ""
        record_count = 0
        try:
            if epoch_id:
                records = aria.storage.list_records_by_epoch(epoch_id)
                record_count = len(records)
        except Exception as exc:  # noqa: BLE001
            _log.debug("aria_status: could not count records: %s", exc)

        return {
            "system_id": aria.system_id,
            "epoch_id": epoch_id,
            "record_count": record_count,
            "db_path": aria._db_path,
        }
    except Exception as exc:  # noqa: BLE001
        _log.error("aria_status failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def aria_verify_local(record_id: str) -> dict:
    """Retrieve a stored inference receipt by record ID.

    Use this to verify that a previously recorded inference was correctly
    persisted and to retrieve its cryptographic hash.

    Args:
        record_id: The record_id returned by a previous aria_record call.

    Returns:
        dict with keys: record_id, record_hash, epoch_id, model_id.
        On failure: dict with key: error.
    """
    try:
        aria = _get_aria()
        record = aria.storage.get_record(record_id)
        if record is None:
            return {"error": f"record not found: {record_id}"}

        return {
            "record_id": record.record_id,
            "record_hash": record.hash(),
            "epoch_id": record.epoch_id,
            "model_id": record.model_id,
        }
    except Exception as exc:  # noqa: BLE001
        _log.error("aria_verify_local failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def aria_close_epoch() -> dict:
    """Close the current audit epoch and compute the Merkle root.

    Seals the current batch of inference records into an immutable Merkle
    tree and (if a BSV key was configured) broadcasts the EPOCH_CLOSE
    transaction to the BSV blockchain.

    Returns:
        dict with keys: epoch_id, merkle_root, records_count, anchored.
        On failure: dict with key: error.
    """
    try:
        aria = _get_aria()
        summary = aria.close()
        return {
            "epoch_id": summary.epoch_id,
            "merkle_root": summary.merkle_root,
            "records_count": summary.records_count,
            "anchored": summary.anchored,
        }
    except Exception as exc:  # noqa: BLE001
        _log.error("aria_close_epoch failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def aria_hash_text(text: str) -> dict:
    """Hash a text string with SHA-256.

    Useful for LLMs to verify their own output hash without recording it.
    The returned hash uses the same format as all ARIA record hashes
    ("sha256:<64 hex chars>") so it can be compared directly to stored
    record_hash values.

    Args:
        text: Any UTF-8 string to hash.

    Returns:
        dict with keys: hash ("sha256:<hex>"), length (character count of input).
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "hash": f"sha256:{digest}",
        "length": len(text),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
