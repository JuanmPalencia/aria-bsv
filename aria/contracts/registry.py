"""
aria.contracts.registry — On-chain ARIA registry via OP_RETURN UTXOs.

The ARIARegistry stores system and model registration records as OP_RETURN
transactions on BSV.  Each entry is immutable once mined.

Protocol::

    OP_RETURN  ARIA-REG  <version:1>  <entry_type:1>  <payload:N>

Entry types:
    0x01 — SYSTEM    payload: <system_id:32> <pubkey_hash:20>
    0x02 — MODEL     payload: <model_id:32>  <model_hash:32>
    0x03 — OPERATOR  payload: <operator_id:32> <pubkey_hash:20>

System registration::

    OP_RETURN  ARIA-REG  0x01  0x01  <system_id:32>  <pubkey_hash160:20>

Model registration::

    OP_RETURN  ARIA-REG  0x01  0x02  <model_id:32>  <model_hash:32>

Querying the registry requires scanning BSV blockchain UTXOs with the magic
prefix.  The ``ARIARegistry.parse_entry()`` static method decodes any such
OP_RETURN output.

Usage::

    from aria.contracts.registry import ARIARegistry, RegistryEntryType

    registry = ARIARegistry()

    # Build OP_RETURN payload for system registration
    payload = registry.build_system_entry("my-system-id", pubkey_hex="04aabb...")
    # Broadcast this payload as OP_RETURN output via wallet

    # Parse an entry from raw OP_RETURN bytes
    entry = ARIARegistry.parse_entry(raw_bytes)
    if entry and entry.entry_type == RegistryEntryType.SYSTEM:
        print(entry.system_id)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------

class RegistryEntryType(IntEnum):
    SYSTEM = 0x01
    MODEL = 0x02
    OPERATOR = 0x03


_MAGIC = b"ARIA-REG"
_VERSION = 0x01


# ---------------------------------------------------------------------------
# Entry model
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """A decoded ARIA registry entry.

    Attributes:
        entry_type:    Type of registration (SYSTEM, MODEL, OPERATOR).
        system_id:     System identifier (for SYSTEM entries).
        model_id:      Model identifier (for MODEL entries).
        operator_id:   Operator identifier (for OPERATOR entries).
        pubkey_hash:   HASH160 of the registrant's public key (hex).
        model_hash:    SHA-256 model hash (for MODEL entries).
        txid:          BSV txid where this entry was recorded (if known).
        recorded_at:   Unix ms timestamp of local registration (off-chain index).
        raw_payload:   Original raw bytes, for verification.
    """
    entry_type: RegistryEntryType
    system_id: str = ""
    model_id: str = ""
    operator_id: str = ""
    pubkey_hash: str = ""         # hex, 40 chars
    model_hash: str = ""          # hex, 64 chars
    txid: str | None = None
    recorded_at: int = field(default_factory=lambda: int(time.time() * 1000))
    raw_payload: bytes = field(default=b"", repr=False)


# ---------------------------------------------------------------------------
# ARIARegistry
# ---------------------------------------------------------------------------

class ARIARegistry:
    """Construct and parse ARIA on-chain registry entries.

    This class has two roles:
    1. **Entry builder** — produces OP_RETURN byte payloads that can be
       embedded in a BSV transaction and broadcast via WalletInterface.
    2. **Local index** — maintains an in-memory cache of known entries,
       populated by scanning blockchain transactions.
    """

    def __init__(self) -> None:
        self._entries: list[RegistryEntry] = []

    # ------------------------------------------------------------------
    # Local index
    # ------------------------------------------------------------------

    def add_entry(self, entry: RegistryEntry) -> None:
        """Add a decoded entry to the local index."""
        self._entries.append(entry)

    def all_entries(self) -> list[RegistryEntry]:
        return list(self._entries)

    def get_system(self, system_id: str) -> RegistryEntry | None:
        for e in self._entries:
            if e.entry_type == RegistryEntryType.SYSTEM and e.system_id == system_id:
                return e
        return None

    def get_model(self, model_id: str) -> RegistryEntry | None:
        for e in self._entries:
            if e.entry_type == RegistryEntryType.MODEL and e.model_id == model_id:
                return e
        return None

    def get_operator(self, operator_id: str) -> RegistryEntry | None:
        for e in self._entries:
            if e.entry_type == RegistryEntryType.OPERATOR and e.operator_id == operator_id:
                return e
        return None

    def is_system_registered(self, system_id: str) -> bool:
        return self.get_system(system_id) is not None

    def is_model_registered(self, model_id: str) -> bool:
        return self.get_model(model_id) is not None

    # ------------------------------------------------------------------
    # Entry builders (produce raw OP_RETURN payload bytes)
    # ------------------------------------------------------------------

    def build_system_entry(self, system_id: str, pubkey_hex: str) -> bytes:
        """Build a SYSTEM registration OP_RETURN payload.

        Args:
            system_id:   System identifier string (up to 32 bytes UTF-8).
            pubkey_hex:  Compressed public key hex.  HASH160 is computed and stored.

        Returns:
            Raw bytes for the OP_RETURN data field.
        """
        pubkey_hash = _hash160_hex(pubkey_hex)
        payload = self._build_payload(
            RegistryEntryType.SYSTEM,
            _str_field(system_id),
            bytes.fromhex(pubkey_hash),
        )
        # Index locally
        entry = RegistryEntry(
            entry_type=RegistryEntryType.SYSTEM,
            system_id=system_id,
            pubkey_hash=pubkey_hash,
            raw_payload=payload,
        )
        self._entries.append(entry)
        return payload

    def build_model_entry(self, model_id: str, model_hash: str) -> bytes:
        """Build a MODEL registration OP_RETURN payload.

        Args:
            model_id:    Model identifier string (up to 32 bytes UTF-8).
            model_hash:  SHA-256 of the model weights file (64 hex chars).

        Returns:
            Raw bytes for the OP_RETURN data field.
        """
        mr_hex = model_hash.lstrip("sha256:").lstrip("0x").zfill(64)
        payload = self._build_payload(
            RegistryEntryType.MODEL,
            _str_field(model_id),
            bytes.fromhex(mr_hex),
        )
        entry = RegistryEntry(
            entry_type=RegistryEntryType.MODEL,
            model_id=model_id,
            model_hash=mr_hex,
            raw_payload=payload,
        )
        self._entries.append(entry)
        return payload

    def build_operator_entry(self, operator_id: str, pubkey_hex: str) -> bytes:
        """Build an OPERATOR registration OP_RETURN payload.

        Args:
            operator_id: Operator identifier string (up to 32 bytes UTF-8).
            pubkey_hex:  Operator's compressed public key hex.

        Returns:
            Raw bytes for the OP_RETURN data field.
        """
        pubkey_hash = _hash160_hex(pubkey_hex)
        payload = self._build_payload(
            RegistryEntryType.OPERATOR,
            _str_field(operator_id),
            bytes.fromhex(pubkey_hash),
        )
        entry = RegistryEntry(
            entry_type=RegistryEntryType.OPERATOR,
            operator_id=operator_id,
            pubkey_hash=pubkey_hash,
            raw_payload=payload,
        )
        self._entries.append(entry)
        return payload

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_entry(raw: bytes, txid: str | None = None) -> RegistryEntry | None:
        """Attempt to parse raw OP_RETURN bytes into a RegistryEntry.

        Returns None if the bytes do not match the ARIA-REG protocol.
        """
        if len(raw) < 11 or raw[:8] != _MAGIC:
            return None
        if raw[8] != _VERSION:
            return None

        entry_type_byte = raw[9]
        try:
            entry_type = RegistryEntryType(entry_type_byte)
        except ValueError:
            return None

        body = raw[10:]

        if entry_type == RegistryEntryType.SYSTEM:
            if len(body) < 52:
                return None
            system_id = body[:32].rstrip(b"\x00").decode(errors="replace")
            pubkey_hash = body[32:52].hex()
            return RegistryEntry(
                entry_type=entry_type,
                system_id=system_id,
                pubkey_hash=pubkey_hash,
                txid=txid,
                raw_payload=raw,
            )

        elif entry_type == RegistryEntryType.MODEL:
            if len(body) < 64:
                return None
            model_id = body[:32].rstrip(b"\x00").decode(errors="replace")
            model_hash = body[32:64].hex()
            return RegistryEntry(
                entry_type=entry_type,
                model_id=model_id,
                model_hash=model_hash,
                txid=txid,
                raw_payload=raw,
            )

        elif entry_type == RegistryEntryType.OPERATOR:
            if len(body) < 52:
                return None
            operator_id = body[:32].rstrip(b"\x00").decode(errors="replace")
            pubkey_hash = body[32:52].hex()
            return RegistryEntry(
                entry_type=entry_type,
                operator_id=operator_id,
                pubkey_hash=pubkey_hash,
                txid=txid,
                raw_payload=raw,
            )

        return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(
        entry_type: RegistryEntryType,
        *fields: bytes,
    ) -> bytes:
        return _MAGIC + bytes([_VERSION, entry_type.value]) + b"".join(fields)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_field(s: str, size: int = 32) -> bytes:
    """Encode a string as a fixed-width zero-padded byte field."""
    return s.encode()[:size].ljust(size, b"\x00")


def _hash160_hex(pubkey_hex: str) -> str:
    """RIPEMD160(SHA256(pubkey_bytes)), returned as hex string."""
    pub = bytes.fromhex(pubkey_hex)
    sha = hashlib.sha256(pub).digest()
    h = hashlib.new("ripemd160", sha).digest()
    return h.hex()
