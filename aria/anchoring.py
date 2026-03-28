"""
aria.anchoring — Multi-chain OP_RETURN anchoring for ARIA epochs.

Anchors the ARIA epoch close commitment to an additional chain (Bitcoin BTC)
alongside the primary BSV broadcast.  Both chains record the same canonical
payload via OP_RETURN, creating independent proof-of-existence at the same
block height (approximately).

Protocol payload (identical on both chains)::

    OP_RETURN  ARIA  <version:1>  <epoch_id:16>  <merkle_root:32>  <records_count:4>

Where:
  - ``ARIA``        — 4-byte magic prefix
  - ``<version>``   — 0x01 = BRC-121 v1
  - ``<epoch_id>``  — first 16 bytes of epoch_id UTF-8 (zero-padded)
  - ``<merkle_root>`` — 32 raw bytes of the Merkle root hash
  - ``<records_count>`` — 4-byte big-endian record count

Usage::

    from aria.anchoring import AnchorPayload, build_op_return_payload
    from aria.anchoring import MultiChainAnchor, BTCBroadcaster

    payload = AnchorPayload.from_epoch_close(epoch_id, merkle_root, records_count)
    raw = build_op_return_payload(payload)

    anchor = MultiChainAnchor(
        btc_broadcaster=BTCBroadcaster(api_url="https://blockstream.info/api"),
        secondary_chains=["btc"],
    )
    results = await anchor.broadcast_all(payload, btc_funding_utxo=utxo)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Minimal BTC raw-transaction builder / signer (secp256k1, P2PKH, SIGHASH_ALL)
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    if n < 0xFD:
        return bytes([n])
    elif n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    elif n <= 0xFFFFFFFF:
        return b"\xfe" + struct.pack("<I", n)
    return b"\xff" + struct.pack("<Q", n)


def _hash256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _hash160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()


def _p2pkh_script(pubkey_bytes: bytes) -> bytes:
    h = _hash160(pubkey_bytes)
    return bytes([0x76, 0xA9, 0x14]) + h + bytes([0x88, 0xAC])


def _base58_decode(s: str) -> bytes:
    ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for c in s.encode():
        n = n * 58 + ALPHABET.index(c)
    result = n.to_bytes(38, "big").lstrip(b"\x00")
    # restore leading zero bytes
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + result


def _decode_wif(wif: str) -> tuple[bytes, bool]:
    """Return (privkey_32_bytes, compressed)."""
    raw = _base58_decode(wif)
    # raw = version(1) + key(32) [+ 0x01 if compressed] + checksum(4)
    payload = raw[:-4]
    compressed = len(payload) == 34 and payload[-1] == 0x01
    privkey = payload[1:33]
    return privkey, compressed


def _privkey_to_pubkey(privkey: bytes, compressed: bool = True) -> bytes:
    try:
        import coincurve
        return coincurve.PublicKey.from_valid_secret(privkey).format(compressed=compressed)
    except ImportError:
        pass
    # Fallback: use cryptography library
    from cryptography.hazmat.primitives.asymmetric.ec import (
        SECP256K1, EllipticCurvePrivateKey, derive_private_key,
    )
    from cryptography.hazmat.backends import default_backend
    priv_int = int.from_bytes(privkey, "big")
    key = derive_private_key(priv_int, SECP256K1(), default_backend())
    pub = key.public_key().public_bytes(
        encoding=__import__("cryptography").hazmat.primitives.serialization.Encoding.X962,
        format=(__import__("cryptography").hazmat.primitives.serialization.PublicFormat.CompressedPoint
                if compressed else
                __import__("cryptography").hazmat.primitives.serialization.PublicFormat.UncompressedPoint),
    )
    return pub


def _sign_hash(privkey: bytes, msg_hash: bytes) -> bytes:
    """Return DER-encoded ECDSA signature (secp256k1)."""
    try:
        import coincurve
        sig = coincurve.PrivateKey(privkey).sign(msg_hash, hasher=None)
        return sig  # coincurve returns DER by default
    except ImportError:
        pass
    from cryptography.hazmat.primitives.asymmetric.ec import (
        SECP256K1, ECDSA, derive_private_key,
    )
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.backends import default_backend
    priv_int = int.from_bytes(privkey, "big")
    key = derive_private_key(priv_int, SECP256K1(), default_backend())
    # cryptography signs with hashing; we pass pre-hashed and use Prehashed
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.primitives.asymmetric import ec
    sig_bytes = key.sign(msg_hash, ec.ECDSA(__import__("cryptography").hazmat.primitives.hashes.Prehashed()))
    return sig_bytes


def _build_btc_tx(
    utxo_txid: str,
    utxo_vout: int,
    utxo_satoshis: int,
    op_return_data: bytes,
    privkey: bytes,
    compressed: bool,
    fee_sat: int = 2000,
) -> str:
    """Build and sign a BTC P2PKH → OP_RETURN + change transaction.

    Returns the signed raw transaction as a hex string.
    """
    pubkey = _privkey_to_pubkey(privkey, compressed)
    locking_script = _p2pkh_script(pubkey)
    change_sat = utxo_satoshis - fee_sat
    op_return_script = _build_op_return_script(op_return_data)

    def _serialize_tx(script_sig: bytes) -> bytes:
        # version
        tx = struct.pack("<I", 1)
        # inputs
        tx += _varint(1)
        tx += bytes.fromhex(utxo_txid)[::-1]   # txid LE
        tx += struct.pack("<I", utxo_vout)
        tx += _varint(len(script_sig)) + script_sig
        tx += b"\xff\xff\xff\xff"               # sequence
        # outputs
        n_outputs = 2 if change_sat > 546 else 1
        tx += _varint(n_outputs)
        # OP_RETURN (0 satoshis)
        tx += struct.pack("<Q", 0)
        tx += _varint(len(op_return_script)) + op_return_script
        # change (P2PKH back to sender)
        if change_sat > 546:
            tx += struct.pack("<Q", change_sat)
            tx += _varint(len(locking_script)) + locking_script
        # locktime
        tx += b"\x00\x00\x00\x00"
        return tx

    # Sighash preimage: serialize with locking_script as scriptSig, append SIGHASH_ALL
    preimage = _serialize_tx(locking_script) + struct.pack("<I", 1)
    sighash = _hash256(preimage)
    der_sig = _sign_hash(privkey, sighash)
    script_sig = (
        bytes([len(der_sig) + 1]) + der_sig + b"\x01"  # sig + SIGHASH_ALL
        + bytes([len(pubkey)]) + pubkey
    )
    return _serialize_tx(script_sig).hex()


# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------

_MAGIC = b"ARIA"
_VERSION = 0x01


@dataclass
class AnchorPayload:
    """The canonical data committed to each chain via OP_RETURN.

    Attributes:
        epoch_id:      Full epoch UUID string (stored; first 16 bytes go on-chain).
        merkle_root:   Hex-encoded Merkle root (64 hex chars = 32 bytes).
        records_count: Number of records in the epoch.
        system_id:     System identifier (metadata only, not on-chain).
    """
    epoch_id: str
    merkle_root: str          # hex string, 64 chars
    records_count: int
    system_id: str = ""

    @classmethod
    def from_epoch_close(
        cls,
        epoch_id: str,
        merkle_root: str,
        records_count: int,
        system_id: str = "",
    ) -> "AnchorPayload":
        return cls(
            epoch_id=epoch_id,
            merkle_root=merkle_root,
            records_count=records_count,
            system_id=system_id,
        )

    def canonical_bytes(self) -> bytes:
        """Return the full OP_RETURN data bytes (72 bytes total)."""
        return build_op_return_payload(self)

    def digest(self) -> str:
        """SHA-256 of canonical_bytes() — used for cross-chain verification."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_op_return_payload(payload: AnchorPayload) -> bytes:
    """Encode the AnchorPayload into the on-chain OP_RETURN byte string.

    Layout (72 bytes):
        4  bytes — magic "ARIA"
        1  byte  — version 0x01
        16 bytes — epoch_id prefix (UTF-8, zero-padded)
        32 bytes — merkle_root raw bytes
        4  bytes — records_count (big-endian uint32)
        15 bytes — zero padding (reserved for future use)

    Returns:
        72-byte bytes object ready to embed as OP_RETURN data.
    """
    epoch_bytes = payload.epoch_id.encode()[:16].ljust(16, b"\x00")

    # Merkle root: accept either hex string or raw bytes
    if isinstance(payload.merkle_root, str):
        mr = payload.merkle_root.lstrip("sha256:").lstrip("0x")
        # Pad to 64 hex chars if shorter (e.g., empty string)
        mr_bytes = bytes.fromhex(mr.zfill(64))[:32]
    else:
        mr_bytes = bytes(payload.merkle_root)[:32]

    count_bytes = struct.pack(">I", payload.records_count)
    padding = b"\x00" * 15

    return _MAGIC + bytes([_VERSION]) + epoch_bytes + mr_bytes + count_bytes + padding


def decode_op_return_payload(data: bytes) -> AnchorPayload | None:
    """Attempt to decode an OP_RETURN payload.  Returns None if not ARIA-formatted."""
    if len(data) < 53 or data[:4] != _MAGIC:
        return None
    if data[4] != _VERSION:
        return None
    epoch_id_bytes = data[5:21].rstrip(b"\x00")
    mr_bytes = data[21:53]
    count_bytes = data[53:57]

    epoch_id = epoch_id_bytes.decode(errors="replace")
    merkle_root = mr_bytes.hex()
    records_count = struct.unpack(">I", count_bytes)[0]
    return AnchorPayload(epoch_id=epoch_id, merkle_root=merkle_root, records_count=records_count)


# ---------------------------------------------------------------------------
# BTC broadcaster (Blockstream.info / Mempool.space compatible)
# ---------------------------------------------------------------------------

@dataclass
class AnchorResult:
    """Result of an OP_RETURN broadcast attempt."""
    chain: str                  # "btc", "bsv", etc.
    txid: str | None = None
    success: bool = False
    error: str | None = None


class BTCBroadcaster:
    """Broadcast OP_RETURN transactions to Bitcoin via HTTP APIs.

    Supports Blockstream.info and Mempool.space Esplora-compatible API formats.

    When a WIF signing key is provided, ``broadcast()`` fetches UTXOs from the
    Esplora API, builds a P2PKH → OP_RETURN + change transaction, signs it
    with secp256k1 (via coincurve or cryptography), and broadcasts it — all
    without requiring external Bitcoin libraries.

    When no WIF is provided, ``broadcast()`` returns an error and callers must
    construct the transaction externally and use ``broadcast_signed()``.

    Args:
        api_url:    Base URL of an Esplora-compatible API.  Defaults to
                    ``"https://blockstream.info/api"`` (mainnet).
        wif:        BTC private key in WIF format.  Required for ``broadcast()``.
        fee_sat:    Miner fee in satoshis (default 2000).
        network:    ``"mainnet"`` or ``"testnet"`` (for logging only).

    Example (with key)::

        broadcaster = BTCBroadcaster(wif=os.environ["BTC_WIF"])
        result = await broadcaster.broadcast(op_return_data)

    Example (pre-signed)::

        broadcaster = BTCBroadcaster()
        result = await broadcaster.broadcast_signed(signed_raw_hex)
    """

    def __init__(
        self,
        api_url: str = "https://blockstream.info/api",
        wif: str | None = None,
        fee_sat: int = 2000,
        network: str = "mainnet",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._fee_sat = fee_sat
        self._network = network
        self._privkey: bytes | None = None
        self._compressed: bool = True
        if wif:
            try:
                self._privkey, self._compressed = _decode_wif(wif)
            except Exception as exc:
                raise ValueError("invalid key material") from exc

    async def _fetch_utxos(self, address: str) -> list[dict]:
        """Fetch confirmed UTXOs for *address* from Esplora API."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self._api_url}/address/{address}/utxo")
            resp.raise_for_status()
            return [u for u in resp.json() if u.get("status", {}).get("confirmed", False)]

    def _derive_address(self) -> str:
        """Derive P2PKH Bitcoin address from the configured WIF key."""
        if self._privkey is None:
            raise ValueError("No WIF configured")
        pubkey = _privkey_to_pubkey(self._privkey, self._compressed)
        h = _hash160(pubkey)
        # Version byte: 0x00 mainnet, 0x6F testnet
        version = b"\x00" if self._network == "mainnet" else b"\x6F"
        payload = version + h
        checksum = _hash256(payload)[:4]
        raw = payload + checksum
        # Base58 encode
        ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        n = int.from_bytes(raw, "big")
        result = ""
        while n:
            n, r = divmod(n, 58)
            result = ALPHABET[r] + result
        return "1" * (len(raw) - len(raw.lstrip(b"\x00"))) + result

    async def broadcast(self, op_return_data: bytes) -> AnchorResult:
        """Build, sign and broadcast an OP_RETURN transaction to Bitcoin.

        Requires a WIF key configured at construction time.  Fetches UTXOs
        from the Esplora API, selects the largest confirmed UTXO, constructs
        a P2PKH → OP_RETURN + change transaction, signs it with secp256k1,
        and broadcasts via ``POST /tx``.

        Args:
            op_return_data: Raw bytes to embed in OP_RETURN (max 80 bytes).

        Returns:
            AnchorResult with txid on success.
        """
        if self._privkey is None:
            return AnchorResult(
                chain="btc",
                success=False,
                error=(
                    "BTCBroadcaster requires a WIF key for broadcast(). "
                    "Pass wif=... at construction, or use broadcast_signed() "
                    "with a pre-signed raw transaction."
                ),
            )
        if len(op_return_data) > 80:
            return AnchorResult(
                chain="btc",
                success=False,
                error=f"OP_RETURN data too large: {len(op_return_data)} bytes (max 80)",
            )
        try:
            address = self._derive_address()
            utxos = await self._fetch_utxos(address)
        except Exception as exc:
            return AnchorResult(chain="btc", success=False, error=f"UTXO fetch failed: {exc}")

        if not utxos:
            return AnchorResult(
                chain="btc",
                success=False,
                error=f"No confirmed UTXOs for {address}",
            )

        # Select largest UTXO
        utxo = max(utxos, key=lambda u: u.get("value", 0))
        utxo_sat: int = utxo["value"]
        if utxo_sat <= self._fee_sat:
            return AnchorResult(
                chain="btc",
                success=False,
                error=f"UTXO value {utxo_sat} sat too small for fee {self._fee_sat} sat",
            )

        try:
            raw_hex = _build_btc_tx(
                utxo_txid=utxo["txid"],
                utxo_vout=utxo["vout"],
                utxo_satoshis=utxo_sat,
                op_return_data=op_return_data,
                privkey=self._privkey,
                compressed=self._compressed,
                fee_sat=self._fee_sat,
            )
        except Exception as exc:
            return AnchorResult(chain="btc", success=False, error=f"Tx build failed: {exc}")

        return await self.broadcast_signed(raw_hex)

    async def broadcast_signed(self, raw_tx_hex: str) -> AnchorResult:
        """Broadcast a fully-signed raw transaction hex string via Esplora API.

        Args:
            raw_tx_hex: Signed raw transaction as a hex string.

        Returns:
            AnchorResult with txid on success.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self._api_url}/tx", content=raw_tx_hex)
                if resp.status_code == 200:
                    return AnchorResult(chain="btc", txid=resp.text.strip(), success=True)
                return AnchorResult(
                    chain="btc",
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as exc:
            return AnchorResult(chain="btc", success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Multi-chain anchor coordinator
# ---------------------------------------------------------------------------

class MultiChainAnchor:
    """Coordinates OP_RETURN broadcasts across multiple chains.

    Fires all broadcasts concurrently using asyncio.gather so that latency on
    one chain does not block others.

    Args:
        btc_broadcaster:    A BTCBroadcaster instance for Bitcoin anchoring.
        secondary_chains:   List of chain identifiers to anchor.  Currently
                            only ``"btc"`` is supported.
    """

    def __init__(
        self,
        btc_broadcaster: BTCBroadcaster | None = None,
        secondary_chains: list[str] | None = None,
    ) -> None:
        self._btc = btc_broadcaster or BTCBroadcaster()
        self._chains = set(secondary_chains or ["btc"])

    async def broadcast_all(self, payload: AnchorPayload) -> list[AnchorResult]:
        """Broadcast to all configured secondary chains concurrently.

        Returns a list of AnchorResult objects, one per chain.

        Example::

            results = await anchor.broadcast_all(payload)
            for r in results:
                if r.success:
                    print(f"{r.chain}: txid={r.txid}")
                else:
                    print(f"{r.chain}: FAILED — {r.error}")
        """
        op_return_data = payload.canonical_bytes()
        tasks = []
        for chain in self._chains:
            if chain == "btc":
                tasks.append(self._btc.broadcast(op_return_data))
            else:
                tasks.append(_unsupported_chain(chain))

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)


async def _unsupported_chain(chain: str) -> AnchorResult:
    return AnchorResult(chain=chain, success=False, error=f"Chain {chain!r} not yet supported")


# ---------------------------------------------------------------------------
# OP_RETURN script builder
# ---------------------------------------------------------------------------

def _build_op_return_script(data: bytes) -> bytes:
    """Build a Bitcoin OP_RETURN script for *data*.

    Format: OP_RETURN <PUSHDATA> <data>
    OP_RETURN = 0x6a, PUSHDATA1 = 0x4c for data > 75 bytes.
    """
    OP_RETURN = 0x6A
    if len(data) <= 75:
        return bytes([OP_RETURN, len(data)]) + data
    else:
        OP_PUSHDATA1 = 0x4C
        return bytes([OP_RETURN, OP_PUSHDATA1, len(data)]) + data
