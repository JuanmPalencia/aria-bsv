"""
aria.wallet.hd — BIP32/BIP44 hierarchical deterministic wallet for BSV.

Implements BIP32 extended key derivation so that an ARIA deployment can
manage many system-specific signing keys from a single master seed.

Key derivation scheme for ARIA systems:

    m / 44' / 236' / 0' / system_index / key_index

where:
  44'  = BIP44 purpose
  236' = BSV coin type (SLIP-44)
  0'   = account 0
  system_index = 0-based index of the ARIA system
  key_index    = 0-based key rotation counter (0 = current, 1+ = after rotation)

Usage::

    from aria.wallet.hd import HDWallet
    from aria.broadcaster.arc import ARCBroadcaster

    broadcaster = ARCBroadcaster(api_url="https://arc.taal.com")
    seed_bytes = bytes.fromhex("...")   # 64-byte seed from BIP39 mnemonic

    wallet = HDWallet.from_seed(seed_bytes, broadcaster=broadcaster)

    # Get the signing key for ARIA system #0
    wif = wallet.derive_system_key(system_index=0).wif()

    # Use directly as a WalletInterface
    auditor = InferenceAuditor(config, model_hashes, wallet=wallet)

    # Rotate key for system #2
    new_wif = wallet.rotate(system_index=2)
    # wallet.derive_system_key(2) now returns the rotated key

Watch-only (verification without private key)::

    from aria.wallet.hd import WatchOnlyHDWallet
    watch = WatchOnlyHDWallet.from_xpub(xpub_string)
    address = watch.derive_system_address(system_index=0)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac
import struct
from dataclasses import dataclass, field
from typing import Any

from ..broadcaster.base import BroadcasterInterface
from ..core.errors import ARIAWalletError
from ..core.hasher import canonical_json
from .base import WalletInterface

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HARDENED = 0x80000000
_HMAC_KEY = b"Bitcoin seed"
_BSV_COIN_TYPE = 236          # SLIP-44 coin type for BSV
_ARIA_PREFIX = b"ARIA"
_MIN_RELAY_FEE = 200          # conservative satoshis

# xprv / xpub version bytes (mainnet BIP32)
_XPRV_VERSION = bytes.fromhex("0488ADE4")
_XPUB_VERSION = bytes.fromhex("0488B21E")
# testnet
_TPRV_VERSION = bytes.fromhex("04358394")
_TPUB_VERSION = bytes.fromhex("043587CF")


# ---------------------------------------------------------------------------
# Pure-Python secp256k1 fallback (used when bsvlib is not installed)
# ---------------------------------------------------------------------------

_SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_SECP256K1_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_SECP256K1_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def _ec_point_add(
    P: tuple[int, int] | None, Q: tuple[int, int] | None
) -> tuple[int, int] | None:
    if P is None:
        return Q
    if Q is None:
        return P
    px, py = P
    qx, qy = Q
    if px == qx:
        if py != qy:
            return None
        lam = (3 * px * px * pow(2 * py, _SECP256K1_P - 2, _SECP256K1_P)) % _SECP256K1_P
    else:
        lam = ((qy - py) * pow(qx - px, _SECP256K1_P - 2, _SECP256K1_P)) % _SECP256K1_P
    rx = (lam * lam - px - qx) % _SECP256K1_P
    ry = (lam * (px - rx) - py) % _SECP256K1_P
    return (rx, ry)


def _ec_point_mul(k: int) -> tuple[int, int]:
    result: tuple[int, int] | None = None
    addend: tuple[int, int] | None = (_SECP256K1_GX, _SECP256K1_GY)
    while k:
        if k & 1:
            result = _ec_point_add(result, addend)
        addend = _ec_point_add(addend, addend)
        k >>= 1
    assert result is not None
    return result


def _pubkey_from_privkey(private_key_bytes: bytes) -> bytes:
    k = int.from_bytes(private_key_bytes, "big")
    x, y = _ec_point_mul(k)
    prefix = b"\x02" if y % 2 == 0 else b"\x03"
    return prefix + x.to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hmac_sha512(key: bytes, data: bytes) -> bytes:
    return _hmac.new(key, data, hashlib.sha512).digest()


def _hash256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _hash160(data: bytes) -> bytes:
    return hashlib.new("ripemd160", hashlib.sha256(data).digest()).digest()


def _base58check_encode(payload: bytes) -> str:
    checksum = _hash256(payload)[:4]
    full = payload + checksum
    # count leading zero bytes → leading '1's in Base58
    leading_zeros = 0
    for b in full:
        if b == 0:
            leading_zeros += 1
        else:
            break
    _BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = int.from_bytes(full, "big")
    result = b""
    while n > 0:
        n, r = divmod(n, 58)
        result = bytes([_BASE58_ALPHABET[r]]) + result
    return (_BASE58_ALPHABET[0:1] * leading_zeros + result).decode("ascii")


def _base58check_decode(s: str) -> bytes:
    _BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = 0
    for c in s:
        n = n * 58 + _BASE58_ALPHABET.index(c)
    leading = len(s) - len(s.lstrip("1"))
    full = b"\x00" * leading + n.to_bytes(
        (n.bit_length() + 7) // 8 or 1, "big"
    )
    payload, checksum = full[:-4], full[-4:]
    if _hash256(payload)[:4] != checksum:
        raise ARIAWalletError("invalid key material")
    return payload


# ---------------------------------------------------------------------------
# _BIP32Node — internal extended key
# ---------------------------------------------------------------------------


@dataclass
class _BIP32Node:
    """Internal representation of a BIP32 extended private key node."""

    private_key_bytes: bytes       # 32 bytes: private key scalar
    chain_code: bytes              # 32 bytes
    depth: int = 0
    index: int = 0
    parent_fingerprint: bytes = field(default_factory=lambda: b"\x00\x00\x00\x00")
    network: str = "mainnet"

    # ------------------------------------------------------------------
    # Public key helpers
    # ------------------------------------------------------------------

    def _compressed_pubkey(self) -> bytes:
        """Return the 33-byte compressed SEC-encoded public key."""
        try:
            from bsv import PrivateKey
            return PrivateKey(self.private_key_bytes).public_key().serialize()
        except ImportError:
            return _pubkey_from_privkey(self.private_key_bytes)
        except Exception:
            raise ARIAWalletError("invalid key material")

    # ------------------------------------------------------------------
    # BIP32 child derivation
    # ------------------------------------------------------------------

    def derive_child(self, index: int) -> "_BIP32Node":
        """Derive a child node at *index* (hardened if index ≥ HARDENED_OFFSET).

        Raises:
            ARIAWalletError: if the derived key is invalid (astronomically rare).
        """
        hardened = index >= _HARDENED
        if hardened:
            data = b"\x00" + self.private_key_bytes + struct.pack(">I", index)
        else:
            data = self._compressed_pubkey() + struct.pack(">I", index)

        I = _hmac_sha512(self.chain_code, data)
        IL, IR = I[:32], I[32:]

        try:
            from bsv.curve import curve as _curve
            n = _curve.n
        except Exception:
            # Fallback: secp256k1 order
            n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

        il_int = int.from_bytes(IL, "big")
        if il_int >= n:
            raise ARIAWalletError("invalid key material")

        child_int = (il_int + int.from_bytes(self.private_key_bytes, "big")) % n
        if child_int == 0:
            raise ARIAWalletError("invalid key material")

        parent_fp = _hash160(self._compressed_pubkey())[:4]

        return _BIP32Node(
            private_key_bytes=child_int.to_bytes(32, "big"),
            chain_code=IR,
            depth=self.depth + 1,
            index=index,
            parent_fingerprint=parent_fp,
            network=self.network,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def xprv(self) -> str:
        """Serialize as BIP32 xprv base58check string."""
        version = _TPRV_VERSION if self.network == "testnet" else _XPRV_VERSION
        payload = (
            version
            + bytes([self.depth])
            + self.parent_fingerprint
            + struct.pack(">I", self.index)
            + self.chain_code
            + b"\x00" + self.private_key_bytes
        )
        return _base58check_encode(payload)

    def xpub(self) -> str:
        """Serialize as BIP32 xpub base58check string."""
        version = _TPUB_VERSION if self.network == "testnet" else _XPUB_VERSION
        payload = (
            version
            + bytes([self.depth])
            + self.parent_fingerprint
            + struct.pack(">I", self.index)
            + self.chain_code
            + self._compressed_pubkey()
        )
        return _base58check_encode(payload)

    def wif(self) -> str:
        """Return the WIF-encoded private key for this node."""
        try:
            from bsv import PrivateKey, Network
            network = Network.MAINNET if self.network == "mainnet" else Network.TESTNET
            return PrivateKey(self.private_key_bytes, network=network).wif()
        except ImportError:
            version = b"\x80" if self.network == "mainnet" else b"\xef"
            return _base58check_encode(version + self.private_key_bytes + b"\x01")
        except Exception:
            raise ARIAWalletError("invalid key material")

    def address(self) -> str:
        """Return the P2PKH address for this node."""
        try:
            from bsv import PrivateKey, Network
            network = Network.MAINNET if self.network == "mainnet" else Network.TESTNET
            return PrivateKey(self.private_key_bytes, network=network).address()
        except ImportError:
            version = b"\x00" if self.network == "mainnet" else b"\x6f"
            return _base58check_encode(version + _hash160(self._compressed_pubkey()))
        except Exception:
            raise ARIAWalletError("invalid key material")


# ---------------------------------------------------------------------------
# HDWallet
# ---------------------------------------------------------------------------


class HDWallet(WalletInterface):
    """BIP32/BIP44 hierarchical deterministic BSV wallet.

    Manages multiple ARIA system signing keys from a single master seed.
    Each system gets its own deterministic key path so that keys can be
    audited, rotated, and verified independently.

    Args:
        seed:        64-byte BIP32 master seed (e.g. from BIP39 mnemonic via
                     ``mnemonic.to_seed()``).
        broadcaster: ARIA broadcaster instance for signing and broadcasting.
        network:     ``"mainnet"`` or ``"testnet"``.

    Raises:
        ARIAWalletError: if the seed produces an invalid master key.

    Key derivation path::

        m / 44' / 236' / 0' / <system_index> / <key_index>
    """

    # Default base path for ARIA keys (BIP44, BSV coin type 236, account 0).
    BASE_PATH = f"m/44'/{_BSV_COIN_TYPE}'/0'"

    def __init__(
        self,
        seed: bytes,
        broadcaster: BroadcasterInterface,
        network: str = "mainnet",
    ) -> None:
        if len(seed) < 16:
            raise ARIAWalletError("invalid key material")
        # _master is always at BASE_PATH so derive_system_key uses relative paths
        root = self._derive_master(seed, network)
        self._master = self._derive_path_from_node(root, self.BASE_PATH)
        self._broadcaster = broadcaster
        self._network = network
        # key_rotation[system_index] = current key_index (0-based)
        self._key_rotation: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_seed(
        cls,
        seed: bytes,
        broadcaster: BroadcasterInterface,
        network: str = "mainnet",
    ) -> "HDWallet":
        """Create HDWallet from a raw 64-byte BIP32 seed.

        Args:
            seed:        64-byte master seed.
            broadcaster: ARIA broadcaster instance.
            network:     ``"mainnet"`` or ``"testnet"``.
        """
        return cls(seed, broadcaster, network)

    @classmethod
    def from_xprv(
        cls,
        xprv: str,
        broadcaster: BroadcasterInterface,
    ) -> "HDWallet":
        """Create HDWallet from a BIP32 xprv string.

        Useful when you have an existing HD wallet and want to import a
        subtree into ARIA without exposing the top-level seed.

        Args:
            xprv:        BIP32 extended private key string (xprv... or tprv...).
            broadcaster: ARIA broadcaster instance.
        """
        try:
            payload = _base58check_decode(xprv)
        except Exception:
            raise ARIAWalletError("invalid key material")

        version = payload[:4]
        network = "testnet" if version == _TPRV_VERSION else "mainnet"

        depth = payload[4]
        parent_fp = payload[5:9]
        index = struct.unpack(">I", payload[9:13])[0]
        chain_code = payload[13:45]
        private_key_bytes = payload[46:78]  # skip the 0x00 prefix byte

        wallet = cls.__new__(cls)
        wallet._master = _BIP32Node(
            private_key_bytes=private_key_bytes,
            chain_code=chain_code,
            depth=depth,
            index=index,
            parent_fingerprint=parent_fp,
            network=network,
        )
        wallet._broadcaster = broadcaster
        wallet._network = network
        wallet._key_rotation = {}
        return wallet

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    def derive(self, path: str) -> _BIP32Node:
        """Derive a child node at *path* relative to the current master.

        Args:
            path: Relative BIP32 path string, e.g. ``"0/1"`` or absolute
                  ``"m/44'/236'/0'/0/0"`` (when wallet was created from a
                  root seed — absolute paths are applied from the root).
                  The leading ``"m/"`` prefix is stripped.

        Returns:
            :class:`_BIP32Node` at the specified path.
        """
        parts = path.strip().split("/")
        if parts and parts[0] == "m":
            parts = parts[1:]

        node = self._master
        for part in parts:
            if not part:
                continue
            hardened = part.endswith("'")
            idx = int(part.rstrip("'"))
            if hardened:
                idx += _HARDENED
            node = node.derive_child(idx)
        return node

    def derive_system_key(
        self,
        system_index: int,
        key_index: int | None = None,
    ) -> _BIP32Node:
        """Derive the signing key for an ARIA system.

        Path relative to master: ``{system_index}/{key_index}``
        Full absolute path from root: ``m/44'/236'/0'/{system_index}/{key_index}``

        Args:
            system_index: Zero-based index of the ARIA system.
            key_index:    Key rotation counter.  If ``None``, uses the current
                          rotation level for this system (default 0).

        Returns:
            :class:`_BIP32Node` for the requested system key.
        """
        if key_index is None:
            key_index = self._key_rotation.get(system_index, 0)
        # Relative derivation from _master (already at BASE_PATH)
        node = self._master.derive_child(system_index).derive_child(key_index)
        return node

    def current_wif(self, system_index: int) -> str:
        """Return the current WIF key for a system.

        Args:
            system_index: Zero-based index of the ARIA system.
        """
        return self.derive_system_key(system_index).wif()

    def rotate(self, system_index: int) -> str:
        """Rotate to the next key for a system.

        Increments the ``key_index`` for *system_index* and returns the new
        WIF-encoded key.  The old key remains derivable at the previous
        ``key_index`` for audit trail continuity.

        Args:
            system_index: Zero-based index of the ARIA system to rotate.

        Returns:
            WIF-encoded private key at the new key_index.
        """
        current = self._key_rotation.get(system_index, 0)
        new_index = current + 1
        self._key_rotation[system_index] = new_index
        return self.derive_system_key(system_index, key_index=new_index).wif()

    def key_index(self, system_index: int) -> int:
        """Return the current key rotation index for a system."""
        return self._key_rotation.get(system_index, 0)

    def xpub(self, path: str | None = None) -> str:
        """Return the xpub string for this wallet's master node (BASE_PATH).

        The xpub can be shared with watch-only nodes and auditors without
        exposing the private key.  Pass *path* to derive a sub-node xpub.

        Args:
            path: Optional relative path string from master, e.g. ``"0"``.
                  If ``None``, returns the master (BASE_PATH) xpub.
        """
        node = self._master.derive_child(int(path)) if path else self._master
        return node.xpub()

    def xprv(self, path: str | None = None) -> str:
        """Return the xprv string for this wallet's master node.  **Keep secret.**

        Args:
            path: Optional relative path from master.  If ``None``, returns
                  the master (BASE_PATH) xprv.
        """
        node = self._master.derive_child(int(path)) if path else self._master
        return node.xprv()

    # ------------------------------------------------------------------
    # WalletInterface implementation
    # ------------------------------------------------------------------

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        """Sign and broadcast *payload* using the master ARIA signing key (system 0).

        For multi-system use, obtain the appropriate key via
        :meth:`current_wif` and construct a :class:`aria.wallet.direct.DirectWallet`
        per system.

        Returns:
            BSV txid of the accepted transaction.
        """
        wif = self.current_wif(system_index=0)
        from .direct import DirectWallet
        temp_wallet = DirectWallet(wif, self._broadcaster, self._network)
        return await temp_wallet.sign_and_broadcast(payload)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_master(seed: bytes, network: str) -> _BIP32Node:
        """Derive the BIP32 master node from a seed."""
        I = _hmac_sha512(_HMAC_KEY, seed)
        IL, IR = I[:32], I[32:]

        try:
            from bsv.curve import curve as _curve
            n = _curve.n
        except Exception:
            n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

        il_int = int.from_bytes(IL, "big")
        if il_int == 0 or il_int >= n:
            raise ARIAWalletError("invalid key material")

        return _BIP32Node(
            private_key_bytes=IL,
            chain_code=IR,
            depth=0,
            index=0,
            parent_fingerprint=b"\x00\x00\x00\x00",
            network=network,
        )

    @staticmethod
    def _derive_path_from_node(node: _BIP32Node, path: str) -> _BIP32Node:
        """Derive from *node* following *path* (ignores leading 'm/')."""
        parts = path.strip().split("/")
        if parts and parts[0] == "m":
            parts = parts[1:]
        for part in parts:
            if not part:
                continue
            hardened = part.endswith("'")
            idx = int(part.rstrip("'"))
            if hardened:
                idx += _HARDENED
            node = node.derive_child(idx)
        return node


# ---------------------------------------------------------------------------
# WatchOnlyHDWallet
# ---------------------------------------------------------------------------


class WatchOnlyHDWallet:
    """Watch-only HD wallet for address derivation and audit verification.

    Holds only the xpub — no private key.  Useful for:
    - Generating expected addresses to verify on-chain records.
    - Verifier nodes that must confirm which address signed an epoch.

    Args:
        xpub: BIP32 extended public key string (xpub... or tpub...).

    Usage::

        watch = WatchOnlyHDWallet.from_xpub(wallet.xpub())
        addr = watch.derive_system_address(system_index=0)
    """

    def __init__(self, xpub: str) -> None:
        try:
            payload = _base58check_decode(xpub)
        except Exception:
            raise ARIAWalletError("invalid key material")

        version = payload[:4]
        self._network = "testnet" if version == _TPUB_VERSION else "mainnet"
        self._depth = payload[4]
        self._parent_fp = payload[5:9]
        self._index = struct.unpack(">I", payload[9:13])[0]
        self._chain_code = payload[13:45]
        self._pubkey_bytes = payload[45:78]   # 33-byte compressed pubkey

    @classmethod
    def from_xpub(cls, xpub: str) -> "WatchOnlyHDWallet":
        """Create a WatchOnlyHDWallet from an xpub string."""
        return cls(xpub)

    def derive_system_address(
        self,
        system_index: int,
        key_index: int = 0,
    ) -> str:
        """Return the P2PKH address for a system key.

        Args:
            system_index: Zero-based ARIA system index.
            key_index:    Key rotation counter.

        Returns:
            P2PKH address string.
        """
        # BIP32 public child derivation (non-hardened only)
        node_pubkey, node_chain = self._pubkey_bytes, self._chain_code

        for idx in [system_index, key_index]:
            data = node_pubkey + struct.pack(">I", idx)
            I = _hmac_sha512(node_chain, data)
            IL, IR = I[:32], I[32:]
            try:
                from bsv import curve_multiply, curve_add
                from bsv.curve import curve as _curve

                il_int = int.from_bytes(IL, "big")
                if il_int >= _curve.n:
                    raise ARIAWalletError("invalid key material")

                # G * il_int
                il_point = curve_multiply(il_int, _curve.g)
                # Parse parent public key to Point
                parent_x, parent_y = _parse_pubkey_point(node_pubkey)
                from bsv.curve import Point
                parent_point = Point(parent_x, parent_y)
                # child pubkey = G*il + parent
                child_point = curve_add(il_point, parent_point)
                node_pubkey = _point_to_bytes((child_point.x, child_point.y))
                node_chain = IR
            except ARIAWalletError:
                raise
            except Exception as exc:
                raise ARIAWalletError("invalid key material") from exc

        # P2PKH address = Base58Check(version + HASH160(pubkey))
        h160 = _hash160(node_pubkey)
        prefix = b"\x6f" if self._network == "testnet" else b"\x00"
        return _base58check_encode(prefix + h160)


def _parse_pubkey_point(pubkey_bytes: bytes) -> tuple[int, int]:
    """Parse compressed SEC-encoded pubkey to (x, y) point."""
    x = int.from_bytes(pubkey_bytes[1:], "big")
    prefix = pubkey_bytes[0]
    # y² = x³ + 7 (mod p)
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    y_sq = (pow(x, 3, p) + 7) % p
    y = pow(y_sq, (p + 1) // 4, p)
    if (y % 2 == 0) != (prefix == 0x02):
        y = p - y
    return (x, y)


def _point_to_bytes(point: tuple[int, int]) -> bytes:
    """Encode (x, y) secp256k1 point to 33-byte compressed SEC format."""
    x, y = point
    prefix = b"\x02" if y % 2 == 0 else b"\x03"
    return prefix + x.to_bytes(32, "big")
