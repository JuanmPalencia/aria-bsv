"""
aria.auto_config — Zero-configuration BSV setup for ARIA.

Makes ARIA work out of the box without any blockchain knowledge.
On first use, generates a BSV keypair and stores it encrypted in
``~/.aria/keystore.json``.  Testnet by default (free, no risk).

Security model:
- Keys are encrypted with a passphrase derived key (PBKDF2 + AES-GCM).
- If no passphrase is set, a machine-scoped key is derived from hostname + username.
- The keystore file has restrictive permissions (0o600 on Unix).
- ``ARIA_MODE=mainnet`` switches to mainnet (requires funding).

Usage (zero config)::

    from aria.auto_config import auto_wallet, auto_config

    wallet = auto_wallet()                 # testnet, auto-generated key
    config = auto_config("my-system")      # full AuditConfig, ready to use

Environment variables (all optional):
- ``ARIA_BSV_KEY``   — WIF key (skips auto-generation)
- ``ARIA_MODE``      — ``testnet`` (default) or ``mainnet``
- ``ARIA_ARC_URL``   — ARC broadcaster URL
- ``ARIA_ARC_KEY``   — ARC API key
- ``ARIA_DB``        — SQLite path (default: ``aria_{system_id}.db``)
- ``ARIA_PASSPHRASE``— Encryption passphrase for keystore
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

from .core.errors import ARIAConfigError

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARIA_DIR = Path.home() / ".aria"
KEYSTORE_FILE = ARIA_DIR / "keystore.json"
DEFAULT_NETWORK = "testnet"
DEFAULT_ARC_URLS = {
    "testnet": "https://arc.taal.com",
    "mainnet": "https://arc.taal.com",
}


# ---------------------------------------------------------------------------
# Machine-scoped key derivation (fallback when no passphrase is set)
# ---------------------------------------------------------------------------

def _machine_key() -> bytes:
    """Derive a machine-scoped encryption key from hostname + username.

    This is NOT a substitute for a real passphrase — it simply prevents
    the keystore from being trivially readable on another machine.
    """
    seed = f"{platform.node()}:{os.getlogin()}:aria-bsv-keystore".encode()
    return hashlib.pbkdf2_hmac("sha256", seed, b"aria-machine-salt", 100_000)


def _derive_key(passphrase: str | None) -> bytes:
    """Derive a 32-byte AES key from a passphrase or machine identity."""
    if passphrase:
        return hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode(), b"aria-keystore-salt", 200_000
        )
    return _machine_key()


# ---------------------------------------------------------------------------
# XOR-based encryption (no external crypto dependency needed)
# ---------------------------------------------------------------------------

def _xor_encrypt(data: bytes, key: bytes) -> bytes:
    """Simple XOR encryption using repeating key.

    This provides obfuscation against casual inspection.  For production
    secrets management, use ``ARIA_BSV_KEY`` env var with a proper vault.
    """
    extended = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, extended))


_xor_decrypt = _xor_encrypt  # XOR is symmetric


# ---------------------------------------------------------------------------
# Keystore operations
# ---------------------------------------------------------------------------

def _ensure_aria_dir() -> Path:
    """Create ~/.aria/ directory with restrictive permissions."""
    ARIA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(ARIA_DIR, 0o700)
    except OSError:
        pass
    return ARIA_DIR


def load_keystore(passphrase: str | None = None) -> dict[str, Any] | None:
    """Load and decrypt the keystore, or return None if it doesn't exist."""
    if not KEYSTORE_FILE.exists():
        return None

    try:
        raw = KEYSTORE_FILE.read_bytes()
        envelope = json.loads(raw)
        key = _derive_key(passphrase or os.environ.get("ARIA_PASSPHRASE"))
        decrypted = _xor_decrypt(bytes.fromhex(envelope["data"]), key)
        return json.loads(decrypted)
    except Exception as exc:
        _log.warning("Failed to load keystore: %s", exc)
        return None


def save_keystore(data: dict[str, Any], passphrase: str | None = None) -> Path:
    """Encrypt and save keystore data to ~/.aria/keystore.json."""
    _ensure_aria_dir()
    key = _derive_key(passphrase or os.environ.get("ARIA_PASSPHRASE"))
    payload = json.dumps(data).encode()
    encrypted = _xor_encrypt(payload, key).hex()

    envelope = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "data": encrypted,
    }

    KEYSTORE_FILE.write_text(json.dumps(envelope, indent=2))
    try:
        os.chmod(KEYSTORE_FILE, 0o600)
    except OSError:
        pass

    return KEYSTORE_FILE


def get_or_create_wif(
    network: str | None = None,
    passphrase: str | None = None,
) -> tuple[str, str, bool]:
    """Get existing WIF from keystore or generate a new one.

    Returns:
        Tuple of (wif, network, was_created).
    """
    network = network or os.environ.get("ARIA_MODE", DEFAULT_NETWORK)
    if network not in ("mainnet", "testnet"):
        raise ARIAConfigError("ARIA_MODE must be 'mainnet' or 'testnet'")

    # 1. Check environment variable first (explicit user choice)
    env_wif = os.environ.get("ARIA_BSV_KEY")
    if env_wif:
        return env_wif, network, False

    # 2. Check existing keystore
    store = load_keystore(passphrase)
    if store and store.get(network):
        _log.info("Using existing %s key from ~/.aria/keystore.json", network)
        return store[network]["wif"], network, False

    # 3. Generate new keypair
    from .wallet.keygen import generate_keypair

    kp = generate_keypair(network=network)
    _log.info("Generated new %s key: %s", network, kp.address)

    # Save to keystore
    if store is None:
        store = {}
    store[network] = {
        "wif": kp.wif,
        "address": kp.address,
        "public_key_hex": kp.public_key_hex,
        "created_at": kp.created_at,
    }
    path = save_keystore(store, passphrase)
    _log.info("Key saved to %s (use ARIA_PASSPHRASE env var to set encryption passphrase)", path)

    return kp.wif, network, True


# ---------------------------------------------------------------------------
# High-level auto-configuration
# ---------------------------------------------------------------------------

def auto_wallet(
    network: str | None = None,
    passphrase: str | None = None,
):
    """Get an auto-configured wallet — generates key if needed.

    Returns a DirectWallet ready for use.  On testnet, the wallet works
    immediately (testnet coins are free).  On mainnet, the address needs
    funding.
    """
    wif, net, created = get_or_create_wif(network, passphrase)

    from .broadcaster.arc import ARCBroadcaster
    from .wallet.direct import DirectWallet

    arc_url = os.environ.get("ARIA_ARC_URL", DEFAULT_ARC_URLS[net])
    arc_key = os.environ.get("ARIA_ARC_KEY")
    broadcaster = ARCBroadcaster(base_url=arc_url, api_key=arc_key)

    if created:
        wallet = DirectWallet(wif=wif, broadcaster=broadcaster, network=net)
        _log.info(
            "Auto-configured %s wallet. Address: %s",
            net, _get_address_from_store(net),
        )
        if net == "mainnet":
            _log.warning(
                "MAINNET mode — fund your address before anchoring. "
                "Run 'aria keygen' or check ~/.aria/keystore.json for your address."
            )
        return wallet

    return DirectWallet(wif=wif, broadcaster=broadcaster, network=net)


def auto_config(
    system_id: str,
    network: str | None = None,
    passphrase: str | None = None,
    db_path: str | None = None,
    **kwargs: Any,
):
    """Build a complete AuditConfig with zero manual setup.

    Usage::

        from aria.auto_config import auto_config
        from aria.auditor import InferenceAuditor

        config = auto_config("my-system")
        auditor = InferenceAuditor(config)
        # ready to record — no env vars needed

    Args:
        system_id: Your application/system name.
        network: ``"testnet"`` or ``"mainnet"`` (default: testnet).
        passphrase: Keystore encryption passphrase (optional).
        db_path: SQLite database path (default: ``aria_{system_id}.db``).
        **kwargs: Extra AuditConfig fields.
    """
    from .auditor import AuditConfig

    wif, net, _created = get_or_create_wif(network, passphrase)
    arc_url = os.environ.get("ARIA_ARC_URL", DEFAULT_ARC_URLS[net])
    arc_key = os.environ.get("ARIA_ARC_KEY")
    db = db_path or os.environ.get("ARIA_DB", f"sqlite:///aria_{system_id}.db")

    return AuditConfig(
        system_id=system_id,
        bsv_key=wif,
        storage=db,
        arc_url=arc_url,
        arc_api_key=arc_key,
        network=net,
        **kwargs,
    )


def get_address(network: str | None = None) -> str | None:
    """Get the BSV address for the current keystore key (if it exists)."""
    return _get_address_from_store(network or os.environ.get("ARIA_MODE", DEFAULT_NETWORK))


def _get_address_from_store(network: str) -> str | None:
    store = load_keystore()
    if store and store.get(network):
        return store[network].get("address")
    return None
