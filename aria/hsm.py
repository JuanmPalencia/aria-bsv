"""
aria.hsm — Hardware Security Module (HSM) key management abstraction.

Provides a unified interface for signing and key management operations
that can be backed by:

* SoftHSM  — PKCS#11-based software HSM (useful for dev/CI)
* AWSKMSHSM — AWS Key Management Service via boto3
* LocalHSM  — In-process software key store (pure Python, no deps)
* MockHSM   — Deterministic in-memory HSM for testing

All backends implement :class:`HSMInterface`. Key material never leaves
the HSM boundary; callers receive only public keys and signatures.

Integrates with ARIA's BRC-121 epoch system: epoch commitment hashes are
signed by the HSM before broadcasting to BSV, providing a hardware-backed
chain of custody.

Usage::

    from aria.hsm import LocalHSM, HSMKeySpec, HSMAlgorithm

    hsm = LocalHSM()
    key_id = hsm.generate_key(HSMKeySpec(algorithm=HSMAlgorithm.ECDSA_P256, label="epoch-signer"))
    sig = hsm.sign(key_id, b"commitment-hash-bytes")
    ok = hsm.verify(key_id, b"commitment-hash-bytes", sig)
    print(ok)  # True
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data models
# ---------------------------------------------------------------------------


class HSMAlgorithm(str, Enum):
    ECDSA_P256     = "ecdsa-p256"
    ECDSA_SECP256K1 = "ecdsa-secp256k1"   # BSV native curve
    ED25519        = "ed25519"
    HMAC_SHA256    = "hmac-sha256"
    RSA_2048       = "rsa-2048"


class HSMKeyState(str, Enum):
    ACTIVE   = "active"
    DISABLED = "disabled"
    DELETED  = "deleted"


@dataclass
class HSMKeySpec:
    """Parameters for key generation."""
    algorithm:   HSMAlgorithm
    label:       str = ""
    extractable: bool = False
    metadata:    dict = field(default_factory=dict)


@dataclass
class HSMKeyInfo:
    """Metadata for a key stored in the HSM (no private key material)."""
    key_id:     str
    algorithm:  HSMAlgorithm
    label:      str
    state:      HSMKeyState
    created_at: str
    public_key: bytes = field(default_factory=bytes)
    metadata:   dict  = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "key_id":     self.key_id,
            "algorithm":  self.algorithm.value,
            "label":      self.label,
            "state":      self.state.value,
            "created_at": self.created_at,
            "public_key": self.public_key.hex(),
            "metadata":   self.metadata,
        }


@dataclass
class HSMSignResult:
    """Result of a signing operation."""
    key_id:    str
    algorithm: HSMAlgorithm
    signature: bytes
    signed_at: str = ""

    def __post_init__(self):
        if not self.signed_at:
            self.signed_at = datetime.now(timezone.utc).isoformat()

    @property
    def signature_hex(self) -> str:
        return self.signature.hex()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HSMError(Exception):
    """Base HSM error."""


class HSMKeyNotFoundError(HSMError):
    """Key ID does not exist in the HSM."""


class HSMKeyDisabledError(HSMError):
    """Key exists but is in DISABLED or DELETED state."""


class HSMAlgorithmMismatchError(HSMError):
    """Operation incompatible with key algorithm."""


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------


@runtime_checkable
class HSMInterface(Protocol):
    """Common interface for all HSM backends."""

    def generate_key(self, spec: HSMKeySpec) -> str:
        """Generate a new key; return its key_id."""
        ...

    def list_keys(self) -> list[HSMKeyInfo]:
        """List all non-deleted key metadata."""
        ...

    def get_key(self, key_id: str) -> HSMKeyInfo:
        """Return key metadata. Raises :exc:`HSMKeyNotFoundError`."""
        ...

    def disable_key(self, key_id: str) -> None:
        """Disable a key (still exists, cannot sign)."""
        ...

    def delete_key(self, key_id: str) -> None:
        """Soft-delete a key (irreversible)."""
        ...

    def sign(self, key_id: str, data: bytes) -> HSMSignResult:
        """Sign *data* with *key_id*."""
        ...

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Verify *signature* against *data* using *key_id*. Returns bool."""
        ...


# ---------------------------------------------------------------------------
# LocalHSM — pure-Python in-process backend
# ---------------------------------------------------------------------------

class LocalHSM:
    """
    Pure-Python software key store using HMAC-SHA256 (default) or
    stub asymmetric signatures.

    Not suitable for production security. Use for development, testing,
    and CI pipelines where real HSM hardware is unavailable.
    """

    def __init__(self) -> None:
        self._keys:        dict[str, HSMKeyInfo]  = {}
        self._private_material: dict[str, bytes]  = {}
        self._key_counter = 0

    # ------------------------------------------------------------------
    # Key lifecycle
    # ------------------------------------------------------------------

    def generate_key(self, spec: HSMKeySpec) -> str:
        self._key_counter += 1
        key_id = f"local-key-{self._key_counter:04d}-{secrets.token_hex(4)}"
        # Generate raw key material (never exposed)
        raw = os.urandom(32)
        self._private_material[key_id] = raw
        # Derive a deterministic public key (for software HSM we use SHA-256 of raw)
        pub = hashlib.sha256(raw).digest()
        info = HSMKeyInfo(
            key_id=key_id,
            algorithm=spec.algorithm,
            label=spec.label,
            state=HSMKeyState.ACTIVE,
            created_at=datetime.now(timezone.utc).isoformat(),
            public_key=pub,
            metadata=dict(spec.metadata),
        )
        self._keys[key_id] = info
        _log.debug("LocalHSM: generated key %s (%s)", key_id, spec.algorithm)
        return key_id

    def list_keys(self) -> list[HSMKeyInfo]:
        return [k for k in self._keys.values() if k.state != HSMKeyState.DELETED]

    def get_key(self, key_id: str) -> HSMKeyInfo:
        info = self._keys.get(key_id)
        if info is None:
            raise HSMKeyNotFoundError(f"Key not found: {key_id}")
        return info

    def disable_key(self, key_id: str) -> None:
        info = self.get_key(key_id)
        if info.state == HSMKeyState.DELETED:
            raise HSMKeyDisabledError(f"Key already deleted: {key_id}")
        info.state = HSMKeyState.DISABLED

    def delete_key(self, key_id: str) -> None:
        info = self.get_key(key_id)
        info.state = HSMKeyState.DELETED
        # Wipe private material
        self._private_material.pop(key_id, None)

    # ------------------------------------------------------------------
    # Cryptographic operations
    # ------------------------------------------------------------------

    def sign(self, key_id: str, data: bytes) -> HSMSignResult:
        info = self._check_operable(key_id)
        raw = self._private_material[key_id]
        sig = self._do_sign(info.algorithm, raw, data)
        return HSMSignResult(
            key_id=key_id,
            algorithm=info.algorithm,
            signature=sig,
        )

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        info = self._check_operable(key_id)
        raw = self._private_material.get(key_id)
        if raw is None:
            return False
        expected = self._do_sign(info.algorithm, raw, data)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_operable(self, key_id: str) -> HSMKeyInfo:
        info = self.get_key(key_id)
        if info.state != HSMKeyState.ACTIVE:
            raise HSMKeyDisabledError(
                f"Key {key_id} is {info.state.value}, cannot use for signing"
            )
        return info

    def _do_sign(self, algorithm: HSMAlgorithm, raw: bytes, data: bytes) -> bytes:
        """Deterministic signing stub — uses HMAC-SHA256 for all algorithms."""
        # In production this would dispatch to appropriate crypto per algorithm.
        # For LocalHSM we use HMAC-SHA256 as a secure, deterministic approximation.
        return hmac.new(raw, data, hashlib.sha256).digest()


# ---------------------------------------------------------------------------
# MockHSM — fully deterministic, for unit tests
# ---------------------------------------------------------------------------

class MockHSM:
    """
    Deterministic in-memory HSM for unit testing.

    Signatures are SHA-256(key_id + data) for full determinism.
    """

    def __init__(self) -> None:
        self._keys:    dict[str, HSMKeyInfo] = {}
        self._secrets: dict[str, bytes] = {}
        self._counter  = 0

    def generate_key(self, spec: HSMKeySpec) -> str:
        self._counter += 1
        kid = f"mock-{self._counter:04d}"
        secret = hashlib.sha256(kid.encode()).digest()
        self._secrets[kid] = secret
        pub = hashlib.sha256(secret).digest()
        self._keys[kid] = HSMKeyInfo(
            key_id=kid,
            algorithm=spec.algorithm,
            label=spec.label,
            state=HSMKeyState.ACTIVE,
            created_at="2025-01-01T00:00:00+00:00",
            public_key=pub,
        )
        return kid

    def list_keys(self) -> list[HSMKeyInfo]:
        return [k for k in self._keys.values() if k.state != HSMKeyState.DELETED]

    def get_key(self, key_id: str) -> HSMKeyInfo:
        if key_id not in self._keys:
            raise HSMKeyNotFoundError(key_id)
        return self._keys[key_id]

    def disable_key(self, key_id: str) -> None:
        self.get_key(key_id).state = HSMKeyState.DISABLED

    def delete_key(self, key_id: str) -> None:
        self.get_key(key_id).state = HSMKeyState.DELETED
        self._secrets.pop(key_id, None)

    def sign(self, key_id: str, data: bytes) -> HSMSignResult:
        info = self.get_key(key_id)
        if info.state != HSMKeyState.ACTIVE:
            raise HSMKeyDisabledError(key_id)
        sig = hashlib.sha256(key_id.encode() + data).digest()
        return HSMSignResult(key_id=key_id, algorithm=info.algorithm, signature=sig)

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        try:
            result = self.sign(key_id, data)
            return hmac.compare_digest(result.signature, signature)
        except (HSMKeyNotFoundError, HSMKeyDisabledError):
            return False


# ---------------------------------------------------------------------------
# HSMSigningProxy — wraps an existing ARIA wallet/signer with HSM signing
# ---------------------------------------------------------------------------

class HSMSigningProxy:
    """
    Wraps any object that needs a `.sign_bytes(data) -> bytes` call
    and routes it through an HSM backend.

    Usage::

        proxy = HSMSigningProxy(hsm=LocalHSM(), key_id=kid)
        sig = proxy.sign_bytes(b"hello")
    """

    def __init__(self, hsm: Any, key_id: str) -> None:
        self._hsm = hsm
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign_bytes(self, data: bytes) -> bytes:
        result = self._hsm.sign(self._key_id, data)
        return result.signature

    def verify_bytes(self, data: bytes, signature: bytes) -> bool:
        return self._hsm.verify(self._key_id, data, signature)

    def public_key_hex(self) -> str:
        info = self._hsm.get_key(self._key_id)
        return info.public_key.hex()


# ---------------------------------------------------------------------------
# AWSKMSHSM — AWS KMS-backed HSM (requires boto3)
# ---------------------------------------------------------------------------

class AWSKMSError(HSMError):
    """Raised when boto3/KMS is unavailable or an API call fails."""


_KMS_STATE_MAP = {
    "Enabled": HSMKeyState.ACTIVE,
    "Disabled": HSMKeyState.DISABLED,
    "PendingDeletion": HSMKeyState.DELETED,
    "PendingImport": HSMKeyState.DISABLED,
    "Unavailable": HSMKeyState.DISABLED,
}

_KMS_SPEC_TO_ALGO: dict[str, HSMAlgorithm] = {
    "ECC_NIST_P256":    HSMAlgorithm.ECDSA_P256,
    "ECC_SECG_P256K1":  HSMAlgorithm.ECDSA_SECP256K1,
    "RSA_2048":         HSMAlgorithm.RSA_2048,
}


class AWSKMSHSM:
    """AWS KMS-backed HSM using boto3.

    Install: ``pip install boto3``

    Private key material never leaves AWS KMS — all signing happens inside
    the AWS hardware security module. This class is a thin wrapper around
    the KMS Sign/Verify/CreateKey APIs.

    Args:
        region:            AWS region (default ``"us-east-1"``).
        key_alias_prefix:  Prefix applied to alias names when listing ARIA keys
                           (default ``"aria/"``). Keys without this prefix are
                           excluded from ``list_keys()``.

    Example::

        hsm = AWSKMSHSM(region="eu-west-1")
        key_id = hsm.generate_key(HSMKeySpec(HSMAlgorithm.ECDSA_SECP256K1))
        result = hsm.sign(key_id, b"epoch-commitment-bytes")
        assert hsm.verify(key_id, b"epoch-commitment-bytes", result.signature)
    """

    def __init__(
        self,
        region: str = "us-east-1",
        key_alias_prefix: str = "aria/",
    ) -> None:
        try:
            import boto3  # type: ignore[import]
            self._boto3 = boto3
            self._client = boto3.client("kms", region_name=region)
        except ImportError as exc:
            raise AWSKMSError(
                "boto3 is required for AWSKMSHSM. Install: pip install boto3"
            ) from exc
        self._region = region
        self._prefix = key_alias_prefix
        self._key_cache: dict[str, HSMKeyInfo] = {}

    # ------------------------------------------------------------------
    # Key lifecycle
    # ------------------------------------------------------------------

    def generate_key(self, spec: HSMKeySpec) -> str:
        kms_spec = _algorithm_to_kms_spec(spec.algorithm)
        try:
            resp = self._client.create_key(
                Description=spec.label or "aria-hsm-key",
                KeyUsage="SIGN_VERIFY",
                CustomerMasterKeySpec=kms_spec,
                Tags=[{"TagKey": "aria-managed", "TagValue": "true"}],
            )
        except Exception as exc:
            raise AWSKMSError(f"KMS CreateKey failed: {exc}") from exc

        key_id: str = resp["KeyMetadata"]["KeyId"]

        try:
            pub_resp = self._client.get_public_key(KeyId=key_id)
            pub = bytes(pub_resp.get("PublicKey", b""))
        except Exception:
            pub = b""

        # Create an alias so the key is discoverable by prefix
        alias = f"alias/{self._prefix}{key_id[:8]}"
        try:
            self._client.create_alias(AliasName=alias, TargetKeyId=key_id)
        except Exception:
            pass  # Alias creation is best-effort

        info = HSMKeyInfo(
            key_id=key_id,
            algorithm=spec.algorithm,
            label=spec.label,
            state=HSMKeyState.ACTIVE,
            created_at=datetime.now(timezone.utc).isoformat(),
            public_key=pub,
        )
        self._key_cache[key_id] = info
        return key_id

    def list_keys(self) -> list[HSMKeyInfo]:
        """List all ARIA-managed KMS keys (paginated).

        Fetches keys from AWS KMS whose aliases start with ``key_alias_prefix``
        and merges them with the local cache. Keys not in the cache have their
        metadata fetched on first access.
        """
        result: list[HSMKeyInfo] = []
        try:
            paginator = self._client.get_paginator("list_aliases")
            for page in paginator.paginate():
                for alias in page.get("Aliases", []):
                    alias_name: str = alias.get("AliasName", "")
                    if not alias_name.startswith(f"alias/{self._prefix}"):
                        continue
                    key_id = alias.get("TargetKeyId")
                    if not key_id:
                        continue
                    if key_id not in self._key_cache:
                        info = self._fetch_key_info(key_id)
                        if info:
                            self._key_cache[key_id] = info
                    if key_id in self._key_cache:
                        result.append(self._key_cache[key_id])
        except Exception:
            # Fall back to cache-only if pagination fails (e.g., in tests)
            result = list(self._key_cache.values())
        return result

    def _fetch_key_info(self, key_id: str) -> HSMKeyInfo | None:
        """Fetch key metadata from KMS and return an HSMKeyInfo."""
        try:
            meta = self._client.describe_key(KeyId=key_id)["KeyMetadata"]
            kms_spec: str = meta.get("CustomerMasterKeySpec", "")
            algorithm = _KMS_SPEC_TO_ALGO.get(kms_spec, HSMAlgorithm.ECDSA_P256)
            state_str: str = meta.get("KeyState", "Enabled")
            state = _KMS_STATE_MAP.get(state_str, HSMKeyState.DISABLED)
            created = meta.get("CreationDate")
            created_iso = created.isoformat() if created else datetime.now(timezone.utc).isoformat()

            try:
                pub_resp = self._client.get_public_key(KeyId=key_id)
                pub = bytes(pub_resp.get("PublicKey", b""))
            except Exception:
                pub = b""

            return HSMKeyInfo(
                key_id=key_id,
                algorithm=algorithm,
                label=meta.get("Description", ""),
                state=state,
                created_at=created_iso,
                public_key=pub,
            )
        except Exception:
            return None

    def get_key(self, key_id: str) -> HSMKeyInfo:
        if key_id not in self._key_cache:
            info = self._fetch_key_info(key_id)
            if info is None:
                raise HSMKeyNotFoundError(key_id)
            self._key_cache[key_id] = info
        return self._key_cache[key_id]

    def disable_key(self, key_id: str) -> None:
        try:
            self._client.disable_key(KeyId=key_id)
        except Exception as exc:
            raise AWSKMSError(f"KMS DisableKey failed: {exc}") from exc
        info = self.get_key(key_id)
        info.state = HSMKeyState.DISABLED

    def delete_key(self, key_id: str, pending_window_days: int = 7) -> None:
        try:
            self._client.schedule_key_deletion(
                KeyId=key_id,
                PendingWindowInDays=pending_window_days,
            )
        except Exception as exc:
            raise AWSKMSError(f"KMS ScheduleKeyDeletion failed: {exc}") from exc
        info = self.get_key(key_id)
        info.state = HSMKeyState.DELETED

    # ------------------------------------------------------------------
    # Signing / verification
    # ------------------------------------------------------------------

    def sign(self, key_id: str, data: bytes) -> HSMSignResult:
        info = self.get_key(key_id)
        if info.state != HSMKeyState.ACTIVE:
            raise AWSKMSError(f"Key {key_id} is not active (state={info.state})")
        signing_algo = _algorithm_to_kms_signing(info.algorithm)
        try:
            resp = self._client.sign(
                KeyId=key_id,
                Message=data,
                MessageType="RAW",
                SigningAlgorithm=signing_algo,
            )
        except Exception as exc:
            raise AWSKMSError(f"KMS Sign failed: {exc}") from exc
        return HSMSignResult(
            key_id=key_id,
            algorithm=info.algorithm,
            signature=bytes(resp["Signature"]),
        )

    def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        info = self.get_key(key_id)
        signing_algo = _algorithm_to_kms_signing(info.algorithm)
        try:
            resp = self._client.verify(
                KeyId=key_id,
                Message=data,
                MessageType="RAW",
                Signature=signature,
                SigningAlgorithm=signing_algo,
            )
            return bool(resp.get("SignatureValid", False))
        except Exception:
            return False


def _algorithm_to_kms_spec(algo: HSMAlgorithm) -> str:
    _map = {
        HSMAlgorithm.ECDSA_P256:      "ECC_NIST_P256",
        HSMAlgorithm.ECDSA_SECP256K1: "ECC_SECG_P256K1",
        HSMAlgorithm.RSA_2048:        "RSA_2048",
    }
    return _map.get(algo, "ECC_NIST_P256")


def _algorithm_to_kms_signing(algo: HSMAlgorithm) -> str:
    _map = {
        HSMAlgorithm.ECDSA_P256:      "ECDSA_SHA_256",
        HSMAlgorithm.ECDSA_SECP256K1: "ECDSA_SHA_256",
        HSMAlgorithm.RSA_2048:        "RSASSA_PKCS1_V1_5_SHA_256",
    }
    return _map.get(algo, "ECDSA_SHA_256")
