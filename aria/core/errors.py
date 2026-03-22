"""
aria/core/errors.py

ARIA exception hierarchy.

All exceptions are subclasses of ARIAError. Catch ARIAError to handle any
ARIA-specific error. Catch a subclass to handle a specific category.

Security note: no exception in this hierarchy ever exposes key material,
raw private keys, or WIF strings in its message.
"""


class ARIAError(Exception):
    """Base class for all ARIA exceptions."""


class ARIAConfigError(ARIAError):
    """
    Invalid or missing configuration.
    Raised during InferenceAuditor initialisation when AuditConfig is malformed.
    """


class ARIASerializationError(ARIAError):
    """
    Object cannot be serialised to canonical JSON.

    Common causes:
    - NaN or Infinity in a numeric field
    - Non-JSON-serialisable Python type (e.g. set, bytes, custom object)
    - Hash string not in expected 'sha256:{hex}' format
    - AuditRecord field fails validation
    """


class ARIAWalletError(ARIAError):
    """
    Wallet signing or broadcast failed.

    The message never contains the private key, WIF string, or any
    derivation of key material.
    """


class ARIABroadcastError(ARIAError):
    """
    Transaction could not be broadcast to BSV.

    Includes the HTTP status code and ARC error code when available,
    but never raw transaction hex or key material.
    """


class ARIAStorageError(ARIAError):
    """
    Local storage operation failed (read, write, or schema migration).
    """


class ARIAVerificationError(ARIAError):
    """
    Verification could not be completed due to an infrastructure error
    (e.g. BSV node unreachable, malformed transaction).

    Distinct from ARIATamperDetected: this means the verification
    process itself failed, not that tampering was found.
    """


class ARIAZKError(ARIAError):
    """
    Zero-knowledge proof operation failed.

    Common causes:
    - ZK prover library (e.g., ezkl) not installed — pip install aria-bsv[zk]
    - Model too large for the configured prover tier
    - Proof verification failed (cryptographic)
    - Circuit compilation failed
    """


class ARIATamperDetected(ARIAError):
    """
    Cryptographic verification found evidence of tampering.

    This is the most critical exception in the hierarchy. When raised,
    the audit trail for the affected epoch cannot be trusted.

    Attributes:
        epoch_id: The epoch in which tampering was detected.
        detail:   Human-readable description of what failed.
    """

    def __init__(self, epoch_id: str, detail: str) -> None:
        self.epoch_id = epoch_id
        self.detail = detail
        super().__init__(f"Tampering detected in epoch '{epoch_id}': {detail}")
