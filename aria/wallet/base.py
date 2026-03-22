"""Abstract wallet interface for ARIA transaction signing."""

from __future__ import annotations

from abc import ABC, abstractmethod


class WalletInterface(ABC):
    """Abstract base class for ARIA wallet implementations.

    A wallet is responsible for:
    1. Building a BSV transaction with an OP_RETURN output that embeds the
       serialised ARIA protocol payload.
    2. Signing the transaction with the operator's private key.
    3. Broadcasting the signed transaction to the BSV network.

    Security contract — non-negotiable:
        Implementations MUST NEVER include private key material, WIF strings,
        or any cryptographic derivative of the key in log messages, exception
        messages, or error responses.  Any signing failure MUST surface as
        ``ARIAWalletError("invalid key material")``.
    """

    @abstractmethod
    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        """Build, sign, and broadcast a BSV OP_RETURN transaction.

        The *payload* dict is serialised to canonical JSON (via
        ``aria.core.hasher.canonical_json``) and embedded in an OP_RETURN
        output prefixed with the ``b"ARIA"`` protocol marker.

        Args:
            payload: ARIA protocol payload (EPOCH_OPEN or EPOCH_CLOSE dict).
                     Must be JSON-serialisable and contain no NaN / Infinity.

        Returns:
            The txid (64 lower-case hex characters) of the accepted transaction.

        Raises:
            ARIAWalletError: On any key or signing error.
                             The message NEVER exposes key material.
            ARIABroadcastError: If the network rejects the transaction.
        """
