"""BRC100Wallet — delegates signing to an external BRC-100 wallet service."""

from __future__ import annotations

import httpx

from ..core.errors import ARIAWalletError
from .base import WalletInterface


class BRC100Wallet(WalletInterface):
    """Wallet that delegates signing and broadcasting to a BRC-100 endpoint.

    Use this when the operator does not want the ARIA SDK to hold the private
    key directly.  The external wallet service receives the ARIA payload dict
    and is responsible for building, signing, and broadcasting the transaction.

    The endpoint must implement:
        POST  <endpoint>/sign
              Body:   {"payload": <dict>}
              200:    {"txid": "<64 hex chars>"}

    Args:
        endpoint: Base URL of the BRC-100 wallet service.
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        """Forward *payload* to the BRC-100 wallet and return the txid.

        Raises:
            ARIAWalletError: On any HTTP, network, or response-format error.
                             The message never exposes key material.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._endpoint}/sign",
                    json={"payload": payload},
                )
                response.raise_for_status()
                data = response.json()
                txid: str = data.get("txid", "")
                if not txid or len(txid) != 64:
                    raise ARIAWalletError("invalid key material")
                return txid
        except ARIAWalletError:
            raise
        except Exception:
            raise ARIAWalletError("invalid key material")
