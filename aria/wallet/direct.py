"""DirectWallet — signs BSV transactions with a local WIF private key."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core.errors import ARIABroadcastError, ARIAWalletError
from ..core.hasher import canonical_json
from ..broadcaster.base import BroadcasterInterface
from .base import WalletInterface

_log = logging.getLogger(__name__)

# ARIA BSV protocol marker — prefixed to every OP_RETURN output.
_ARIA_PREFIX = b"ARIA"

# Conservative relay fee for a minimal OP_RETURN transaction (satoshis).
_MIN_RELAY_FEE = 200


class DirectWallet(WalletInterface):
    """Wallet that builds and signs BSV transactions with a local WIF key.

    The private key is held in memory as a ``PrivateKey`` object from
    ``bsvlib``.  It is NEVER written to logs, tracebacks, or error messages.

    UTXO fetching defaults to WhatsOnChain; override ``_get_utxos`` in a
    subclass (or monkeypatch in tests) to inject alternative providers.

    Args:
        wif:         WIF-encoded private key for the funding address.
        broadcaster: Broadcaster instance used to submit the signed transaction.
        network:     ``"mainnet"`` or ``"testnet"``.
    """

    def __init__(
        self,
        wif: str,
        broadcaster: BroadcasterInterface,
        network: str = "mainnet",
    ) -> None:
        try:
            from bsvlib.keys import PrivateKey  # type: ignore[import]
            from bsvlib.constants import Chain  # type: ignore[import]

            self._key = PrivateKey(wif)
            self._chain = Chain.MAIN if network == "mainnet" else Chain.TEST
        except Exception:
            raise ARIAWalletError("invalid key material")

        self._broadcaster = broadcaster
        self._network = network

    async def sign_and_broadcast(self, payload: dict) -> str:  # type: ignore[type-arg]
        """Serialise *payload* to canonical JSON, embed in OP_RETURN, sign, broadcast.

        Returns:
            The txid of the accepted transaction.

        Raises:
            ARIAWalletError:    If the transaction cannot be signed (e.g. no UTXOs).
            ARIABroadcastError: If the broadcaster rejects the transaction.
        """
        data = canonical_json(payload)
        try:
            raw_hex = await self._build_signed_tx(data)
        except ARIAWalletError:
            raise
        except Exception:
            # Never let internal exceptions leak key material.
            raise ARIAWalletError("invalid key material")

        result = await self._broadcaster.broadcast(raw_hex)
        if not result.propagated:
            raise ARIABroadcastError(f"transaction not propagated: {result.message}")
        return result.txid

    async def _build_signed_tx(self, data: bytes) -> str:
        """Build a raw signed BSV transaction with an OP_RETURN embedding *data*."""
        utxos = await asyncio.to_thread(self._get_utxos)
        if not utxos:
            raise ARIAWalletError("invalid key material")

        from bsvlib.transaction.transaction import Transaction, TxOutput  # type: ignore[import]
        from bsvlib.transaction.unspent import Unspent  # type: ignore[import]

        unspent = Unspent(**utxos[0])
        tx = Transaction(chain=self._chain)
        tx.add_input(unspent)

        # OP_RETURN output: ARIA prefix + canonical JSON payload.
        tx.add_output(TxOutput(out=[_ARIA_PREFIX, data], satoshi=0))

        # Return change to the funding address (minus relay fee).
        change = unspent.satoshi - _MIN_RELAY_FEE
        if change > 0:
            address = self._key.address()
            tx.add_output(TxOutput(out=address, satoshi=change))

        tx.sign()
        return tx.hex()

    def _get_utxos(self) -> list[dict[str, Any]]:
        """Fetch UTXOs for the funding address via WhatsOnChain.

        Override this method to inject a custom UTXO provider or test doubles.
        Returns a list of dicts compatible with ``bsvlib.transaction.unspent.Unspent``.
        """
        from bsvlib.service.service import WhatsOnChain  # type: ignore[import]

        service = WhatsOnChain(chain=self._chain)
        address = self._key.address()
        return service.get_unspents(private_keys=[self._key], address=address)  # type: ignore[return-value]
