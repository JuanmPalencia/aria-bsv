"""Abstract broadcaster interface for ARIA BSV transactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TxStatus:
    """Result of a broadcast attempt.

    Attributes:
        txid:       The transaction ID returned by the network (64 hex chars).
        propagated: True if the network accepted the transaction.
        message:    Human-readable status or error message from the broadcaster.
    """

    txid: str
    propagated: bool
    message: str = field(default="")


class BroadcasterInterface(ABC):
    """Abstract base class for broadcasting signed BSV transactions.

    Implementations handle the network transport (ARC, WhatsOnChain, etc.)
    and MUST raise ARIABroadcastError if the transaction cannot be delivered
    after all retry attempts.
    """

    @abstractmethod
    async def broadcast(self, raw_tx: str) -> TxStatus:
        """Broadcast a signed raw transaction (hex-encoded) to BSV.

        Args:
            raw_tx: Hex-encoded signed transaction bytes.

        Returns:
            TxStatus describing the broadcast result.

        Raises:
            ARIABroadcastError: If all retry attempts fail.
        """
