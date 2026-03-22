"""ARCBroadcaster — broadcasts BSV transactions via the TAAL ARC API."""

from __future__ import annotations

import asyncio
import logging

import httpx

from ..core.errors import ARIABroadcastError
from .base import BroadcasterInterface, TxStatus

_log = logging.getLogger(__name__)

# ARC txStatus values that indicate successful propagation.
_PROPAGATED_STATUSES = frozenset(
    {"SEEN_ON_NETWORK", "SEEN_IN_ORPHAN_MEMPOOL", "STORED", "ANNOUNCED_TO_NETWORK", "MINED"}
)


class ARCBroadcaster(BroadcasterInterface):
    """Broadcasts transactions to BSV via the TAAL ARC REST API.

    Retries on transient HTTP / network errors with exponential backoff.
    A permanent rejection from ARC (4xx) is raised immediately without retry.

    Args:
        api_url:     Base URL of the ARC endpoint.
                     Defaults to the TAAL mainnet endpoint.
        api_key:     Optional Bearer token for authenticated ARC access.
        max_retries: Maximum number of broadcast attempts (including the first).
        base_delay:  Base sleep duration (seconds) before the first retry.
                     Subsequent retries double the delay (1x, 2x, 4x, …).
    """

    def __init__(
        self,
        api_url: str = "https://arc.taal.com",
        api_key: str | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay = base_delay

    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def broadcast(self, raw_tx: str) -> TxStatus:
        """POST the raw transaction to ARC /v1/tx with exponential backoff retry.

        Args:
            raw_tx: Hex-encoded signed transaction.

        Returns:
            TxStatus with txid and propagation flag.

        Raises:
            ARIABroadcastError: If all attempts fail or ARC returns a permanent error.
        """
        last_error = ""
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self._api_url}/v1/tx",
                        headers=self._headers,
                        json={"rawTx": raw_tx},
                    )

                # Permanent client-side rejection — no point retrying.
                if 400 <= response.status_code < 500:
                    body = response.text[:300]
                    raise ARIABroadcastError(
                        f"ARC rejected transaction (HTTP {response.status_code}): {body}"
                    )

                if response.status_code == 200:
                    data = response.json()
                    txid: str = data.get("txid", "")
                    status: str = data.get("txStatus", "")
                    propagated = status in _PROPAGATED_STATUSES
                    return TxStatus(txid=txid, propagated=propagated, message=status)

                last_error = f"HTTP {response.status_code}: {response.text[:200]}"

            except ARIABroadcastError:
                raise
            except httpx.TransportError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)

            if attempt < self._max_retries - 1:
                delay = self._base_delay * (2**attempt)
                _log.warning(
                    "ARC broadcast attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        raise ARIABroadcastError(
            f"ARC broadcast failed after {self._max_retries} attempts: {last_error}"
        )
