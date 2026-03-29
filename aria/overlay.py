"""aria.overlay — BSV Overlay Services client (BRC-31).

Provides:
    - TopicManager: submit transactions to a topic and track admitted outputs.
    - LookupService: query a lookup service for outputs matching a given query.
    - OverlayClient: high-level client that combines both.

All network I/O is async (httpx).  Classes are fully dependency-injectable
for testing via the ``client`` parameter.

BRC-31 reference: https://github.com/bitcoin-sv/BRCs/blob/master/overlay/0031.md

The ARIA overlay protocol uses two topics:
    - ``tm_aria_epochs``:  EPOCH_OPEN and EPOCH_CLOSE transactions.
    - ``tm_aria_records``: per-record OP_RETURN anchoring (when enabled).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx

from aria.core.errors import ARIAError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OverlayError(ARIAError):
    """Raised when an overlay network operation fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmittanceResult:
    """Result of submitting a transaction to a topic.

    Attributes:
        txid:            64-char hex transaction ID.
        topic:           Topic name (e.g. ``"tm_aria_epochs"``).
        admitted:        True if at least one output was admitted.
        admitted_outputs: 0-based indices of admitted outputs.
        message:         Human-readable status message.
    """

    txid: str
    topic: str
    admitted: bool
    admitted_outputs: list[int]
    message: str


@dataclass(frozen=True)
class LookupResult:
    """A single output returned by a lookup service query.

    Attributes:
        txid:       Transaction ID.
        output_index: Output index within the transaction.
        beef:       BEEF-encoded transaction data (hex), if available.
        data:       Arbitrary metadata returned by the lookup service.
    """

    txid: str
    output_index: int
    beef: str | None
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# TopicManager
# ---------------------------------------------------------------------------


class TopicManager:
    """Submit BSV transactions to an overlay topic.

    The topic manager admits (or rejects) transactions based on protocol
    rules.  For ARIA, epoch transactions are submitted to ``tm_aria_epochs``.

    Args:
        base_url: Overlay node base URL.
        topic:    Topic name (default: ``"tm_aria_epochs"``).
        api_key:  Optional Bearer token for the overlay node.
        client:   Optional ``httpx.AsyncClient`` (injected in tests).
    """

    def __init__(
        self,
        base_url: str,
        topic: str = "tm_aria_epochs",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._topic = topic
        self._api_key = api_key
        self._client = client

    @property
    def topic(self) -> str:
        return self._topic

    async def submit(self, raw_tx: str) -> AdmittanceResult:
        """Submit a raw hex transaction to the topic.

        Args:
            raw_tx: Hex-encoded raw transaction bytes.

        Returns:
            :class:`AdmittanceResult` describing the overlay node's decision.

        Raises:
            OverlayError: On network failure or unexpected HTTP error.
        """
        url = f"{self._base}/v1/submit"
        headers = self._headers()
        body = {"rawTx": raw_tx, "topics": [self._topic]}

        try:
            resp = await self._get(url, headers=headers, body=body)
        except OverlayError:
            raise
        except Exception as exc:
            raise OverlayError(f"Failed to submit to topic {self._topic}: {exc}") from exc

        # BRC-31 response: {"topics": {"tm_aria_epochs": {"outputsToAdmit": [...], "coinsToRetain": [...]}}}
        topic_data = resp.get("topics", {}).get(self._topic, {})
        admitted_outputs: list[int] = topic_data.get("outputsToAdmit", [])
        admitted = len(admitted_outputs) > 0

        # Extract txid from BEEF or response
        txid: str = resp.get("txid", "")

        return AdmittanceResult(
            txid=txid,
            topic=self._topic,
            admitted=admitted,
            admitted_outputs=admitted_outputs,
            message="OK" if admitted else "No outputs admitted",
        )

    async def _get(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        if self._client is not None:
            resp = await self._client.post(url, json=body, headers=headers, timeout=15.0)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=body, headers=headers, timeout=15.0)
        if not resp.is_success:
            raise OverlayError(
                f"Overlay node returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h


# ---------------------------------------------------------------------------
# LookupService
# ---------------------------------------------------------------------------


class LookupService:
    """Query an overlay lookup service for outputs.

    Args:
        base_url:        Overlay node base URL.
        service_name:    Lookup service name (default: ``"ls_aria"``).
        api_key:         Optional Bearer token.
        client:          Optional ``httpx.AsyncClient`` (injected in tests).
    """

    def __init__(
        self,
        base_url: str,
        service_name: str = "ls_aria",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._service = service_name
        self._api_key = api_key
        self._client = client

    @property
    def service_name(self) -> str:
        return self._service

    async def lookup(
        self,
        query: dict[str, Any],
        limit: int = 10,
    ) -> list[LookupResult]:
        """Query the lookup service for matching outputs.

        Args:
            query:  Service-specific query dict.  For ARIA, pass e.g.
                    ``{"system_id": "my-system", "epoch_id": "..."}``
            limit:  Maximum number of results to return (default: 10).

        Returns:
            List of :class:`LookupResult` objects.

        Raises:
            OverlayError: On network failure.
        """
        url = f"{self._base}/v1/lookup"
        headers = self._headers()
        body = {
            "service": self._service,
            "query": query,
            "limit": limit,
        }

        try:
            raw = await self._post(url, headers=headers, body=body)
        except OverlayError:
            raise
        except Exception as exc:
            raise OverlayError(
                f"Lookup service {self._service} query failed: {exc}"
            ) from exc

        results: list[LookupResult] = []
        for item in raw.get("results", []):
            results.append(
                LookupResult(
                    txid=str(item.get("txid", "")),
                    output_index=int(item.get("outputIndex", 0)),
                    beef=item.get("beef"),
                    data=dict(item.get("data", {})),
                )
            )
        return results

    async def _post(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        if self._client is not None:
            resp = await self._client.post(url, json=body, headers=headers, timeout=15.0)
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=body, headers=headers, timeout=15.0)
        if not resp.is_success:
            raise OverlayError(
                f"Lookup service returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()  # type: ignore[no-any-return]

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h


# ---------------------------------------------------------------------------
# OverlayClient — high-level combined interface
# ---------------------------------------------------------------------------


class OverlayClient:
    """High-level client combining a :class:`TopicManager` and a
    :class:`LookupService` for the ARIA overlay protocol.

    Args:
        base_url:       Overlay node base URL.
        epoch_topic:    Topic for epoch transactions
                        (default: ``"tm_aria_epochs"``).
        lookup_service: Lookup service name (default: ``"ls_aria"``).
        api_key:        Optional Bearer token.
        client:         Optional ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        base_url: str,
        epoch_topic: str = "tm_aria_epochs",
        lookup_service: str = "ls_aria",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.topic_manager = TopicManager(
            base_url=base_url,
            topic=epoch_topic,
            api_key=api_key,
            client=client,
        )
        self.lookup_service = LookupService(
            base_url=base_url,
            service_name=lookup_service,
            api_key=api_key,
            client=client,
        )

    async def submit_epoch(self, raw_tx: str) -> AdmittanceResult:
        """Submit an epoch transaction (OPEN or CLOSE) to the overlay network."""
        return await self.topic_manager.submit(raw_tx)

    async def find_epochs(
        self,
        system_id: str,
        epoch_id: str | None = None,
        limit: int = 10,
    ) -> list[LookupResult]:
        """Query the overlay network for epoch records.

        Args:
            system_id:  ARIA system ID to filter by.
            epoch_id:   Optional specific epoch ID.
            limit:      Maximum results to return.
        """
        query: dict[str, Any] = {"system_id": system_id}
        if epoch_id is not None:
            query["epoch_id"] = epoch_id
        return await self.lookup_service.lookup(query, limit=limit)

    async def find_records(
        self,
        system_id: str,
        epoch_id: str,
        limit: int = 100,
    ) -> list[LookupResult]:
        """Query the overlay network for audit records in an epoch."""
        query = {
            "system_id": system_id,
            "epoch_id": epoch_id,
            "type": "AUDIT_RECORD",
        }
        return await self.lookup_service.lookup(query, limit=limit)
