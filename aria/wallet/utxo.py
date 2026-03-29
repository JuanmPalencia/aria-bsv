"""aria.wallet.utxo — UTXO manager with fee oracle for BSV transactions.

Provides:
    - UTXOSet: fetches and caches spendable UTXOs for a P2PKH address.
    - FeeOracle: queries the ARC endpoint for current fee rate (sat/byte).
    - select_utxos(): largest-first coin selection with change calculation.

All network I/O is async (httpx).  Every public class is dependency-injectable
for testing — pass a custom httpx.AsyncClient via the ``client`` parameter.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Sequence

import httpx

from aria.core.errors import ARIAWalletError

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UTXO:
    """A single unspent transaction output."""

    txid: str
    """64-char lowercase hex transaction ID."""
    vout: int
    """Output index within the transaction."""
    satoshis: int
    """Value in satoshis (must be > 0)."""
    script_pubkey: str
    """Hex-encoded locking script (P2PKH)."""
    height: int = 0
    """Block height (0 = unconfirmed)."""

    def __post_init__(self) -> None:
        if self.satoshis <= 0:
            raise ARIAWalletError(f"UTXO satoshis must be > 0, got {self.satoshis}")
        if len(self.txid) != 64:
            raise ARIAWalletError(
                f"UTXO txid must be 64 hex chars, got length {len(self.txid)}"
            )


@dataclass
class CoinSelection:
    """Result of a UTXO coin-selection operation."""

    inputs: list[UTXO]
    """Selected UTXOs to use as transaction inputs."""
    change_satoshis: int
    """Change amount to return to the sender (0 if no change needed)."""
    fee_satoshis: int
    """Estimated transaction fee in satoshis."""
    total_in: int
    """Sum of input values in satoshis."""

    @property
    def sufficient(self) -> bool:
        """True if the selection covers target + fee."""
        return self.change_satoshis >= 0


# ---------------------------------------------------------------------------
# Fee oracle
# ---------------------------------------------------------------------------

_DEFAULT_ARC_URL = "https://arc.taal.com"
_DEFAULT_FEE_RATE = 1  # sat/byte — conservative default


class FeeOracle:
    """Queries an ARC endpoint for the current BSV fee rate.

    The fee rate is cached for ``ttl_seconds`` (default 60 s) to avoid
    hammering the API on every transaction.

    Args:
        arc_url:     ARC base URL (default: ``https://arc.taal.com``).
        arc_api_key: Optional Bearer token for the ARC API.
        ttl_seconds: How long to cache the fee rate (default: 60 s).
        client:      Optional pre-built ``httpx.AsyncClient`` (injected in
                     tests to avoid real network calls).
    """

    def __init__(
        self,
        arc_url: str = _DEFAULT_ARC_URL,
        arc_api_key: str | None = None,
        ttl_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._arc_url = arc_url.rstrip("/")
        self._arc_api_key = arc_api_key
        self._ttl = ttl_seconds
        self._client = client
        self._cached_rate: float = float(_DEFAULT_FEE_RATE)
        self._cache_ts: float = 0.0
        self._lock = asyncio.Lock()

    async def get_rate(self) -> float:
        """Return the current fee rate in sat/byte.

        Returns the cached value if it is still fresh.  Falls back to
        :data:`_DEFAULT_FEE_RATE` if the ARC API is unreachable.
        """
        now = time.monotonic()
        async with self._lock:
            if now - self._cache_ts < self._ttl:
                return self._cached_rate
            rate = await self._fetch_rate()
            self._cached_rate = rate
            self._cache_ts = now
        return rate

    async def _fetch_rate(self) -> float:
        url = f"{self._arc_url}/v1/policy"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._arc_api_key:
            headers["Authorization"] = f"Bearer {self._arc_api_key}"
        try:
            if self._client is not None:
                resp = await self._client.get(url, headers=headers, timeout=5.0)
            else:
                async with httpx.AsyncClient() as c:
                    resp = await c.get(url, headers=headers, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            # ARC /v1/policy returns {"policy": {"miningFee": {"satoshis": N, "bytes": M}}}
            mining_fee = data.get("policy", {}).get("miningFee", {})
            sats = int(mining_fee.get("satoshis", 1))
            byt = int(mining_fee.get("bytes", 1))
            if byt > 0:
                return sats / byt
        except Exception:
            pass
        return float(_DEFAULT_FEE_RATE)

    def invalidate(self) -> None:
        """Force a re-fetch on the next call to :meth:`get_rate`."""
        self._cache_ts = 0.0


# ---------------------------------------------------------------------------
# UTXO set (fetcher + cache)
# ---------------------------------------------------------------------------

_WOC_BASE = "https://api.whatsonchain.com/v1/bsv/main"


class UTXOSet:
    """Fetches and caches spendable UTXOs for a P2PKH BSV address.

    Args:
        address:     BSV P2PKH address (base58check).
        explorer_url: Block explorer base URL (default: WhatsOnChain).
        ttl_seconds: Cache lifetime in seconds (default: 30 s).
        client:      Optional ``httpx.AsyncClient`` (injected in tests).
    """

    def __init__(
        self,
        address: str,
        explorer_url: str = _WOC_BASE,
        ttl_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._address = address
        self._base = explorer_url.rstrip("/")
        self._ttl = ttl_seconds
        self._client = client
        self._utxos: list[UTXO] = []
        self._cache_ts: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def address(self) -> str:
        return self._address

    async def get(self, *, force_refresh: bool = False) -> list[UTXO]:
        """Return the list of UTXOs for the address.

        Args:
            force_refresh: Bypass cache and re-fetch from the explorer.

        Returns:
            List of :class:`UTXO` objects sorted by value descending.
        """
        now = time.monotonic()
        async with self._lock:
            if not force_refresh and (now - self._cache_ts) < self._ttl:
                return list(self._utxos)
            self._utxos = await self._fetch()
            self._cache_ts = now
        return list(self._utxos)

    async def total_satoshis(self) -> int:
        """Return the sum of all spendable satoshis."""
        return sum(u.satoshis for u in await self.get())

    def invalidate(self) -> None:
        """Clear cache so the next :meth:`get` re-fetches from the network."""
        self._cache_ts = 0.0

    async def _fetch(self) -> list[UTXO]:
        """Fetch UTXOs from WhatsOnChain (or compatible explorer)."""
        url = f"{self._base}/address/{self._address}/unspent"
        try:
            if self._client is not None:
                resp = await self._client.get(url, timeout=10.0)
            else:
                async with httpx.AsyncClient() as c:
                    resp = await c.get(url, timeout=10.0)
            resp.raise_for_status()
            raw: list[dict[str, object]] = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ARIAWalletError(
                f"Explorer returned HTTP {exc.response.status_code} "
                f"for address {self._address}"
            ) from exc
        except Exception as exc:
            raise ARIAWalletError(
                f"Failed to fetch UTXOs for {self._address}: {exc}"
            ) from exc

        utxos: list[UTXO] = []
        for item in raw:
            try:
                utxos.append(
                    UTXO(
                        txid=str(item["tx_hash"]),
                        vout=int(item["tx_pos"]),  # type: ignore[arg-type]
                        satoshis=int(item["value"]),  # type: ignore[arg-type]
                        script_pubkey=str(item.get("script", "")),
                        height=int(item.get("height", 0)),  # type: ignore[arg-type]
                    )
                )
            except (KeyError, ARIAWalletError):
                # Skip malformed or zero-value entries
                continue

        # Sort largest-first for greedy selection
        utxos.sort(key=lambda u: u.satoshis, reverse=True)
        return utxos


# ---------------------------------------------------------------------------
# Coin selection
# ---------------------------------------------------------------------------

# Approximate byte sizes for P2PKH transactions
_INPUT_BYTES = 148   # one P2PKH input
_OUTPUT_BYTES = 34   # one P2PKH output
_OVERHEAD_BYTES = 10  # version + locktime + input/output counts


def estimate_tx_bytes(n_inputs: int, n_outputs: int) -> int:
    """Estimate the serialised size of a P2PKH transaction in bytes.

    Uses standard constants: 148 bytes/input, 34 bytes/output, 10 overhead.
    """
    return _OVERHEAD_BYTES + n_inputs * _INPUT_BYTES + n_outputs * _OUTPUT_BYTES


def select_utxos(
    utxos: Sequence[UTXO],
    target_satoshis: int,
    fee_rate: float = 1.0,
    n_outputs: int = 2,
) -> CoinSelection:
    """Select the minimum set of UTXOs to cover *target_satoshis* + fee.

    Uses a largest-first greedy strategy.

    Args:
        utxos:           Available UTXOs (sorted largest-first is fastest).
        target_satoshis: Amount to send (excluding fee) in satoshis.
        fee_rate:        Sat/byte fee rate (from :class:`FeeOracle`).
        n_outputs:       Number of outputs in the target transaction
                         (typically 2: recipient + change).

    Returns:
        A :class:`CoinSelection` describing the selected inputs and change.

    Raises:
        ARIAWalletError: If the total available balance is insufficient.
    """
    if target_satoshis <= 0:
        raise ARIAWalletError(
            f"target_satoshis must be positive, got {target_satoshis}"
        )

    # Sort largest-first (defensive — caller may pass unsorted)
    sorted_utxos = sorted(utxos, key=lambda u: u.satoshis, reverse=True)

    selected: list[UTXO] = []
    total_in = 0

    for utxo in sorted_utxos:
        selected.append(utxo)
        total_in += utxo.satoshis

        n_inputs = len(selected)
        fee = int(estimate_tx_bytes(n_inputs, n_outputs) * fee_rate) + 1
        required = target_satoshis + fee

        if total_in >= required:
            change = total_in - required
            return CoinSelection(
                inputs=selected,
                change_satoshis=change,
                fee_satoshis=fee,
                total_in=total_in,
            )

    # Insufficient funds
    available = sum(u.satoshis for u in sorted_utxos)
    fee = int(estimate_tx_bytes(len(sorted_utxos), n_outputs) * fee_rate) + 1
    raise ARIAWalletError(
        f"Insufficient funds: need {target_satoshis + fee} satoshis "
        f"({target_satoshis} + {fee} fee), have {available}"
    )
