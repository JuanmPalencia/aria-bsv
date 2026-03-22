"""ARIA Portal backend — public epoch verification API."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from aria.verify import VerificationResult, Verifier

from ._rate_limit import _RateLimiter, get_limiter

app = FastAPI(
    title="ARIA Portal",
    description="Public verification portal for ARIA-audited AI systems.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dependencies (overridable in tests)
# ---------------------------------------------------------------------------

_network: str = "mainnet"


def get_verifier() -> Verifier:
    return Verifier(network=_network)


VerifierDep = Annotated[Verifier, Depends(get_verifier)]
LimiterDep = Annotated[_RateLimiter, Depends(get_limiter)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "valid": result.valid,
        "tampered": result.tampered,
        "epoch_id": result.epoch_id,
        "system_id": result.system_id,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "decided_at": result.decided_at.isoformat() if result.decided_at else None,
        "records_count": result.records_count,
        "merkle_root": result.merkle_root,
        "error": result.error,
    }


def _check_rate(request: Request, limiter: _RateLimiter) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded — retry in 60 seconds",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/verify/{open_txid}", response_model=dict)
async def verify_epoch(
    open_txid: str,
    request: Request,
    verifier: VerifierDep,
    limiter: LimiterDep,
    close_txid: str | None = None,
) -> dict[str, Any]:
    """Verify an ARIA epoch by its EPOCH_OPEN transaction ID.

    Optionally supply ``close_txid`` as a query parameter if the close
    transaction is known.  Without local storage the close txid must be
    provided explicitly.
    """
    _check_rate(request, limiter)
    result = await verifier.verify_epoch(open_txid, close_txid=close_txid)
    return _result_to_dict(result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
