"""ARIA + KAIROS integration example.

KAIROS is a hypothetical AI-driven emergency dispatch system that processes
incoming 112 calls and assigns a triage priority. This example shows how to
integrate ARIA into a FastAPI service that runs two sequential models:

    1. triage_model   — classifies the emergency type and urgency
    2. dispatch_model — determines which unit to dispatch

Both models are audited independently within the same ARIA epoch, giving
regulators a complete record of every decision: which model version,
which input (PII-stripped), which output, and when.

Run:
    uvicorn examples.kairos.main:app --reload

Environment variables (see minimal/main.py for setup):
    ARIA_BSV_KEY        — funded BSV testnet WIF key
    ARIA_ARC_API_KEY    — ARC broadcaster API key (optional on testnet)
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aria import AuditConfig, InferenceAuditor
from aria.integrations.fastapi import ARIAMiddleware, audit_inference

# ---------------------------------------------------------------------------
# Simulated KAIROS models
# ---------------------------------------------------------------------------

_TRIAGE_MODEL_HASH = "sha256:" + hashlib.sha256(b"triage-model-v2.3").hexdigest()
_DISPATCH_MODEL_HASH = "sha256:" + hashlib.sha256(b"dispatch-model-v1.7").hexdigest()


def _triage_model(call_data: dict) -> dict:
    """Assign emergency priority based on reported symptoms."""
    keywords = str(call_data.get("description", "")).lower()
    if any(k in keywords for k in ("chest", "cardiac", "stroke", "unconscious")):
        return {"priority": 1, "category": "life_threatening", "confidence": 0.97}
    if any(k in keywords for k in ("fracture", "bleeding", "accident")):
        return {"priority": 2, "category": "urgent", "confidence": 0.89}
    return {"priority": 3, "category": "non_urgent", "confidence": 0.76}


def _dispatch_model(triage_result: dict, location: str) -> dict:
    """Choose the nearest appropriate unit based on triage result."""
    priority = triage_result["priority"]
    unit_map = {1: "ALPHA-07", 2: "BRAVO-12", 3: "CHARLIE-03"}
    return {
        "unit": unit_map.get(priority, "CHARLIE-03"),
        "estimated_arrival_min": 4 if priority == 1 else 8 if priority == 2 else 15,
        "location": location,
    }


# ---------------------------------------------------------------------------
# ARIA setup
# ---------------------------------------------------------------------------

_auditor: InferenceAuditor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _auditor
    _auditor = InferenceAuditor(
        config=AuditConfig(
            system_id="kairos-v2",
            bsv_key=os.environ.get("ARIA_BSV_KEY", "placeholder"),
            arc_api_key=os.environ.get("ARIA_ARC_API_KEY"),
            network="mainnet",
            batch_ms=1_500,          # KAIROS closes epoch every 1.5 seconds
            batch_size=500,
            pii_fields=["caller_phone", "caller_name", "patient_name"],
        ),
        model_hashes={
            "triage-v2.3": _TRIAGE_MODEL_HASH,
            "dispatch-v1.7": _DISPATCH_MODEL_HASH,
        },
        initial_state={
            "protocol_version": "KAIROS-2026-001",
            "coverage_area": "Valencia Metropolitan Area",
            "active_units": 24,
        },
    )
    yield
    if _auditor:
        _auditor.close()


app = FastAPI(
    title="KAIROS Emergency Dispatch",
    description="AI-driven 112 emergency dispatch — ARIA-audited via BRC-121.",
    lifespan=lifespan,
)

app.add_middleware(ARIAMiddleware, auditor=lambda: _auditor)

# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class DispatchRequest(BaseModel):
    incident_id: str
    caller_phone: str          # PII — stripped by ARIA before hashing
    caller_name: str           # PII — stripped by ARIA before hashing
    description: str
    location: str


class DispatchResponse(BaseModel):
    incident_id: str
    triage_priority: int
    triage_category: str
    assigned_unit: str
    estimated_arrival_min: int
    audit: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(req: DispatchRequest) -> DispatchResponse:
    """Process an emergency call and dispatch the appropriate unit.

    Both the triage and dispatch decisions are independently recorded in ARIA.
    The caller's phone number and name are stripped before hashing (PII fields).
    """
    if _auditor is None:
        raise HTTPException(503, "Auditor not ready")

    start = time.monotonic()

    # Step 1 — Triage
    triage_input = {
        "incident_id": req.incident_id,
        "description": req.description,
        "location": req.location,
        # caller_phone and caller_name are PII — ARIA will strip them
        "caller_phone": req.caller_phone,
        "caller_name": req.caller_name,
    }
    triage_output = _triage_model(triage_input)
    triage_latency = int((time.monotonic() - start) * 1000)

    triage_record_id = _auditor.record(
        model_id="triage-v2.3",
        input=triage_input,
        output=triage_output,
        confidence=triage_output["confidence"],
        latency_ms=triage_latency,
        metadata={"incident_id": req.incident_id, "decision_class": "triage"},
    )

    # Step 2 — Dispatch
    dispatch_input = {
        "triage_result": triage_output,
        "location": req.location,
        "incident_id": req.incident_id,
    }
    dispatch_output = _dispatch_model(triage_output, req.location)
    dispatch_latency = int((time.monotonic() - start) * 1000) - triage_latency

    dispatch_record_id = _auditor.record(
        model_id="dispatch-v1.7",
        input=dispatch_input,
        output=dispatch_output,
        latency_ms=dispatch_latency,
        metadata={"incident_id": req.incident_id, "decision_class": "dispatch"},
    )

    receipt = _auditor.get_receipt(dispatch_record_id)

    return DispatchResponse(
        incident_id=req.incident_id,
        triage_priority=triage_output["priority"],
        triage_category=triage_output["category"],
        assigned_unit=dispatch_output["unit"],
        estimated_arrival_min=dispatch_output["estimated_arrival_min"],
        audit={
            "epoch_id": receipt.epoch_id,
            "triage_record_id": triage_record_id,
            "dispatch_record_id": dispatch_record_id,
            "open_txid": receipt.open_txid,
            "verification_url": f"https://portal.aria-bsv.io/verify/{receipt.open_txid}",
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "system_id": "kairos-v2"}
