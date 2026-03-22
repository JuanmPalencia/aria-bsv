"""ARIA minimal quickstart.

Demonstrates the 5-line integration pattern:
    1. Configure with system_id and BSV key.
    2. Create an InferenceAuditor with model hashes.
    3. Call record() after each inference.
    4. ARIA handles epochs, Merkle trees, and BSV broadcasts automatically.

Usage:
    Set the environment variables below, then run:
        python examples/minimal/main.py

Required environment variables:
    ARIA_BSV_KEY    — WIF private key for a funded BSV testnet address
    ARIA_SYSTEM_ID  — unique identifier for this system (e.g. "my-demo-system")
    ARIA_MODEL_HASH — SHA-256 of your model file (use aria.hash_file to get it)
"""

from __future__ import annotations

import os
import time

from aria import AuditConfig, InferenceAuditor


def simulate_inference(prompt: str) -> dict:
    """Placeholder for a real model call."""
    time.sleep(0.01)  # simulate 10ms latency
    return {"answer": f"processed: {prompt}", "confidence": 0.91}


def main() -> None:
    bsv_key = os.environ.get("ARIA_BSV_KEY", "")
    system_id = os.environ.get("ARIA_SYSTEM_ID", "minimal-demo")
    model_hash = os.environ.get("ARIA_MODEL_HASH", "sha256:" + "a" * 64)

    if not bsv_key:
        print("ARIA_BSV_KEY not set — running in dry-run mode (no real BSV broadcasts).")

    # --- 1. Configure ---
    config = AuditConfig(
        system_id=system_id,
        bsv_key=bsv_key or "placeholder",  # replaced by mock in dry-run
        network="mainnet",
        batch_ms=10_000,   # close epoch every 10 seconds
        batch_size=100,    # or after 100 records
    )

    # --- 2. Create auditor ---
    auditor = InferenceAuditor(
        config=config,
        model_hashes={"demo-model": model_hash},
        initial_state={"version": "1.0.0", "threshold": 0.5},
    )

    print(f"Auditor ready — system_id={system_id!r}")

    # --- 3. Run some inferences ---
    prompts = [
        "Is the patient stable?",
        "Approve this loan application?",
        "Is this content safe?",
    ]

    for i, prompt in enumerate(prompts):
        start = time.monotonic()
        output = simulate_inference(prompt)
        latency_ms = int((time.monotonic() - start) * 1000)

        record_id = auditor.record(
            model_id="demo-model",
            input={"prompt": prompt},
            output=output,
            confidence=output["confidence"],
            latency_ms=latency_ms,
        )
        print(f"  [{i}] recorded: {record_id}  latency={latency_ms}ms")

    # --- 4. Force close epoch and get receipt ---
    auditor.flush()

    # Show receipt for last record
    receipt = auditor.get_receipt(record_id)
    print(f"\nReceipt for last record:")
    print(f"  record_id : {receipt.record_id}")
    print(f"  epoch_id  : {receipt.epoch_id}")
    print(f"  open_txid : {receipt.open_txid}")

    # --- 5. Clean up ---
    auditor.close()
    print("\nDone. Epoch sealed on BSV.")


if __name__ == "__main__":
    main()
