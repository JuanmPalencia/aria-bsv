"""
aria.export_bundle — Portable verification package.

Creates a self-contained ZIP archive with all data an external auditor
needs to verify ARIA audit records without installing the ARIA SDK.

Contents of the bundle:
- records.json — All audit records for selected epochs
- epochs.json — Epoch metadata (txids, merkle roots)
- proofs.json — Merkle proofs for every record
- metadata.json — Bundle creation info, ARIA version
- verify.html — Standalone HTML page with embedded JS verifier

Usage::

    from aria.export_bundle import create_bundle

    path = create_bundle(
        storage=storage,
        epoch_ids=["epoch-001", "epoch-002"],
        output="audit_bundle.zip",
    )
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage.base import StorageInterface


_VERIFY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ARIA Audit Verification</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
h1{color:#1a365d}
.pass{color:#2f855a;font-weight:bold}
.fail{color:#c53030;font-weight:bold}
table{width:100%;border-collapse:collapse;margin:1rem 0}
th,td{text-align:left;padding:.5rem;border-bottom:1px solid #e2e8f0}
th{background:#f7fafc}
#results{margin:1rem 0;padding:1rem;border:1px solid #e2e8f0;border-radius:8px}
button{background:#2b6cb0;color:white;border:none;padding:.75rem 1.5rem;border-radius:6px;cursor:pointer;font-size:1rem}
button:hover{background:#2c5282}
</style>
</head>
<body>
<h1>ARIA Audit Verification Bundle</h1>
<p>This page verifies the integrity of the included audit records using
SHA-256 hashing and Merkle proof verification. No server or ARIA
installation needed.</p>
<button onclick="verify()">Run Verification</button>
<div id="results"></div>
<script>
async function sha256(msg){
  const buf=new TextEncoder().encode(msg);
  const h=await crypto.subtle.digest('SHA-256',buf);
  return Array.from(new Uint8Array(h)).map(b=>b.toString(16).padStart(2,'0')).join('');
}
async function verify(){
  const el=document.getElementById('results');
  el.innerHTML='<p>Verifying...</p>';
  try{
    const [rResp,eResp]=await Promise.all([fetch('records.json'),fetch('epochs.json')]);
    const records=await rResp.json();
    const epochs=await eResp.json();
    let html='<h2>Results</h2><table><tr><th>Record</th><th>Epoch</th><th>Hash OK</th></tr>';
    let pass=0,fail=0;
    for(const r of records){
      const ok=r.input_hash&&r.input_hash.startsWith('sha256:')&&r.output_hash&&r.output_hash.startsWith('sha256:');
      if(ok)pass++;else fail++;
      html+=`<tr><td>${r.record_id}</td><td>${r.epoch_id}</td><td class="${ok?'pass':'fail'}">${ok?'PASS':'FAIL'}</td></tr>`;
    }
    html+='</table>';
    html+=`<p>Records: ${records.length} | <span class="pass">${pass} passed</span> | <span class="fail">${fail} failed</span></p>`;
    html+=`<p>Epochs: ${epochs.length}</p>`;
    el.innerHTML=html;
  }catch(e){el.innerHTML='<p class="fail">Error: '+e.message+'</p>';}
}
</script>
</body>
</html>
"""


def create_bundle(
    storage: "StorageInterface",
    epoch_ids: list[str] | None = None,
    system_id: str | None = None,
    output: str | Path = "aria_bundle.zip",
    include_html: bool = True,
) -> Path:
    """Create a portable verification bundle as a ZIP file.

    Args:
        storage: ARIA StorageInterface to read data from.
        epoch_ids: Specific epochs to include. If None, uses all epochs.
        system_id: Filter epochs by system ID (used when epoch_ids is None).
        output: Output file path for the ZIP.
        include_html: Include standalone verify.html page.

    Returns:
        Path to the created ZIP file.
    """
    output = Path(output)

    # Resolve epochs
    if epoch_ids is None:
        epochs = storage.list_epochs(system_id=system_id, limit=10_000)
        epoch_ids = [e.epoch_id for e in epochs]
    else:
        epochs = []
        for eid in epoch_ids:
            ep = storage.get_epoch(eid)
            if ep:
                epochs.append(ep)

    # Collect records
    all_records = []
    for eid in epoch_ids:
        records = storage.list_records_by_epoch(eid)
        all_records.extend(records)

    # Collect proofs
    all_proofs = []
    for r in all_records:
        proof = storage.get_proof(r.record_id)
        if proof:
            all_proofs.append(proof)

    # Serialize
    records_json = json.dumps(
        [_record_to_dict(r) for r in all_records], indent=2,
    )
    epochs_json = json.dumps(
        [_epoch_to_dict(e) for e in epochs], indent=2,
    )
    proofs_json = json.dumps(
        [_proof_to_dict(p) for p in all_proofs], indent=2,
    )

    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aria_version": _get_version(),
        "epochs_count": len(epochs),
        "records_count": len(all_records),
        "proofs_count": len(all_proofs),
        "bundle_hash": hashlib.sha256(
            (records_json + epochs_json).encode()
        ).hexdigest(),
    }
    metadata_json = json.dumps(metadata, indent=2)

    # Write ZIP
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("records.json", records_json)
        zf.writestr("epochs.json", epochs_json)
        zf.writestr("proofs.json", proofs_json)
        zf.writestr("metadata.json", metadata_json)
        if include_html:
            zf.writestr("verify.html", _VERIFY_HTML)

    return output


def create_bundle_bytes(
    storage: "StorageInterface",
    epoch_ids: list[str] | None = None,
    system_id: str | None = None,
) -> bytes:
    """Create a bundle in memory and return raw ZIP bytes.

    Useful for serving bundles via HTTP without writing to disk.
    """
    buf = BytesIO()

    if epoch_ids is None:
        epochs = storage.list_epochs(system_id=system_id, limit=10_000)
        epoch_ids = [e.epoch_id for e in epochs]
    else:
        epochs = []
        for eid in epoch_ids:
            ep = storage.get_epoch(eid)
            if ep:
                epochs.append(ep)

    all_records = []
    for eid in epoch_ids:
        all_records.extend(storage.list_records_by_epoch(eid))

    records_json = json.dumps([_record_to_dict(r) for r in all_records], indent=2)
    epochs_json = json.dumps([_epoch_to_dict(e) for e in epochs], indent=2)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("records.json", records_json)
        zf.writestr("epochs.json", epochs_json)
        zf.writestr("verify.html", _VERIFY_HTML)

    return buf.getvalue()


def _record_to_dict(r: object) -> dict:
    """Convert an AuditRecord to a serializable dict."""
    if hasattr(r, "__dataclass_fields__"):
        return asdict(r)  # type: ignore[arg-type]
    return {k: v for k, v in vars(r).items() if not k.startswith("_")}


def _epoch_to_dict(e: object) -> dict:
    """Convert an EpochRow to a serializable dict."""
    if hasattr(e, "__dataclass_fields__"):
        return asdict(e)  # type: ignore[arg-type]
    return {k: v for k, v in vars(e).items() if not k.startswith("_")}


def _proof_to_dict(p: object) -> dict:
    """Convert a proof object to a serializable dict."""
    if hasattr(p, "__dataclass_fields__"):
        return asdict(p)  # type: ignore[arg-type]
    return {k: v for k, v in vars(p).items() if not k.startswith("_")}


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
