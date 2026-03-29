"""
aria.dashboard — Lightweight local web dashboard for ARIA.

Serves a single-page dashboard showing epochs, records, compliance
status, and system health. Built with FastAPI and vanilla HTML/JS,
no frontend build step required.

Usage::

    from aria.dashboard import create_dashboard_app, serve

    # Create a FastAPI app
    app = create_dashboard_app(storage)

    # Or serve directly (blocking)
    serve(storage, port=8710)

    # CLI: aria dashboard --port 8710
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .storage.base import StorageInterface

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARIA Dashboard</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--text:#e2e8f0;--accent:#3b82f6;--green:#22c55e;--yellow:#eab308;--red:#ef4444}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);padding:1.5rem}
h1{font-size:1.75rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:.75rem}
h1 span{color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin-bottom:1.5rem}
.card{background:var(--card);border-radius:12px;padding:1.25rem}
.card h3{font-size:.875rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}
.card .value{font-size:2rem;font-weight:700}
.card .value.green{color:var(--green)}
.card .value.yellow{color:var(--yellow)}
.card .value.red{color:var(--red)}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:12px;overflow:hidden}
th{text-align:left;padding:.75rem 1rem;background:#334155;color:#94a3b8;font-size:.8rem;text-transform:uppercase}
td{padding:.75rem 1rem;border-top:1px solid #334155;font-size:.9rem}
tr:hover td{background:#334155}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:9999px;font-size:.75rem;font-weight:600}
.badge.open{background:#22c55e33;color:var(--green)}
.badge.closed{background:#3b82f633;color:var(--accent)}
.section{margin-bottom:2rem}
.section h2{font-size:1.25rem;margin-bottom:1rem}
#error{color:var(--red);margin:1rem 0}
.refresh{background:var(--accent);color:white;border:none;padding:.5rem 1rem;border-radius:8px;cursor:pointer;font-size:.875rem}
.refresh:hover{background:#2563eb}
</style>
</head>
<body>
<h1><span>ARIA</span> Dashboard <button class="refresh" onclick="load()">Refresh</button></h1>
<div id="error"></div>
<div class="grid" id="stats"></div>
<div class="section">
<h2>Recent Epochs</h2>
<table id="epochs"><thead><tr><th>Epoch ID</th><th>System</th><th>Status</th><th>Records</th><th>Merkle Root</th></tr></thead><tbody></tbody></table>
</div>
<div class="section">
<h2>Recent Records</h2>
<table id="records"><thead><tr><th>Record ID</th><th>Model</th><th>Confidence</th><th>Latency</th><th>Epoch</th></tr></thead><tbody></tbody></table>
</div>
<script>
async function load(){
  try{
    const r=await fetch('/api/dashboard');
    const d=await r.json();
    document.getElementById('error').textContent='';
    // Stats cards
    const s=document.getElementById('stats');
    s.innerHTML=`
      <div class="card"><h3>Total Epochs</h3><div class="value">${d.total_epochs}</div></div>
      <div class="card"><h3>Total Records</h3><div class="value">${d.total_records}</div></div>
      <div class="card"><h3>Open Epochs</h3><div class="value green">${d.open_epochs}</div></div>
      <div class="card"><h3>Avg Confidence</h3><div class="value ${d.avg_confidence>0.8?'green':d.avg_confidence>0.6?'yellow':'red'}">${d.avg_confidence!==null?d.avg_confidence.toFixed(3):'N/A'}</div></div>
      <div class="card"><h3>Avg Latency</h3><div class="value">${d.avg_latency_ms.toFixed(0)}ms</div></div>
      <div class="card"><h3>Models</h3><div class="value">${d.models_count}</div></div>
    `;
    // Epochs table
    const eb=document.querySelector('#epochs tbody');
    eb.innerHTML=d.epochs.map(e=>`<tr>
      <td>${e.epoch_id.substring(0,20)}...</td>
      <td>${e.system_id||'-'}</td>
      <td><span class="badge ${e.close_txid?'closed':'open'}">${e.close_txid?'Closed':'Open'}</span></td>
      <td>${e.records_count}</td>
      <td style="font-family:monospace;font-size:.8rem">${(e.merkle_root||'-').substring(0,20)}</td>
    </tr>`).join('');
    // Records table
    const rb=document.querySelector('#records tbody');
    rb.innerHTML=d.records.map(r=>`<tr>
      <td style="font-family:monospace;font-size:.8rem">${r.record_id.substring(0,24)}</td>
      <td>${r.model_id}</td>
      <td class="${r.confidence>0.8?'':r.confidence>0.6?'':''}" style="color:${r.confidence!==null?(r.confidence>0.8?'var(--green)':r.confidence>0.6?'var(--yellow)':'var(--red)'):'#94a3b8'}">${r.confidence!==null?r.confidence.toFixed(3):'N/A'}</td>
      <td>${r.latency_ms}ms</td>
      <td>${r.epoch_id.substring(0,16)}</td>
    </tr>`).join('');
  }catch(e){document.getElementById('error').textContent='Error: '+e.message}
}
load();
setInterval(load,15000);
</script>
</body>
</html>
"""


def create_dashboard_app(storage: "StorageInterface") -> Any:
    """Create a FastAPI application serving the ARIA dashboard.

    Args:
        storage: ARIA StorageInterface for data access.

    Returns:
        FastAPI application instance.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="ARIA Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _DASHBOARD_HTML

    @app.get("/api/dashboard")
    async def dashboard_data() -> JSONResponse:
        data = _gather_dashboard_data(storage)
        return JSONResponse(content=data)

    @app.get("/api/epochs")
    async def list_epochs(limit: int = 50) -> JSONResponse:
        epochs = storage.list_epochs(limit=limit)
        return JSONResponse(content=[_obj_to_dict(e) for e in epochs])

    @app.get("/api/epochs/{epoch_id}/records")
    async def epoch_records(epoch_id: str) -> JSONResponse:
        records = storage.list_records_by_epoch(epoch_id)
        return JSONResponse(content=[_obj_to_dict(r) for r in records])

    return app


def serve(
    storage: "StorageInterface",
    host: str = "127.0.0.1",
    port: int = 8710,
) -> None:
    """Start the dashboard server (blocking).

    Args:
        storage: ARIA StorageInterface for data access.
        host: Bind address (default: localhost only).
        port: Port number (default: 8710).
    """
    import uvicorn

    app = create_dashboard_app(storage)
    uvicorn.run(app, host=host, port=port, log_level="info")


def _gather_dashboard_data(
    storage: "StorageInterface",
    limit: int = 50,
) -> dict[str, Any]:
    """Gather all data needed for the dashboard."""
    epochs = storage.list_epochs(limit=limit)

    all_records = []
    for e in epochs:
        all_records.extend(storage.list_records_by_epoch(e.epoch_id))

    # Stats
    confs = [r.confidence for r in all_records if r.confidence is not None]
    lats = [r.latency_ms for r in all_records]
    models = set(r.model_id for r in all_records)
    open_epochs = sum(1 for e in epochs if not e.close_txid)

    return {
        "total_epochs": len(epochs),
        "total_records": len(all_records),
        "open_epochs": open_epochs,
        "avg_confidence": round(sum(confs) / len(confs), 6) if confs else None,
        "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
        "models_count": len(models),
        "epochs": [_obj_to_dict(e) for e in epochs[:20]],
        "records": [_obj_to_dict(r) for r in all_records[:100]],
    }


def _obj_to_dict(obj: Any) -> dict:
    from dataclasses import asdict
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
