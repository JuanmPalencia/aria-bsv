# ARIA — Auditable Real-time Inference Architecture

> *Cryptographic accountability for production AI systems. No blockchain knowledge required.*

[![CI](https://github.com/JuanmPalencia/aria-bsv/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanmPalencia/aria-bsv/actions)
[![PyPI version](https://img.shields.io/pypi/v/aria-bsv.svg)](https://pypi.org/project/aria-bsv/)
[![Tests](https://img.shields.io/badge/tests-1578%20passing-brightgreen.svg)](https://github.com/JuanmPalencia/aria-bsv/actions)
[![License: Open BSV](https://img.shields.io/badge/License-Open%20BSV-blue.svg)](LICENSE)
[![BRC-121](https://img.shields.io/badge/Standard-BRC--121-orange.svg)](https://github.com/bitcoin-sv/BRCs/pull/129)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Art.%2012%20compliant-green.svg)](https://github.com/JuanmPalencia/aria-bsv)

---

## What is ARIA?

ARIA is an open protocol and Python SDK that makes any AI system independently auditable — meaning a regulator, a judge, or a citizen can verify what an AI decided, when, and with which model, **without asking the system operator for access**.

It works by publishing cryptographic commitments to the [BSV blockchain](https://bitcoinsv.io) before and after each inference cycle. These commitments make it impossible to alter or fabricate AI decision records after the fact.

ARIA is the reference implementation of **[BRC-121](brc/0121.md)** — the first open standard for AI inference accountability on BSV.

---

## The problem it solves

When an AI system makes a critical decision — dispatching an ambulance, denying a loan, flagging a security threat — three problems exist today with no standard technical solution:

1. **No proof of which model decided** — the operator can claim any model version
2. **Logs are alterable** — any database can be modified before an audit
3. **Verification requires the operator** — regulators and citizens cannot verify independently

The [EU AI Act (Regulation 2024/1689)](docs/03-eu-ai-act-compliance.md) requires solving all three for high-risk AI systems from 2026. ARIA solves all three.

---

## Quick start

```bash
pip install aria-bsv
```

```python
from aria import InferenceAuditor, AuditConfig

auditor = InferenceAuditor(AuditConfig(
    system_id = "my-ai-system-v1",
    bsv_key   = os.environ["BSV_WIF"],
    storage   = "sqlite:///aria.db",
))

# Add to any existing function — zero changes to your model code
@auditor.track(model="my_classifier")
def classify(input_data: dict) -> dict:
    return model.predict(input_data)        # your code, unchanged

# Verify any decision — works without the operator
result = auditor.verify(epoch_open_txid="a3f9...")
print(result.valid)        # True
print(result.tampered)     # False
print(result.decided_at)   # "2026-03-22T14:32:01Z"
```

That is everything the operator needs to do. Verification is available to anyone with the transaction ID.

---

## How it works

ARIA uses a **Pre-Commitment Protocol**: before any AI model executes, a hash of the system state and model versions is published to BSV. After execution, the Merkle root of all inference records is published, linked to the pre-commitment.

```
BSV blockchain:
  ┌─────────────────┐     ┌──────────────────┐
  │  EPOCH OPEN     │────▶│  EPOCH CLOSE     │
  │  model_hashes   │     │  merkle_root     │
  │  state_hash     │     │  records_count   │
  │  timestamp      │     │  prev_txid       │
  └─────────────────┘     └──────────────────┘
    published BEFORE          published AFTER
    AI executes               AI executes
```

This makes backdating and model substitution cryptographically impossible.

**Cost**: approximately $0.0001 per epoch on BSV mainnet. A system running every 1.5 seconds costs ~$2.10/year.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Layer 4 — AI Systems Registry                       │
│  Public API. Searchable. Verifiable track records.   │
├──────────────────────────────────────────────────────┤
│  Layer 3 — Verification Portal                       │
│  Paste a txid. See what the AI decided. No crypto.   │
├──────────────────────────────────────────────────────┤
│  Layer 2 — BRC-121 Standard                          │
│  Open protocol. Any language can implement it.       │
├──────────────────────────────────────────────────────┤
│  Layer 1 — Python SDK (aria-bsv)                     │
│  5 lines. No blockchain knowledge required.          │
└──────────────────────────────────────────────────────┘
```

---

## EU AI Act compliance

ARIA covers the following requirements of Regulation (EU) 2024/1689 for high-risk AI systems (Annex III):

| Article | Requirement | ARIA component |
|---------|-------------|----------------|
| Art. 12.1 | Automatic logging of events | `@audit.track` decorator |
| Art. 12.2 | Lifecycle traceability | Epoch chain (OPEN→CLOSE) |
| Art. 12.3 | Independent post-hoc verification | `verify.py` + Portal |
| Art. 9 | Continuous risk management | AI Systems Registry |
| Art. 13 | Transparency to users | Public verification portal |
| Art. 11 | Pre-deployment technical documentation | BRC-121 spec |

Full mapping: [docs/03-eu-ai-act-compliance.md](docs/03-eu-ai-act-compliance.md)

---

## Production references

| System | Domain | Status |
|--------|--------|--------|
| [KAIROS CDS](examples/kairos/) | Emergency medical dispatch — Madrid | Reference implementation |
| [Urban VS](examples/urban-vs/) | Computer vision traffic forensics | Reference implementation |

---

## Documentation

| Document | Description |
|----------|-------------|
| [BRC-121 spec](https://github.com/bitcoin-sv/BRCs/pull/129) | Formal protocol specification |
| [examples/minimal/](examples/minimal/) | Quickstart — 5 lines, zero config |
| [examples/kairos/](examples/kairos/) | Production example — KAIROS CDS integration |

---

## Repository structure

```
aria-bsv/
├── aria/                  # Python package
│   ├── core/              # Cryptographic primitives (hasher, merkle, epoch, record)
│   ├── wallet/            # BSV signing abstraction
│   ├── broadcaster/       # ARC transaction broadcaster
│   ├── storage/           # SQLite / PostgreSQL
│   ├── integrations/      # FastAPI / Django / Flask
│   ├── verify.py          # Independent verification engine
│   └── auditor.py         # Main public API
├── portal/                # Verification web app (FastAPI + React)
├── registry/              # AI Systems Registry API
├── brc/0121.md            # BRC-121 specification
├── docs/                  # Full documentation (9 documents)
├── examples/              # kairos/, urban-vs/, minimal/
├── tests/                 # unit/, integration/, fixtures/
├── deploy/                # docker-compose, helm, k8s
└── paper/                 # arXiv preprint
```

---

## Security

ARIA is designed around a zero-trust model:
- The SDK **never handles raw private keys** in logs or error messages
- Only **hashes go on-chain**, never raw inference data
- **Verification never requires contacting the operator** — it uses BSV directly
- Full threat model: [docs/04-security-model.md](docs/04-security-model.md)

To report a security vulnerability: see [SECURITY.md](SECURITY.md)

---

## Status

ARIA v0.4.0 — **all phases complete**. 1578 tests passing. Mainnet-verified.

BRC-121 PR open: [bitcoin-sv/BRCs#129](https://github.com/bitcoin-sv/BRCs/pull/129)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

This project proposes [BRC-121](brc/0121.md) as an open standard to the BSV ecosystem. Implementations in other languages that follow the spec are welcome and will be listed here.

---

## License

[Open BSV License](LICENSE) — consistent with the BSV ecosystem.
