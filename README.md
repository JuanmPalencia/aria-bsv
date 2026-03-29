# ARIA — Auditable Real-time Inference Architecture

> *Cryptographic accountability for production AI systems. No blockchain knowledge required.*

[![CI](https://github.com/JuanmPalencia/aria-bsv/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanmPalencia/aria-bsv/actions)
[![PyPI version](https://img.shields.io/pypi/v/aria-bsv.svg)](https://pypi.org/project/aria-bsv/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2440%20passing-brightgreen.svg)](https://github.com/JuanmPalencia/aria-bsv/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![BRC-121](https://img.shields.io/badge/Standard-BRC--121-orange.svg)](https://github.com/bitcoin-sv/BRCs/pull/129)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Art.%2012%20compliant-green.svg)](https://github.com/JuanmPalencia/aria-bsv)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-blue.svg)](https://peps.python.org/pep-0561/)

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

### Zero-config mode (no blockchain knowledge needed)

```python
from aria.quick import ARIAQuick

with ARIAQuick("my-ai-system") as aria:
    aria.record("gpt-4", {"prompt": "hello"}, {"text": "hi"}, confidence=0.95)
    summary = aria.close()
    print(summary)
```

### Full control mode

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

### CLI

```bash
aria init                          # interactive project setup
aria selftest                      # verify SDK installation
aria estimate --records 10000      # cost estimation
aria export my-system --format pdf # compliance report
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

## Features

### Core SDK
- **Pre-Commitment Protocol** — epoch OPEN/CLOSE with Merkle trees, second-preimage protection (RFC 6962)
- **Independent verification** — verify any decision from the blockchain without operator access
- **Offline mode** — audit locally, sync to BSV later with `OfflineAuditor`
- **Pipeline auditing** — trace multi-model inference chains with `PipelineAuditor`
- **Retry queue** — persistent dead-letter queue with exponential backoff for failed broadcasts
- **Cost estimator** — estimate BSV costs before going to production
- **Declarative config** — `aria.toml` project files with environment variable overrides
- **24+ CLI commands** — `aria init`, `aria selftest`, `aria estimate`, `aria export`, `aria sync`, and more

### AI Framework Integrations (20+)
OpenAI, Anthropic, Azure OpenAI, Google Gemini, Cohere, Mistral, Ollama, vLLM, HuggingFace Transformers, LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, SageMaker, Vertex AI, MLflow, Weights & Biases

### Web Framework Integrations
FastAPI middleware, Django middleware, Flask middleware

### Regulatory & Compliance
- **EU AI Act** compliance checker (Articles 9–14, 72)
- **GDPR** PII masking, pseudonymisation, consent management
- **Model cards** auto-generated from epoch data
- **Regulatory export** — PDF/JSON reports for EU AI Act, NIST AI RMF, ISO 42001

### Infrastructure
- **Storage**: SQLite (zero-config), PostgreSQL (production)
- **Events**: In-memory bus, Redis Pub/Sub for distributed deployments
- **Metrics**: Prometheus, OpenTelemetry (spans + meters)
- **SIEM**: JSON, CEF, LEEF export formats
- **Multi-tenancy**: tenant isolation with SHA-256 API keys
- **HSM**: hardware security module abstraction (LocalHSM, AWS KMS stub)
- **Webhooks**: HMAC-SHA256 verified webhook receiver with FastAPI router
- **Jupyter**: `%%aria` cell magic for notebook auditing

### Multi-chain & Federation
- **BSV** (primary) — ~$0.0001/epoch
- **Ethereum** anchor (optional)
- **Nostr** anchor (NIP-01)
- **Federation hub** for multi-organization deployments

### Developer Experience
- **Zero blockchain knowledge required** — `ARIAQuick` context manager
- **`aria init`** — interactive project scaffolding
- **`aria selftest`** — verify installation
- **GitHub Action** — reusable `verify-epoch` action for CI/CD
- **Cross-SDK test vectors** — shared `vectors.json` for Go/TypeScript SDK compatibility
- **Docker + Compose** — production-ready container stack
- **Helm chart + Terraform** — Kubernetes & cloud deployment

### Multi-language SDKs
- **Python** (reference implementation) — full-featured
- **TypeScript** (`sdk-ts/`) — hasher, merkle, auditor, verifier, contracts, overlay, streaming, SPV
- **Go** (`sdk-go/`) — hasher, merkle, auditor

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
| [Whitepaper](docs/01-whitepaper.md) | Technical overview and motivation |
| [Protocol spec](docs/02-protocol-spec.md) | BRC-121 protocol details |
| [EU AI Act mapping](docs/03-eu-ai-act-compliance.md) | Article-by-article compliance mapping |
| [Security model](docs/04-security-model.md) | Zero-trust security architecture |
| [Threat model](docs/05-threat-model.md) | Attack vectors and mitigations |
| [Performance](docs/06-performance.md) | Benchmarks and optimization |
| [Getting started](docs/07-getting-started.md) | Installation and first steps |
| [API reference](docs/08-api-reference.md) | Full Python API documentation |
| [Implementation guide](docs/09-implementation-guide.md) | Guide for other language implementations |
| [BRC-121 spec](https://github.com/bitcoin-sv/BRCs/pull/129) | Formal protocol specification |
| [examples/minimal/](examples/minimal/) | Quickstart — 5 lines, zero config |
| [examples/kairos/](examples/kairos/) | Production example — KAIROS CDS integration |

---

## Repository structure

```
aria-bsv/
├── aria/                  # Python SDK package
│   ├── core/              # Cryptographic primitives (hasher, merkle, epoch, record)
│   ├── wallet/            # BSV signing abstraction (DirectWallet, BRC-100)
│   ├── broadcaster/       # ARC transaction broadcaster with retry
│   ├── storage/           # SQLite (zero-config) / PostgreSQL (production)
│   ├── integrations/      # 20+ AI framework integrations
│   ├── contracts/         # On-chain contract wrappers
│   ├── telemetry/         # OpenTelemetry + Prometheus metrics
│   ├── zk/                # Zero-knowledge proof layer (optional)
│   ├── auditor.py         # Main public API
│   ├── verify.py          # Independent verification engine
│   ├── pipeline.py        # Multi-model pipeline auditing
│   ├── offline.py         # Offline mode with deferred sync
│   ├── quick.py           # Zero-config entry point (ARIAQuick)
│   ├── cli.py             # 24+ CLI commands
│   └── ...                # 40+ modules total
├── sdk-ts/                # TypeScript SDK
├── sdk-go/                # Go SDK
├── portal/                # Verification web app (FastAPI + React)
├── registry/              # AI Systems Registry API
├── brc/0121.md            # BRC-121 specification
├── docs/                  # 9 technical documents
├── examples/              # kairos/, minimal/
├── tests/                 # 2440 tests (unit, chaos, perf, property, cross-sdk)
├── deploy/                # Docker, Helm, Terraform, Grafana, Prometheus
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

## Installation

```bash
# Core SDK (SQLite storage, no blockchain)
pip install aria-bsv

# With BSV anchoring
pip install aria-bsv[bsv]

# With CLI
pip install aria-bsv[cli]

# With AI framework integrations
pip install aria-bsv[openai]           # OpenAI
pip install aria-bsv[anthropic]        # Anthropic
pip install aria-bsv[huggingface]      # HuggingFace
pip install aria-bsv[langchain]        # LangChain

# Full installation (all optional dependencies)
pip install aria-bsv[all]
```

---

## Status

ARIA v0.5.0 — **all phases complete**. 2440 tests passing. Mainnet-verified.

BRC-121 PR open: [bitcoin-sv/BRCs#129](https://github.com/bitcoin-sv/BRCs/pull/129)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and contribution guidelines.

This project proposes [BRC-121](brc/0121.md) as an open standard to the BSV ecosystem. Implementations in other languages that follow the spec are welcome and will be listed here.

Please note that this project follows a [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

[MIT License](LICENSE)
