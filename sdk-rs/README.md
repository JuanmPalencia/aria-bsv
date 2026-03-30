# aria-bsv — Rust SDK

Rust SDK for [ARIA (Auditable Real-time Inference Architecture)](https://github.com/JuanmPalencia/aria-bsv) — cryptographic accountability for AI systems, anchored on BSV blockchain via BRC-121.

[![Crates.io](https://img.shields.io/crates/v/aria-bsv)](https://crates.io/crates/aria-bsv)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
aria-bsv-hasher  = "0.1"
aria-bsv-merkle  = "0.1"
aria-bsv-auditor = "0.1"
```

## Quick start

```rust
use aria_bsv_auditor::{InferenceAuditor, AuditorConfig, InferenceRecord};
use std::collections::HashMap;

fn main() {
    let mut model_hashes = HashMap::new();
    model_hashes.insert("gpt-4o".to_string(), "sha256:abc123...".to_string());

    let config = AuditorConfig {
        system_id: "my-ai-system".to_string(),
        model_hashes,
    };

    let mut auditor = InferenceAuditor::new(config);
    auditor.open_epoch().unwrap();

    auditor.record(InferenceRecord {
        model_id: "gpt-4o".to_string(),
        input: "What is 2+2?".to_string(),
        output: "4".to_string(),
        confidence: Some(0.99),
    }).unwrap();

    let summary = auditor.close_epoch().unwrap();
    println!("Epoch: {}", summary.epoch_id);
    println!("Merkle root: {}", summary.merkle_root);
}
```

## Crates

### `aria-bsv-hasher`

Deterministic canonical JSON hashing (SHA-256), compatible with BRC-121.

```rust
use aria_bsv_hasher::{hash_object, hash_bytes, canonical_json};
use serde_json::json;

// Hash any serde-serializable value
let h = hash_object(&json!({
    "model": "gpt-4o",
    "input": "hello"
})).unwrap();
println!("{}", h); // sha256:abc123...

// Hash raw bytes
let raw = hash_bytes(b"hello world");

// Get canonical JSON string
let json_str = canonical_json(&json!({"b": 2, "a": 1})).unwrap();
assert_eq!(json_str, r#"{"a":1,"b":2}"#); // keys sorted
```

### `aria-bsv-merkle`

BRC-121 Merkle tree with second-preimage attack protection.

```rust
use aria_bsv_merkle::MerkleTree;

let mut tree = MerkleTree::new();
tree.add_leaf("record-hash-1");
tree.add_leaf("record-hash-2");
tree.add_leaf("record-hash-3");

let root = tree.root().unwrap();
let proof = tree.proof(0).unwrap();
assert!(aria_bsv_merkle::verify("record-hash-1", &proof, &root));
```

### `aria-bsv-auditor`

High-level inference auditor.

```rust
use aria_bsv_auditor::{InferenceAuditor, AuditorConfig, InferenceRecord};

let mut auditor = InferenceAuditor::new(AuditorConfig {
    system_id: "prod".to_string(),
    model_hashes: [("llama3".to_string(), "sha256:...".to_string())].into(),
});

auditor.open_epoch().unwrap();
auditor.record(InferenceRecord {
    model_id: "llama3".to_string(),
    input: "Classify this email.".to_string(),
    output: "spam".to_string(),
    confidence: Some(0.97),
}).unwrap();
let summary = auditor.close_epoch().unwrap();
```

### `aria-bsv-wasm`

WebAssembly bindings — use the Rust core directly from JavaScript/TypeScript in the browser or Node.js.

```bash
wasm-pack build sdk-rs/wasm --target web
```

```javascript
import init, { hashObject, MerkleTree } from "./aria_bsv_wasm.js";

await init();

const hash = hashObject(JSON.stringify({ model: "gpt-4o", input: "hello" }));
```

## Running tests

```bash
cargo test --workspace
```

## Compatibility

Hash and Merkle outputs are fully interoperable with the Python, Go, and TypeScript ARIA SDKs.

## License

MIT — Juan Manuel Palencia Osorio
