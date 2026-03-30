# aria-bsv — Go SDK

Go SDK for [ARIA (Auditable Real-time Inference Architecture)](https://github.com/JuanmPalencia/aria-bsv) — cryptographic accountability for AI systems, anchored on BSV blockchain via BRC-121.

[![Go Reference](https://pkg.go.dev/badge/github.com/JuanmPalencia/aria-bsv/sdk-go.svg)](https://pkg.go.dev/github.com/JuanmPalencia/aria-bsv/sdk-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
go get github.com/JuanmPalencia/aria-bsv/sdk-go
```

## Quick start

```go
package main

import (
    "fmt"
    "github.com/JuanmPalencia/aria-bsv/sdk-go/auditor"
    "github.com/JuanmPalencia/aria-bsv/sdk-go/hasher"
)

func main() {
    cfg := auditor.Config{
        SystemID:   "my-ai-system",
        ModelHashes: map[string]string{
            "gpt-4o": "sha256:abc123...",
        },
    }

    a := auditor.New(cfg)

    if err := a.OpenEpoch(); err != nil {
        panic(err)
    }

    record := auditor.InferenceRecord{
        ModelID:    "gpt-4o",
        Input:      "What is 2+2?",
        Output:     "4",
        Confidence: 0.99,
    }

    if err := a.Record(record); err != nil {
        panic(err)
    }

    summary, err := a.CloseEpoch()
    if err != nil {
        panic(err)
    }

    fmt.Println("Epoch anchored:", summary.EpochID)
}
```

## Packages

### `hasher`

Deterministic canonical JSON hashing (SHA-256), compatible with the BRC-121 spec.

```go
import "github.com/JuanmPalencia/aria-bsv/sdk-go/hasher"

// Hash any serializable value
h, err := hasher.HashObject(map[string]any{
    "model": "gpt-4o",
    "input": "hello",
})
fmt.Println(h) // sha256:abc123...

// Hash raw bytes
raw := hasher.HashBytes([]byte("hello world"))
```

### `merkle`

BRC-121 Merkle tree with second-preimage attack protection (leaf prefix `0x00`, internal node prefix `0x01`).

```go
import "github.com/JuanmPalencia/aria-bsv/sdk-go/merkle"

tree := merkle.New()
tree.AddLeaf("record-hash-1")
tree.AddLeaf("record-hash-2")
tree.AddLeaf("record-hash-3")

root := tree.Root()
proof := tree.Proof(0)
valid := merkle.Verify("record-hash-1", proof, root)
```

### `auditor`

High-level inference auditor. Records AI inputs/outputs, builds Merkle trees, and prepares BRC-121 epoch payloads.

```go
import "github.com/JuanmPalencia/aria-bsv/sdk-go/auditor"

cfg := auditor.Config{
    SystemID:    "production-system",
    ModelHashes: map[string]string{"llama3": "sha256:..."},
}

a := auditor.New(cfg)
a.OpenEpoch()
a.Record(auditor.InferenceRecord{
    ModelID: "llama3",
    Input:   "Summarize this document.",
    Output:  "This document covers...",
})
summary, _ := a.CloseEpoch()
fmt.Println(summary.MerkleRoot)
```

## Running tests

```bash
go test ./...
```

With race detector:

```bash
go test ./... -race
```

Fuzz tests (requires Go 1.21+):

```bash
go test ./hasher/ -fuzz=FuzzHashBytes -fuzztime=30s
go test ./merkle/ -fuzz=FuzzAddLeaf  -fuzztime=30s
```

## Compatibility

Hashes and Merkle roots produced by this SDK are fully interoperable with the Python (`aria-bsv`) and TypeScript (`aria-bsv` npm) SDKs — all implement the same BRC-121 canonical serialization.

## License

MIT — Juan Manuel Palencia Osorio
