# aria-bsv — .NET SDK

C# / .NET 8 SDK for [ARIA (Auditable Real-time Inference Architecture)](https://github.com/JuanmPalencia/aria-bsv) — cryptographic accountability for AI systems, anchored on BSV blockchain via BRC-121.

[![NuGet](https://img.shields.io/nuget/v/AriaBsv.Core)](https://www.nuget.org/packages/AriaBsv.Core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
dotnet add package AriaBsv.Core
```

## Quick start

```csharp
using AriaBsv.Core;

var config = new AuditorConfig
{
    SystemId = "my-ai-system",
    ModelHashes = new Dictionary<string, string>
    {
        ["gpt-4o"] = "sha256:abc123..."
    }
};

var auditor = new InferenceAuditor(config);
await auditor.OpenEpochAsync();

await auditor.RecordAsync(new AuditRecord
{
    ModelId    = "gpt-4o",
    Input      = "What is 2+2?",
    Output     = "4",
    Confidence = 0.99
});

var summary = await auditor.CloseEpochAsync();
Console.WriteLine($"Epoch: {summary.EpochId}");
Console.WriteLine($"Merkle root: {summary.MerkleRoot}");
```

## Classes

### `Hasher`

Deterministic canonical JSON hashing (SHA-256), compatible with BRC-121.

```csharp
using AriaBsv.Core;

// Hash any object (keys sorted, canonical JSON)
string hash = Hasher.HashObject(new { model = "gpt-4o", input = "hello" });
Console.WriteLine(hash); // sha256:abc123...

// Hash raw bytes
string raw = Hasher.HashBytes(Encoding.UTF8.GetBytes("hello world"));

// Get canonical JSON string
string json = Hasher.CanonicalJson(new { b = 2, a = 1 });
// {"a":1,"b":2}
```

### `MerkleTree`

BRC-121 Merkle tree with second-preimage attack protection (leaf prefix `0x00`, internal node prefix `0x01`).

```csharp
using AriaBsv.Core;

var tree = new MerkleTree();
tree.AddLeaf("record-hash-1");
tree.AddLeaf("record-hash-2");
tree.AddLeaf("record-hash-3");

string root = tree.Root();
bool valid = MerkleTree.Verify("record-hash-1", tree.Proof(0), root);
```

### `InferenceAuditor`

High-level auditor that manages epoch lifecycle and Merkle accumulation.

```csharp
using AriaBsv.Core;

var auditor = new InferenceAuditor(new AuditorConfig
{
    SystemId     = "production",
    ModelHashes  = new() { ["llama3"] = "sha256:..." }
});

await auditor.OpenEpochAsync();

await auditor.RecordAsync(new AuditRecord
{
    ModelId    = "llama3",
    Input      = "Summarize this document.",
    Output     = "This document covers...",
    Confidence = 0.95
});

var summary = await auditor.CloseEpochAsync();
```

### `AuditRecord`

```csharp
var record = new AuditRecord
{
    RecordId   = Guid.NewGuid().ToString(),  // auto-generated if omitted
    ModelId    = "gpt-4o",
    Input      = "user prompt",
    Output     = "model response",
    Confidence = 0.98,
    Timestamp  = DateTimeOffset.UtcNow
};
```

## Running tests

```bash
dotnet test
```

## Requirements

- .NET 8+
- `System.Text.Json` (included)

## Compatibility

Hash and Merkle outputs are fully interoperable with the Python, Go, TypeScript, and Rust ARIA SDKs.

## License

MIT — Juan Manuel Palencia Osorio
