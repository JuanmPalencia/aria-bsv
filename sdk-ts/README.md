# aria-bsv — TypeScript SDK

TypeScript/JavaScript SDK for [ARIA (Auditable Real-time Inference Architecture)](https://github.com/JuanmPalencia/aria-bsv) — cryptographic accountability for AI systems, anchored on BSV blockchain via [BRC-121](https://github.com/JuanmPalencia/aria-bsv).

[![npm](https://img.shields.io/npm/v/aria-bsv)](https://www.npmjs.com/package/aria-bsv)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
npm install aria-bsv
```

## Quick start

```typescript
import { InferenceAuditor, ARCBroadcaster } from "aria-bsv";

const broadcaster = new ARCBroadcaster("https://arc.taal.com");

const auditor = new InferenceAuditor({
  systemId: "my-ai-system",
  modelHashes: { "gpt-4o": "sha256:abc123..." },
  bsvKey: process.env.BSV_WIF!,
}, broadcaster);

await auditor.openEpoch();

await auditor.record({
  modelId: "gpt-4o",
  input: "What is 2+2?",
  output: "4",
  confidence: 0.99,
});

const summary = await auditor.closeEpoch();
console.log("Epoch anchored:", summary.epochId);
console.log("Close TXID:", summary.closeTxid);
```

## Modules

### Hashing

```typescript
import { hashObject, hashObjectSync, canonicalJson } from "aria-bsv";

// Deterministic canonical JSON hash (SHA-256)
const hash = await hashObject({ model: "gpt-4o", input: "hello" });

// Synchronous version
const hashSync = hashObjectSync({ model: "gpt-4o", input: "hello" });
```

### Merkle tree

```typescript
import { MerkleTree, generateMerkleProof, verifyMerkleProof } from "aria-bsv";

const tree = new MerkleTree();
tree.addLeaf("record1-hash");
tree.addLeaf("record2-hash");

const root = await tree.getRoot();
const proof = generateMerkleProof(tree, 0);
const valid = verifyMerkleProof("record1-hash", proof, root);
```

### Epoch verification

```typescript
import { verifyEpoch } from "aria-bsv";

const result = await verifyEpoch({
  openTxid: "0c2e80c23d17f423...",
  closeTxid: "a1b2c3d4e5f6...",
  arcApiUrl: "https://arc.taal.com",
});

console.log(result.valid);         // true
console.log(result.recordsCount);  // 42
console.log(result.merkleRoot);    // sha256:...
```

### Dataset anchoring

```typescript
import { DatasetAnchorer } from "aria-bsv";

const anchorer = new DatasetAnchorer({ systemId: "my-system" });

// Hash a file buffer
const anchor = await anchorer.anchorBuffer(fileBuffer, {
  mimeType: "text/csv",
  fileName: "training-data.csv",
});

console.log(anchor.contentHash);  // sha256:8faba755...
```

### Streaming auditor (OpenAI / Anthropic)

```typescript
import { auditOpenAIStream } from "aria-bsv";
import OpenAI from "openai";

const client = new OpenAI();

const stream = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
  stream: true,
});

const record = await auditOpenAIStream(stream, {
  systemId: "my-system",
  modelId: "gpt-4o",
});
```

### Multi-chain broadcasting

Supports BSV (ARC), Ethereum, and Nostr relay publishing out of the box:

```typescript
import { ARCBroadcaster } from "aria-bsv";

const broadcaster = new ARCBroadcaster("https://arc.taal.com", {
  apiKey: process.env.ARC_API_KEY,
});
```

### BRC-121 on-chain contracts

```typescript
import { ARIAEpochContract, ARIATimelockContract } from "aria-bsv";

const contract = new ARIAEpochContract({
  merkleRoot: "sha256:abc...",
  operatorPkh: "deadbeef...",
  epochId: "ep_001",
});

const result = contract.verify({
  sigHex: "3045...",
  pubKeyHex: "02abc...",
  closeRoot: "sha256:abc...",
});
```

## SPV verification

```typescript
import { verifySpvProof } from "aria-bsv";

const result = verifySpvProof({
  txid: "abc123...",
  proof: { /* merkle branch */ },
  header: { /* block header */ },
});
```

## EU AI Act compliance

```typescript
import { EUAIActRisk } from "aria-bsv";

// Risk levels: UNACCEPTABLE | HIGH | LIMITED | MINIMAL
const risk = EUAIActRisk.HIGH;
```

## Contributing

See the [main repository](https://github.com/JuanmPalencia/aria-bsv) for the full monorepo including Python SDK, Go SDK, Rust SDK, and the BRC-121 specification.

## License

MIT — Juan Manuel Palencia Osorio
