# aria-bsv — Java SDK

Java SDK for [ARIA (Auditable Real-time Inference Architecture)](https://github.com/JuanmPalencia/aria-bsv) — cryptographic accountability for AI systems, anchored on BSV blockchain via BRC-121.

[![Maven Central](https://img.shields.io/maven-central/v/io.aria.bsv/aria-bsv-java)](https://central.sonatype.com/artifact/io.aria.bsv/aria-bsv-java)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Install

**Gradle:**

```groovy
dependencies {
    implementation 'io.aria.bsv:aria-bsv-java:0.5.0'
}
```

**Maven:**

```xml
<dependency>
    <groupId>io.aria.bsv</groupId>
    <artifactId>aria-bsv-java</artifactId>
    <version>0.5.0</version>
</dependency>
```

## Quick start

```java
import io.aria.bsv.auditor.InferenceAuditor;
import io.aria.bsv.auditor.AuditorConfig;
import io.aria.bsv.auditor.InferenceRecord;

import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        AuditorConfig config = AuditorConfig.builder()
            .systemId("my-ai-system")
            .modelHashes(Map.of("gpt-4o", "sha256:abc123..."))
            .build();

        InferenceAuditor auditor = new InferenceAuditor(config);
        auditor.openEpoch();

        auditor.record(InferenceRecord.builder()
            .modelId("gpt-4o")
            .input("What is 2+2?")
            .output("4")
            .confidence(0.99)
            .build());

        var summary = auditor.closeEpoch();
        System.out.println("Epoch: " + summary.getEpochId());
        System.out.println("Merkle root: " + summary.getMerkleRoot());
    }
}
```

## Classes

### `Hasher`

Deterministic canonical JSON hashing (SHA-256), compatible with BRC-121.

```java
import io.aria.bsv.hasher.Hasher;

// Hash any serializable object (keys sorted, canonical JSON)
String hash = Hasher.hashObject(Map.of("model", "gpt-4o", "input", "hello"));
System.out.println(hash); // sha256:abc123...

// Hash raw bytes
String raw = Hasher.hashBytes("hello world".getBytes(StandardCharsets.UTF_8));

// Get canonical JSON string
String json = Hasher.canonicalJson(Map.of("b", 2, "a", 1));
// {"a":1,"b":2}
```

### `MerkleTree`

BRC-121 Merkle tree with second-preimage attack protection.

```java
import io.aria.bsv.merkle.MerkleTree;

MerkleTree tree = new MerkleTree();
tree.addLeaf("record-hash-1");
tree.addLeaf("record-hash-2");
tree.addLeaf("record-hash-3");

String root = tree.root();
List<String> proof = tree.proof(0);
boolean valid = MerkleTree.verify("record-hash-1", proof, root);
```

### `InferenceAuditor`

High-level auditor managing epoch lifecycle and Merkle accumulation.

```java
import io.aria.bsv.auditor.*;

InferenceAuditor auditor = new InferenceAuditor(
    AuditorConfig.builder()
        .systemId("production")
        .modelHashes(Map.of("llama3", "sha256:..."))
        .build()
);

auditor.openEpoch();

auditor.record(InferenceRecord.builder()
    .modelId("llama3")
    .input("Classify this email.")
    .output("not-spam")
    .confidence(0.97)
    .build());

EpochSummary summary = auditor.closeEpoch();
```

## Running tests

```bash
./gradlew test
```

## Requirements

- Java 17+
- No external dependencies (pure Java, uses `java.security.MessageDigest` for SHA-256)

## Compatibility

Hash and Merkle outputs are fully interoperable with the Python, Go, TypeScript, and Rust ARIA SDKs.

## License

MIT — Juan Manuel Palencia Osorio
