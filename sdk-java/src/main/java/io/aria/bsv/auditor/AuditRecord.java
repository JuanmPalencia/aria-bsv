package io.aria.bsv.auditor;

import java.util.Collections;
import java.util.Map;

/**
 * A single recorded AI inference event anchored to the ARIA BRC-121 protocol.
 *
 * <p>Fields mirror the Python, TypeScript, Go, and Rust SDK {@code AuditRecord}
 * structures for cross-SDK compatibility. Raw input and output are never stored
 * on-chain; only their SHA-256 hashes appear here.
 */
public final class AuditRecord {

    /** Unique identifier: {@code "rec_{epochId}_{seq:06}"}. */
    private final String recordId;

    /** The ARIA system identifier supplied at auditor construction time. */
    private final String systemId;

    /** Model identifier (e.g. {@code "gpt-4o"}). */
    private final String modelId;

    /** SHA-256 hex hash of the canonical JSON of the raw input. */
    private final String inputHash;

    /** SHA-256 hex hash of the canonical JSON of the raw output. */
    private final String outputHash;

    /**
     * Model confidence in {@code [0.0, 1.0]}, or {@code null} if not available.
     */
    private final Double confidence;

    /** Wall-clock inference duration in milliseconds. */
    private final long latencyMs;

    /** ISO-8601 UTC timestamp of the inference (e.g. {@code "2025-01-01T00:00:00Z"}). */
    private final String timestamp;

    /** 0-based sequence number within the lifetime of the auditor. */
    private final int sequence;

    /** Epoch identifier this record belongs to. */
    private final String epochId;

    /** Additional caller-supplied metadata (may be empty, never null). */
    private final Map<String, Object> metadata;

    public AuditRecord(
            String recordId,
            String systemId,
            String modelId,
            String inputHash,
            String outputHash,
            Double confidence,
            long latencyMs,
            String timestamp,
            int sequence,
            String epochId,
            Map<String, Object> metadata) {
        this.recordId  = recordId;
        this.systemId  = systemId;
        this.modelId   = modelId;
        this.inputHash  = inputHash;
        this.outputHash = outputHash;
        this.confidence = confidence;
        this.latencyMs  = latencyMs;
        this.timestamp  = timestamp;
        this.sequence   = sequence;
        this.epochId    = epochId;
        this.metadata   = metadata == null ? Collections.emptyMap() : metadata;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public String getRecordId()              { return recordId; }
    public String getSystemId()              { return systemId; }
    public String getModelId()               { return modelId; }
    public String getInputHash()             { return inputHash; }
    public String getOutputHash()            { return outputHash; }
    public Double getConfidence()            { return confidence; }
    public long   getLatencyMs()             { return latencyMs; }
    public String getTimestamp()             { return timestamp; }
    public int    getSequence()              { return sequence; }
    public String getEpochId()               { return epochId; }
    public Map<String, Object> getMetadata() { return metadata; }

    // -------------------------------------------------------------------------
    // Canonical map for Merkle hashing
    // -------------------------------------------------------------------------

    /**
     * Return a {@code Map<String,Object>} with snake_case keys that matches the
     * canonical serialization used by all other ARIA SDKs.  This map is passed
     * to {@link io.aria.bsv.hasher.Hasher#hashObject(Object)} when computing the
     * per-record leaf hash for the flush Merkle tree.
     */
    public Map<String, Object> toCanonicalMap() {
        java.util.LinkedHashMap<String, Object> m = new java.util.LinkedHashMap<>();
        m.put("confidence",  confidence);
        m.put("epoch_id",    epochId);
        m.put("input_hash",  inputHash);
        m.put("latency_ms",  latencyMs);
        m.put("metadata",    metadata);
        m.put("model_id",    modelId);
        m.put("output_hash", outputHash);
        m.put("record_id",   recordId);
        m.put("sequence",    sequence);
        m.put("system_id",   systemId);
        m.put("timestamp",   timestamp);
        return m;
    }

    @Override
    public String toString() {
        return "AuditRecord{recordId='" + recordId
                + "', systemId='" + systemId
                + "', modelId='" + modelId
                + "', sequence=" + sequence
                + ", epochId='" + epochId + "'}";
    }
}
