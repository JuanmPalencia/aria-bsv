package io.aria.bsv.auditor;

import com.google.gson.JsonObject;
import io.aria.bsv.hasher.Hasher;
import io.aria.bsv.merkle.MerkleException;
import io.aria.bsv.merkle.MerkleTree;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Records AI inference events and produces BRC-121 audit epochs.
 *
 * <p>Usage pattern:
 * <ol>
 *   <li>Construct with a system identifier.</li>
 *   <li>Call {@link #record} for every inference event.</li>
 *   <li>Call {@link #closeEpoch} to seal the batch, receive the Merkle root,
 *       and reset the auditor for the next epoch.</li>
 * </ol>
 *
 * <p>Raw inputs and outputs are never retained — only their SHA-256 hashes
 * (via {@link Hasher#hashObject}) are stored in each {@link AuditRecord}.
 */
public final class InferenceAuditor {

    private final String systemId;

    /** Current epoch identifier; reset on each {@link #closeEpoch} call. */
    private String epochId;

    /** Accumulated records for the current epoch. */
    private final List<AuditRecord> records = new ArrayList<>();

    /** 0-based sequence counter; reset on each {@link #closeEpoch} call. */
    private int sequence = 0;

    /**
     * Create a new auditor for the given system.
     *
     * @param systemId stable identifier for the AI system being audited
     *                 (e.g. {@code "my-recommender-v2"})
     */
    public InferenceAuditor(String systemId) {
        this.systemId = systemId;
        this.epochId  = UUID.randomUUID().toString();
    }

    // -------------------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------------------

    /**
     * Record a single inference event.
     *
     * <p>The method hashes {@code input} and {@code output} via
     * {@link Hasher#hashObject} (canonical JSON, then SHA-256) and stores the
     * resulting {@link AuditRecord} internally for the current epoch.
     *
     * @param modelId human-readable model identifier (e.g. {@code "gpt-4o"})
     * @param input   raw inference input as a Gson {@link JsonObject}
     * @param output  raw inference output as a Gson {@link JsonObject}
     * @return the created {@link AuditRecord} (caller may inspect or log it)
     */
    public AuditRecord record(String modelId, JsonObject input, JsonObject output) {
        String inputHash  = Hasher.hashObject(input);
        String outputHash = Hasher.hashObject(output);

        String timestamp = Instant.now().toString(); // ISO-8601 UTC
        String recordId  = String.format("rec_%s_%06d", epochId, sequence);

        AuditRecord rec = new AuditRecord(
                recordId,
                systemId,
                modelId,
                inputHash,
                outputHash,
                null,                   // confidence — not measured here
                0L,                     // latencyMs  — not measured here
                timestamp,
                sequence,
                epochId,
                Collections.emptyMap()
        );

        records.add(rec);
        sequence++;
        return rec;
    }

    // -------------------------------------------------------------------------
    // Epoch management
    // -------------------------------------------------------------------------

    /**
     * Close the current epoch.
     *
     * <p>Computes a {@link MerkleTree} over the SHA-256 hashes of all records
     * accumulated since the last close (or since construction), seals the epoch
     * summary, then resets the auditor so a new epoch can begin.
     *
     * <p>Each Merkle leaf is the UTF-8 encoding of the hex digest produced by
     * {@link Hasher#hashObject} on the record's canonical map.
     *
     * @return {@link EpochSummary} containing the epoch identifier, system
     *         identifier, record count, and Merkle root
     * @throws MerkleException if no records have been added to the current epoch
     */
    public EpochSummary closeEpoch() throws MerkleException {
        MerkleTree tree = new MerkleTree();
        for (AuditRecord rec : records) {
            String recHash = Hasher.hashObject(rec.toCanonicalMap());
            tree.addLeaf(recHash.getBytes(StandardCharsets.UTF_8));
        }

        String merkleRoot = tree.root(); // throws MerkleException if no records

        EpochSummary summary = new EpochSummary(
                epochId,
                systemId,
                records.size(),
                merkleRoot
        );

        // Reset for the next epoch
        epochId  = UUID.randomUUID().toString();
        records.clear();
        sequence = 0;

        return summary;
    }

    // -------------------------------------------------------------------------
    // EpochSummary
    // -------------------------------------------------------------------------

    /**
     * Immutable summary returned by {@link InferenceAuditor#closeEpoch}.
     *
     * <p>The {@code merkleRoot} is the root of a BRC-121 Merkle tree built from
     * the SHA-256 hashes of all records in the epoch.
     */
    public static final class EpochSummary {

        /** The UUID string that identified this epoch. */
        private final String epochId;

        /** The ARIA system identifier. */
        private final String systemId;

        /** Number of inference records in this epoch. */
        private final int recordCount;

        /**
         * Lowercase hex Merkle root (64 characters) computed over all record
         * hashes in this epoch.
         */
        private final String merkleRoot;

        public EpochSummary(String epochId, String systemId, int recordCount, String merkleRoot) {
            this.epochId     = epochId;
            this.systemId    = systemId;
            this.recordCount = recordCount;
            this.merkleRoot  = merkleRoot;
        }

        public String getEpochId()     { return epochId; }
        public String getSystemId()    { return systemId; }
        public int    getRecordCount() { return recordCount; }
        public String getMerkleRoot()  { return merkleRoot; }

        @Override
        public String toString() {
            return "EpochSummary{epochId='" + epochId
                    + "', systemId='" + systemId
                    + "', recordCount=" + recordCount
                    + ", merkleRoot='" + merkleRoot + "'}";
        }
    }
}
