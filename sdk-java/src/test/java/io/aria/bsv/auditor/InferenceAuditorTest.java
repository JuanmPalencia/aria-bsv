package io.aria.bsv.auditor;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link InferenceAuditor}.
 *
 * <p>Tests cover:
 * <ul>
 *   <li>record() ID format and validation</li>
 *   <li>flush() Merkle root and record list</li>
 *   <li>Auto-flush at batch size</li>
 *   <li>reset() behaviour</li>
 *   <li>stats() snapshot</li>
 *   <li>onFlush callback</li>
 *   <li>recordWithMeta() metadata attachment</li>
 * </ul>
 */
class InferenceAuditorTest {

    // =========================================================================
    // record() — basic behaviour
    // =========================================================================

    @Test
    void record_returnsNonEmptyId() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        String id = a.record("gpt-4", "hello", "world", 0.9, 42L);
        assertNotNull(id);
        assertFalse(id.isEmpty());
    }

    @Test
    void record_idStartsWithRec() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        String id = a.record("m", "in", "out", null, 10L);
        assertTrue(id.startsWith("rec_"), "Record ID must start with 'rec_': " + id);
    }

    @Test
    void record_storesRecordInBuffer() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        assertEquals(1, a.stats().getBuffered());
    }

    @Test
    void record_incrementsTotalRecorded() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        a.record("m", "in", "out", null, 0L);
        assertEquals(2L, a.stats().getTotalRecorded());
    }

    // =========================================================================
    // record() — input validation
    // =========================================================================

    @Test
    void record_invalidConfidenceTooHigh_throws() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertThrows(IllegalArgumentException.class,
                () -> a.record("m", "in", "out", 1.5, 0L),
                "confidence > 1 must throw IllegalArgumentException");
    }

    @Test
    void record_invalidConfidenceNegative_throws() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertThrows(IllegalArgumentException.class,
                () -> a.record("m", "in", "out", -0.1, 0L),
                "confidence < 0 must throw IllegalArgumentException");
    }

    @Test
    void record_validConfidenceBoundaryValues() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertDoesNotThrow(() -> a.record("m", "in", "out", 0.0, 0L));
        assertDoesNotThrow(() -> a.record("m", "in", "out", 1.0, 0L));
    }

    @Test
    void record_nullConfidenceAccepted() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertDoesNotThrow(() -> a.record("m", "in", "out", null, 0L));
    }

    @Test
    void record_negativeLatency_throws() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertThrows(IllegalArgumentException.class,
                () -> a.record("m", "in", "out", null, -1L),
                "negative latencyMs must throw IllegalArgumentException");
    }

    @Test
    void record_inputHashIs64HexChars() {
        // Access via flush to inspect the resulting record
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", Map.of("prompt", "test"), "response", 0.8, 10L);
        InferenceAuditor.FlushResult result = a.flush();
        String inputHash = result.getRecords().get(0).getInputHash();
        assertEquals(64, inputHash.length(), "inputHash must be 64 hex chars");
        assertTrue(inputHash.matches("[0-9a-f]+"), "inputHash must be lowercase hex");
    }

    @Test
    void record_outputHashIs64HexChars() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "input", Map.of("result", 42), null, 5L);
        InferenceAuditor.FlushResult result = a.flush();
        String outputHash = result.getRecords().get(0).getOutputHash();
        assertEquals(64, outputHash.length());
        assertTrue(outputHash.matches("[0-9a-f]+"));
    }

    // =========================================================================
    // flush()
    // =========================================================================

    @Test
    void flush_returnsMerkleRootWith64Chars() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", 0.9, 10L);
        InferenceAuditor.FlushResult result = a.flush();
        assertNotNull(result.getMerkleRoot());
        assertEquals(64, result.getMerkleRoot().length(),
                "Merkle root must be 64 hex chars");
    }

    @Test
    void flush_returnsCorrectRecords() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in1", "out1", null, 1L);
        a.record("m", "in2", "out2", null, 2L);
        InferenceAuditor.FlushResult result = a.flush();
        assertEquals(2, result.getRecords().size());
    }

    @Test
    void flush_clearsBuffer() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        a.flush();
        assertEquals(0, a.stats().getBuffered());
    }

    @Test
    void flush_emptyBuffer_returnsNullRoot() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        InferenceAuditor.FlushResult result = a.flush();
        assertNull(result.getMerkleRoot(),
                "Flushing empty buffer must return null Merkle root");
        assertTrue(result.getRecords().isEmpty());
    }

    @Test
    void flush_rotatesEpochId() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        String ep1 = a.epochId();
        a.record("m", "in", "out", null, 0L);
        a.flush();
        String ep2 = a.epochId();
        assertNotEquals(ep1, ep2, "flush() must rotate the epoch ID");
    }

    @Test
    void flush_merkleRootIs64LowercaseHex() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", Map.of("x", 1), Map.of("y", 2), 0.7, 15L);
        String root = a.flush().getMerkleRoot();
        assertNotNull(root);
        assertEquals(64, root.length());
        assertEquals(root, root.toLowerCase(java.util.Locale.ROOT));
    }

    // =========================================================================
    // Auto-flush at batch size
    // =========================================================================

    @Test
    void autoFlush_atBatchSize() {
        InferenceAuditor a = new InferenceAuditor("sys", 2);
        a.record("m", "in", "out", null, 0L);
        assertEquals(1, a.stats().getBuffered());
        a.record("m", "in2", "out2", null, 0L); // triggers auto-flush
        assertEquals(0, a.stats().getBuffered(),
                "Buffer must be empty after auto-flush at batchSize");
        assertEquals(2L, a.stats().getTotalRecorded());
    }

    @Test
    void autoFlush_callsOnFlushCallback() {
        AtomicInteger callCount = new AtomicInteger(0);
        InferenceAuditor a = new InferenceAuditor("sys", 2);
        a.setOnFlush((root, records) -> callCount.incrementAndGet());

        a.record("m", "in", "out", null, 0L);
        a.record("m", "in2", "out2", null, 0L); // auto-flush
        assertEquals(1, callCount.get());
    }

    // =========================================================================
    // reset()
    // =========================================================================

    @Test
    void reset_clearsBuffer() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        a.record("m", "in", "out", null, 0L);
        a.reset();
        assertEquals(0, a.stats().getBuffered());
    }

    @Test
    void reset_rotatesEpochId() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        String ep1 = a.epochId();
        a.reset();
        assertNotEquals(ep1, a.epochId(), "reset() must rotate the epoch ID");
    }

    @Test
    void reset_doesNotCallOnFlush() {
        AtomicInteger callCount = new AtomicInteger(0);
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.setOnFlush((root, records) -> callCount.incrementAndGet());
        a.record("m", "in", "out", null, 0L);
        a.reset();
        assertEquals(0, callCount.get(), "reset() must NOT invoke the flush callback");
    }

    @Test
    void reset_preservesTotalRecorded() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        a.reset();
        assertEquals(1L, a.stats().getTotalRecorded(),
                "totalRecorded must be preserved across reset()");
    }

    // =========================================================================
    // stats()
    // =========================================================================

    @Test
    void stats_initialState() {
        InferenceAuditor a = new InferenceAuditor("my-system", 50);
        InferenceAuditor.AuditStats s = a.stats();
        assertEquals(0, s.getBuffered());
        assertEquals(0L, s.getTotalRecorded());
        assertEquals(50, s.getBatchSize());
        assertEquals("my-system", s.getSystemId());
    }

    @Test
    void stats_afterRecordsAndFlush() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.record("m", "in", "out", null, 0L);
        a.flush();
        a.record("m", "in", "out", null, 0L);

        InferenceAuditor.AuditStats s = a.stats();
        assertEquals(1, s.getBuffered());
        assertEquals(2L, s.getTotalRecorded(),
                "totalRecorded includes both flushed and buffered records");
    }

    // =========================================================================
    // onFlush callback
    // =========================================================================

    @Test
    void onFlush_calledWithMerkleRootAndRecords() {
        AtomicReference<String> capturedRoot = new AtomicReference<>();
        AtomicReference<List<AuditRecord>> capturedRecords = new AtomicReference<>();

        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.setOnFlush((root, records) -> {
            capturedRoot.set(root);
            capturedRecords.set(records);
        });

        a.record("m", "in", "out", 0.5, 10L);
        a.flush();

        assertNotNull(capturedRoot.get(), "onFlush must receive a non-null Merkle root");
        assertEquals(64, capturedRoot.get().length());
        assertNotNull(capturedRecords.get());
        assertEquals(1, capturedRecords.get().size());
    }

    @Test
    void onFlush_receivesAllRecords() {
        AtomicInteger capturedCount = new AtomicInteger(0);
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        a.setOnFlush((root, records) -> capturedCount.set(records.size()));

        a.record("m", "in1", "out1", null, 0L);
        a.record("m", "in2", "out2", null, 0L);
        a.record("m", "in3", "out3", null, 0L);
        a.flush();

        assertEquals(3, capturedCount.get());
    }

    // =========================================================================
    // recordWithMeta()
    // =========================================================================

    @Test
    void recordWithMeta_attachesMetadata() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        Map<String, Object> meta = new HashMap<>();
        meta.put("session", "abc");
        meta.put("user", "alice");

        a.recordWithMeta("m", "in", "out", 0.9, 10L, meta);
        InferenceAuditor.FlushResult result = a.flush();

        AuditRecord rec = result.getRecords().get(0);
        assertEquals("abc", rec.getMetadata().get("session"));
        assertEquals("alice", rec.getMetadata().get("user"));
    }

    @Test
    void recordWithMeta_nullMetaAccepted() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertDoesNotThrow(() -> a.recordWithMeta("m", "in", "out", null, 0L, null));
        assertEquals(1, a.stats().getBuffered());
    }

    // =========================================================================
    // epochId()
    // =========================================================================

    @Test
    void epochId_startsWithEp() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        assertTrue(a.epochId().startsWith("ep_"),
                "epochId must start with 'ep_': " + a.epochId());
    }

    @Test
    void epochId_recordMatchesAuditorEpoch() {
        InferenceAuditor a = new InferenceAuditor("sys", 0);
        String epochBefore = a.epochId();
        a.record("m", "in", "out", null, 0L);
        InferenceAuditor.FlushResult result = a.flush();
        // The record was created in epochBefore
        assertEquals(epochBefore, result.getRecords().get(0).getEpochId());
    }
}
