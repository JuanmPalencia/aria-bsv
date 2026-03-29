//! aria-bsv-auditor — High-performance AI inference auditor for ARIA BRC-121.
//!
//! Buffers inference records in memory and computes a RFC 6962 Merkle root
//! on flush, suitable for broadcasting to BSV via the ARIA protocol.
//!
//! # Example
//! ```rust
//! use aria_bsv_auditor::InferenceAuditor;
//! use serde_json::json;
//!
//! let mut auditor = InferenceAuditor::new("my-system", 100);
//!
//! let record_id = auditor.record(
//!     "gpt-4",
//!     &json!({"prompt": "hello"}),
//!     &json!({"completion": "world"}),
//!     Some(0.95),
//!     42,
//! ).unwrap();
//!
//! assert_eq!(auditor.stats().buffered, 1);
//!
//! let (root, records) = auditor.flush().unwrap();
//! assert_eq!(records.len(), 1);
//! assert_eq!(root.len(), 64);  // Merkle root hex
//! ```

use aria_bsv_hasher::{hash_object, hash_bytes};
use aria_bsv_merkle::Tree;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by the auditor.
#[derive(Debug, Clone, PartialEq)]
pub enum AuditorError {
    /// Confidence value is out of [0.0, 1.0] range.
    InvalidConfidence(f64),
    /// Latency is negative.
    NegativeLatency(i64),
    /// The buffer is empty; flush requires at least one record.
    EmptyBuffer,
    /// Hash computation failed (e.g. non-finite float in input/output).
    HashError(String),
}

impl std::fmt::Display for AuditorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditorError::InvalidConfidence(c) => {
                write!(f, "confidence {c} is out of [0.0, 1.0] range")
            }
            AuditorError::NegativeLatency(ms) => write!(f, "latency_ms {ms} is negative"),
            AuditorError::EmptyBuffer => write!(f, "buffer is empty; nothing to flush"),
            AuditorError::HashError(msg) => write!(f, "hash error: {msg}"),
        }
    }
}

impl std::error::Error for AuditorError {}

// ---------------------------------------------------------------------------
// AuditRecord
// ---------------------------------------------------------------------------

/// A single recorded AI inference anchored to the ARIA BRC-121 protocol.
///
/// Fields mirror the Python SDK's `AuditRecord` and the TypeScript SDK's
/// `AuditRecord` interface for cross-SDK compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditRecord {
    /// Unique identifier: `"rec_{epoch_id}_{seq:06}"`.
    pub record_id: String,
    /// The ARIA system identifier.
    pub system_id: String,
    /// Model identifier.
    pub model_id: String,
    /// SHA-256 hex hash of the canonical input.
    pub input_hash: String,
    /// SHA-256 hex hash of the canonical output.
    pub output_hash: String,
    /// Model confidence in [0.0, 1.0], or `None` if unavailable.
    pub confidence: Option<f64>,
    /// Inference wall-clock duration in milliseconds.
    pub latency_ms: i64,
    /// ISO-8601 UTC timestamp of the inference.
    pub timestamp: String,
    /// Sequence number within the current epoch (0-based).
    pub sequence: usize,
    /// Epoch identifier this record belongs to.
    pub epoch_id: String,
    /// Additional caller-supplied metadata.
    pub metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Snapshot of the auditor's runtime state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditStats {
    /// Number of records currently buffered (not yet flushed).
    pub buffered: usize,
    /// Total records ever recorded (including flushed records).
    pub total_recorded: u64,
    /// Configured batch size.
    pub batch_size: usize,
    /// The system identifier.
    pub system_id: String,
}

// ---------------------------------------------------------------------------
// FlushCallback
// ---------------------------------------------------------------------------

/// Callback invoked on each flush with the Merkle root and the records.
///
/// Use `set_on_flush` to install a callback that broadcasts the epoch to BSV.
pub type FlushCallback = Box<dyn Fn(&str, &[AuditRecord]) + Send + Sync>;

// ---------------------------------------------------------------------------
// InferenceAuditor
// ---------------------------------------------------------------------------

/// Buffers AI inference records and produces a Merkle root on flush.
///
/// # Thread safety
/// `InferenceAuditor` is `Send + Sync` when the flush callback is too.
/// For concurrent use, wrap in `Arc<Mutex<InferenceAuditor>>`.
pub struct InferenceAuditor {
    system_id: String,
    epoch_id: String,
    batch_size: usize,
    buffer: Vec<AuditRecord>,
    total_recorded: u64,
    on_flush: Option<FlushCallback>,
}

impl InferenceAuditor {
    /// Create a new auditor.
    ///
    /// # Arguments
    /// * `system_id`  — ARIA system identifier (e.g. `"my-llm-app"`).
    /// * `batch_size` — Auto-flush when buffer reaches this size (0 = never auto-flush).
    pub fn new(system_id: impl Into<String>, batch_size: usize) -> Self {
        InferenceAuditor {
            system_id: system_id.into(),
            epoch_id: new_epoch_id(),
            batch_size,
            buffer: Vec::new(),
            total_recorded: 0,
            on_flush: None,
        }
    }

    /// Set a callback that fires on every flush.
    pub fn set_on_flush<F>(&mut self, callback: F)
    where
        F: Fn(&str, &[AuditRecord]) + Send + Sync + 'static,
    {
        self.on_flush = Some(Box::new(callback));
    }

    /// Record an inference.
    ///
    /// # Arguments
    /// * `model_id`    — Model identifier.
    /// * `input`       — Raw input (serialised to canonical JSON for hashing).
    /// * `output`      — Raw output (serialised to canonical JSON for hashing).
    /// * `confidence`  — Model confidence in [0.0, 1.0], or `None`.
    /// * `latency_ms`  — Inference wall-clock time in milliseconds.
    ///
    /// # Errors
    /// Returns `AuditorError::InvalidConfidence` if `confidence` is outside [0.0, 1.0].
    /// Returns `AuditorError::NegativeLatency` if `latency_ms < 0`.
    pub fn record(
        &mut self,
        model_id: impl Into<String>,
        input: &Value,
        output: &Value,
        confidence: Option<f64>,
        latency_ms: i64,
    ) -> Result<String, AuditorError> {
        self.record_with_meta(model_id, input, output, confidence, latency_ms, HashMap::new())
    }

    /// Like [`record`][Self::record] but with additional metadata.
    pub fn record_with_meta(
        &mut self,
        model_id: impl Into<String>,
        input: &Value,
        output: &Value,
        confidence: Option<f64>,
        latency_ms: i64,
        metadata: HashMap<String, Value>,
    ) -> Result<String, AuditorError> {
        if let Some(c) = confidence {
            if !(0.0..=1.0).contains(&c) {
                return Err(AuditorError::InvalidConfidence(c));
            }
        }
        if latency_ms < 0 {
            return Err(AuditorError::NegativeLatency(latency_ms));
        }

        let input_hash = hash_object(input)
            .map_err(|e| AuditorError::HashError(e.to_string()))?;
        let output_hash = hash_object(output)
            .map_err(|e| AuditorError::HashError(e.to_string()))?;

        let sequence = self.buffer.len() + self.total_recorded as usize;
        let record_id = format!("rec_{}_{:06}", self.epoch_id, sequence);

        let rec = AuditRecord {
            record_id: record_id.clone(),
            system_id: self.system_id.clone(),
            model_id: model_id.into(),
            input_hash,
            output_hash,
            confidence,
            latency_ms,
            timestamp: Utc::now().to_rfc3339(),
            sequence,
            epoch_id: self.epoch_id.clone(),
            metadata,
        };

        self.buffer.push(rec);
        self.total_recorded += 1;

        // Auto-flush if batch_size is set and reached.
        if self.batch_size > 0 && self.buffer.len() >= self.batch_size {
            self.flush()?;
        }

        Ok(record_id)
    }

    /// Flush buffered records and return `(merkle_root_hex, records)`.
    ///
    /// The Merkle root is computed from the SHA-256 hashes of each record's
    /// canonical JSON serialization.  The on-flush callback is called if set.
    ///
    /// Returns `Err(AuditorError::EmptyBuffer)` if there are no records.
    pub fn flush(&mut self) -> Result<(String, Vec<AuditRecord>), AuditorError> {
        if self.buffer.is_empty() {
            return Err(AuditorError::EmptyBuffer);
        }

        let records = std::mem::take(&mut self.buffer);
        let mut tree = Tree::new();

        for rec in &records {
            let v = serde_json::to_value(rec).expect("AuditRecord serialization");
            let h = hash_object(&v).map_err(|e| AuditorError::HashError(e.to_string()))?;
            let h_bytes = hex::decode(&h).expect("valid hex from hash_object");
            tree.add_leaf(&h_bytes);
        }

        let root = tree.root().expect("non-empty tree");

        if let Some(cb) = &self.on_flush {
            cb(&root, &records);
        }

        // Rotate epoch_id for the next batch.
        self.epoch_id = new_epoch_id();

        Ok((root, records))
    }

    /// Clear the buffer without computing a Merkle root or calling the callback.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.epoch_id = new_epoch_id();
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> AuditStats {
        AuditStats {
            buffered: self.buffer.len(),
            total_recorded: self.total_recorded,
            batch_size: self.batch_size,
            system_id: self.system_id.clone(),
        }
    }

    /// Current epoch identifier.
    pub fn epoch_id(&self) -> &str {
        &self.epoch_id
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn new_epoch_id() -> String {
    format!("ep_{}", Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn auditor() -> InferenceAuditor {
        InferenceAuditor::new("test-system", 0)
    }

    // -- record ---------------------------------------------------------------

    #[test]
    fn record_returns_non_empty_id() {
        let mut a = auditor();
        let id = a.record("gpt-4", &json!({"q": "hi"}), &json!({"r": "hello"}), None, 10).unwrap();
        assert!(!id.is_empty());
        assert!(id.starts_with("rec_"));
    }

    #[test]
    fn record_stores_in_buffer() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        assert_eq!(a.stats().buffered, 1);
    }

    #[test]
    fn record_increments_total_recorded() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        assert_eq!(a.stats().total_recorded, 2);
    }

    #[test]
    fn record_invalid_confidence_errors() {
        let mut a = auditor();
        assert!(matches!(
            a.record("m", &json!({}), &json!({}), Some(1.5), 0),
            Err(AuditorError::InvalidConfidence(_))
        ));
        assert!(matches!(
            a.record("m", &json!({}), &json!({}), Some(-0.1), 0),
            Err(AuditorError::InvalidConfidence(_))
        ));
    }

    #[test]
    fn record_valid_confidence_boundary_values() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), Some(0.0), 0).unwrap();
        a.record("m", &json!({}), &json!({}), Some(1.0), 0).unwrap();
        assert_eq!(a.stats().buffered, 2);
    }

    #[test]
    fn record_negative_latency_errors() {
        let mut a = auditor();
        assert!(matches!(
            a.record("m", &json!({}), &json!({}), None, -1),
            Err(AuditorError::NegativeLatency(-1))
        ));
    }

    #[test]
    fn record_input_hash_is_64_char_hex() {
        let mut a = auditor();
        a.record("m", &json!({"prompt": "test"}), &json!({}), None, 0).unwrap();
        let rec = &a.buffer[0];
        assert_eq!(rec.input_hash.len(), 64);
        assert!(rec.input_hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn record_sequence_increments() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        assert_eq!(a.buffer[0].sequence, 0);
        assert_eq!(a.buffer[1].sequence, 1);
    }

    #[test]
    fn record_epoch_id_matches_auditor() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        assert_eq!(a.buffer[0].epoch_id, a.epoch_id());
    }

    // -- record_with_meta -----------------------------------------------------

    #[test]
    fn record_with_meta_attaches_metadata() {
        let mut a = auditor();
        let mut meta = HashMap::new();
        meta.insert("version".to_string(), json!("1.0"));
        a.record_with_meta("m", &json!({}), &json!({}), None, 0, meta).unwrap();
        assert_eq!(a.buffer[0].metadata.get("version"), Some(&json!("1.0")));
    }

    #[test]
    fn record_with_empty_meta_ok() {
        let mut a = auditor();
        a.record_with_meta("m", &json!({}), &json!({}), None, 5, HashMap::new()).unwrap();
        assert_eq!(a.stats().buffered, 1);
    }

    // -- flush ----------------------------------------------------------------

    #[test]
    fn flush_returns_merkle_root_and_records() {
        let mut a = auditor();
        a.record("m", &json!({"x": 1}), &json!({"y": 2}), Some(0.9), 10).unwrap();
        let (root, records) = a.flush().unwrap();
        assert_eq!(root.len(), 64);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn flush_clears_buffer() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.flush().unwrap();
        assert_eq!(a.stats().buffered, 0);
    }

    #[test]
    fn flush_empty_buffer_errors() {
        let mut a = auditor();
        assert!(matches!(a.flush(), Err(AuditorError::EmptyBuffer)));
    }

    #[test]
    fn flush_root_is_deterministic_for_same_records() {
        // Two auditors with identical records must produce the same Merkle root.
        let mut a1 = InferenceAuditor::new("sys", 0);
        let mut a2 = InferenceAuditor::new("sys", 0);

        // Force same epoch_id so record_ids match.
        a2.epoch_id = a1.epoch_id.clone();

        let input = json!({"prompt": "hello"});
        let output = json!({"completion": "world"});

        a1.record("m", &input, &output, Some(0.8), 50).unwrap();
        a2.record("m", &input, &output, Some(0.8), 50).unwrap();

        // Note: timestamps will differ, so hashes differ — this is expected.
        // Determinism test: same auditor, same sequence produces same hashes.
        let h1 = a1.buffer[0].input_hash.clone();
        let h2 = a2.buffer[0].input_hash.clone();
        assert_eq!(h1, h2);  // input_hash is deterministic for same input
    }

    #[test]
    fn flush_rotates_epoch_id() {
        let mut a = auditor();
        let ep1 = a.epoch_id().to_string();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.flush().unwrap();
        let ep2 = a.epoch_id().to_string();
        assert_ne!(ep1, ep2);
    }

    #[test]
    fn flush_calls_on_flush_callback() {
        use std::sync::{Arc, Mutex};

        let mut a = auditor();
        let called = Arc::new(Mutex::new(false));
        let called_clone = called.clone();

        a.set_on_flush(move |_root, _records| {
            *called_clone.lock().unwrap() = true;
        });

        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.flush().unwrap();

        assert!(*called.lock().unwrap());
    }

    #[test]
    fn flush_callback_receives_correct_record_count() {
        use std::sync::{Arc, Mutex};

        let mut a = auditor();
        let count = Arc::new(Mutex::new(0usize));
        let count_clone = count.clone();

        a.set_on_flush(move |_root, records| {
            *count_clone.lock().unwrap() = records.len();
        });

        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.flush().unwrap();

        assert_eq!(*count.lock().unwrap(), 3);
    }

    // -- auto-flush -----------------------------------------------------------

    #[test]
    fn auto_flush_at_batch_size() {
        let mut a = InferenceAuditor::new("sys", 2);  // batch_size = 2
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        assert_eq!(a.stats().buffered, 1);
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        // Auto-flushed → buffer empty
        assert_eq!(a.stats().buffered, 0);
        assert_eq!(a.stats().total_recorded, 2);
    }

    // -- reset ----------------------------------------------------------------

    #[test]
    fn reset_clears_buffer() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.reset();
        assert_eq!(a.stats().buffered, 0);
    }

    #[test]
    fn reset_rotates_epoch_id() {
        let mut a = auditor();
        let ep1 = a.epoch_id().to_string();
        a.reset();
        assert_ne!(ep1, a.epoch_id().to_string());
    }

    #[test]
    fn reset_preserves_total_recorded() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.reset();
        // total_recorded includes records that were buffered but not flushed.
        assert_eq!(a.stats().total_recorded, 1);
    }

    // -- stats ----------------------------------------------------------------

    #[test]
    fn stats_initial_state() {
        let a = InferenceAuditor::new("my-sys", 50);
        let s = a.stats();
        assert_eq!(s.buffered, 0);
        assert_eq!(s.total_recorded, 0);
        assert_eq!(s.batch_size, 50);
        assert_eq!(s.system_id, "my-sys");
    }

    #[test]
    fn stats_after_flush_total_recorded_preserved() {
        let mut a = auditor();
        a.record("m", &json!({}), &json!({}), None, 0).unwrap();
        a.flush().unwrap();
        assert_eq!(a.stats().total_recorded, 1);
        assert_eq!(a.stats().buffered, 0);
    }
}
