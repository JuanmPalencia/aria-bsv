// Package auditor provides the ARIA InferenceAuditor for Go applications.
//
// InferenceAuditor records AI inference events, batches them, and can flush
// them to an external sink (e.g., ARIA portal API). It is goroutine-safe.
package auditor

import (
	"fmt"
	"sync"
	"time"

	"github.com/JuanmPalencia/aria-bsv/sdk-go/hasher"
)

// Record represents a single audited inference event.
type Record struct {
	RecordID    string                 `json:"record_id"`
	SystemID    string                 `json:"system_id"`
	ModelID     string                 `json:"model_id"`
	Input       interface{}            `json:"input"`
	Output      interface{}            `json:"output"`
	Confidence  float64                `json:"confidence"`
	LatencyMS   int64                  `json:"latency_ms"`
	InputHash   string                 `json:"input_hash"`
	OutputHash  string                 `json:"output_hash"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// FlushFunc is called with a batch of records when Flush is triggered.
// Implementations should persist or broadcast the records. Errors are logged
// but do not block the auditor.
type FlushFunc func(records []*Record) error

// Config holds configuration for an InferenceAuditor.
type Config struct {
	// SystemID identifies the AI system being audited.
	SystemID string
	// BatchSize is the maximum number of records before auto-flush.
	// Defaults to 100.
	BatchSize int
	// OnFlush is called with a batch of records. May be nil (records are
	// discarded after accumulation, useful for testing).
	OnFlush FlushFunc
}

// InferenceAuditor records AI inferences in a thread-safe batch buffer.
type InferenceAuditor struct {
	cfg     Config
	mu      sync.Mutex
	batch   []*Record
	counter uint64
}

// New creates a new InferenceAuditor with the given configuration.
func New(cfg Config) *InferenceAuditor {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 100
	}
	return &InferenceAuditor{
		cfg:   cfg,
		batch: make([]*Record, 0, cfg.BatchSize),
	}
}

// Record adds a single inference event to the batch buffer.
// model is the model identifier; input and output are arbitrary values
// (they will be canonically hashed). confidence must be in [0, 1].
// Returns the record_id of the newly created record.
func (a *InferenceAuditor) Record(model string, input, output interface{}, confidence float64, latencyMS int64) (string, error) {
	if confidence < 0 || confidence > 1 {
		return "", fmt.Errorf("confidence must be in [0, 1], got %f", confidence)
	}

	inputHash, err := hasher.HashObject(input)
	if err != nil {
		return "", fmt.Errorf("hashing input: %w", err)
	}
	outputHash, err := hasher.HashObject(output)
	if err != nil {
		return "", fmt.Errorf("hashing output: %w", err)
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	a.counter++
	recordID := fmt.Sprintf("rec_%d_%d", time.Now().UnixMilli(), a.counter)

	rec := &Record{
		RecordID:   recordID,
		SystemID:   a.cfg.SystemID,
		ModelID:    model,
		Input:      input,
		Output:     output,
		Confidence: confidence,
		LatencyMS:  latencyMS,
		InputHash:  inputHash,
		OutputHash: outputHash,
		Timestamp:  time.Now().UTC(),
	}
	a.batch = append(a.batch, rec)

	if len(a.batch) >= a.cfg.BatchSize {
		if err := a.flushLocked(); err != nil {
			return recordID, fmt.Errorf("auto-flush: %w", err)
		}
	}

	return recordID, nil
}

// Flush forces an immediate flush of all buffered records.
// It is safe to call from any goroutine.
func (a *InferenceAuditor) Flush() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.flushLocked()
}

// Len returns the number of records currently buffered (not yet flushed).
func (a *InferenceAuditor) Len() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.batch)
}

// flushLocked must be called with a.mu held.
func (a *InferenceAuditor) flushLocked() error {
	if len(a.batch) == 0 {
		return nil
	}
	toFlush := a.batch
	a.batch = make([]*Record, 0, a.cfg.BatchSize)

	if a.cfg.OnFlush != nil {
		return a.cfg.OnFlush(toFlush)
	}
	return nil
}
