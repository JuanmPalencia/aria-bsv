package auditor_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/JuanmPalencia/aria-bsv/sdk-go/auditor"
)

func TestNew_DefaultBatchSize(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "test"})
	if a == nil {
		t.Fatal("expected non-nil auditor")
	}
}

func TestRecord_Basic(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys-1", BatchSize: 10})
	id, err := a.Record("model-a", "hello", "world", 0.9, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty record ID")
	}
	if a.Len() != 1 {
		t.Errorf("expected 1 buffered record, got %d", a.Len())
	}
}

func TestRecord_InvalidConfidenceTooHigh(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys"})
	_, err := a.Record("m", "in", "out", 1.5, 0)
	if err == nil {
		t.Error("expected error for confidence > 1")
	}
}

func TestRecord_InvalidConfidenceNegative(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys"})
	_, err := a.Record("m", "in", "out", -0.1, 0)
	if err == nil {
		t.Error("expected error for negative confidence")
	}
}

func TestRecord_BoundaryConfidence(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys"})
	for _, conf := range []float64{0.0, 0.5, 1.0} {
		_, err := a.Record("m", "in", "out", conf, 0)
		if err != nil {
			t.Errorf("confidence %f should be valid: %v", conf, err)
		}
	}
}

func TestRecord_HashesArePopulated(t *testing.T) {
	var flushed []*auditor.Record
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			flushed = append(flushed, recs...)
			return nil
		},
	})
	a.Record("m", map[string]interface{}{"q": "test"}, "answer", 0.8, 10)
	a.Flush()
	if len(flushed) != 1 {
		t.Fatalf("expected 1 flushed record, got %d", len(flushed))
	}
	r := flushed[0]
	if len(r.InputHash) != 64 {
		t.Errorf("expected 64-char input hash, got %d", len(r.InputHash))
	}
	if len(r.OutputHash) != 64 {
		t.Errorf("expected 64-char output hash, got %d", len(r.OutputHash))
	}
}

func TestRecord_DeterministicHashes(t *testing.T) {
	var hashes []string
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			for _, r := range recs {
				hashes = append(hashes, r.InputHash)
			}
			return nil
		},
	})
	input := map[string]interface{}{"z": 2, "a": 1}
	a.Record("m", input, "out", 0.5, 0)
	a.Record("m", input, "out", 0.5, 0)
	a.Flush()
	if len(hashes) != 2 {
		t.Fatalf("expected 2, got %d", len(hashes))
	}
	if hashes[0] != hashes[1] {
		t.Error("same input must produce same hash")
	}
}

func TestRecord_DifferentInputsDifferentHashes(t *testing.T) {
	var hashes []string
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			for _, r := range recs {
				hashes = append(hashes, r.InputHash)
			}
			return nil
		},
	})
	a.Record("m", "input-1", "out", 0.5, 0)
	a.Record("m", "input-2", "out", 0.5, 0)
	a.Flush()
	if hashes[0] == hashes[1] {
		t.Error("different inputs must produce different hashes")
	}
}

func TestFlush_EmptyBatch(t *testing.T) {
	var called bool
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		OnFlush: func(recs []*auditor.Record) error {
			called = true
			return nil
		},
	})
	err := a.Flush()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Error("OnFlush should not be called for empty batch")
	}
}

func TestAutoFlush_OnBatchSizeReached(t *testing.T) {
	var mu sync.Mutex
	var total int
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 5,
		OnFlush: func(recs []*auditor.Record) error {
			mu.Lock()
			total += len(recs)
			mu.Unlock()
			return nil
		},
	})
	for i := 0; i < 5; i++ {
		a.Record("m", i, "out", 0.5, 0)
	}
	mu.Lock()
	got := total
	mu.Unlock()
	if got != 5 {
		t.Errorf("expected auto-flush of 5, got %d", got)
	}
	if a.Len() != 0 {
		t.Errorf("expected 0 buffered after auto-flush, got %d", a.Len())
	}
}

func TestFlush_PropagatesError(t *testing.T) {
	want := errors.New("sink down")
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 100,
		OnFlush: func(_ []*auditor.Record) error {
			return want
		},
	})
	a.Record("m", "in", "out", 0.5, 0)
	err := a.Flush()
	if !errors.Is(err, want) {
		t.Errorf("expected %v, got %v", want, err)
	}
}

func TestLen_UpdatesCorrectly(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys", BatchSize: 100})
	if a.Len() != 0 {
		t.Error("expected 0 initially")
	}
	a.Record("m", "in", "out", 0.7, 0)
	a.Record("m", "in2", "out2", 0.8, 0)
	if a.Len() != 2 {
		t.Errorf("expected 2, got %d", a.Len())
	}
	a.Flush()
	if a.Len() != 0 {
		t.Errorf("expected 0 after flush, got %d", a.Len())
	}
}

func TestConcurrentRecord(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys", BatchSize: 1000})
	var wg sync.WaitGroup
	n := 200
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			a.Record("m", i, "out", 0.5, 0)
		}(i)
	}
	wg.Wait()
	a.Flush()
	// After flush the buffer should be empty with no panics or races
	if a.Len() != 0 {
		t.Errorf("expected 0 after flush, got %d", a.Len())
	}
}

func TestRecordID_Unique(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "sys", BatchSize: 1000})
	ids := make(map[string]bool)
	for i := 0; i < 50; i++ {
		id, _ := a.Record("m", i, "out", 0.5, 0)
		if ids[id] {
			t.Errorf("duplicate record ID: %s", id)
		}
		ids[id] = true
	}
}

func TestRecord_SystemIDPropagated(t *testing.T) {
	var got string
	a := auditor.New(auditor.Config{
		SystemID:  "aria-production",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			got = recs[0].SystemID
			return nil
		},
	})
	a.Record("m", "in", "out", 0.5, 0)
	a.Flush()
	if got != "aria-production" {
		t.Errorf("expected aria-production, got %s", got)
	}
}

func TestRecord_LatencyPreserved(t *testing.T) {
	var got int64
	a := auditor.New(auditor.Config{
		SystemID:  "sys",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			got = recs[0].LatencyMS
			return nil
		},
	})
	a.Record("m", "in", "out", 0.5, 314)
	a.Flush()
	if got != 314 {
		t.Errorf("expected latency 314, got %d", got)
	}
}

func TestFlush_OnFlushNil(t *testing.T) {
	// When OnFlush is nil, flush should succeed silently
	a := auditor.New(auditor.Config{SystemID: "sys", BatchSize: 100})
	a.Record("m", "in", "out", 0.5, 0)
	err := a.Flush()
	if err != nil {
		t.Errorf("unexpected error with nil OnFlush: %v", err)
	}
	if a.Len() != 0 {
		t.Error("expected buffer cleared after flush")
	}
}

func TestStats_InitialState(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "my-system", BatchSize: 50})
	s := a.Stats()
	if s.SystemID != "my-system" {
		t.Errorf("wrong system_id: %s", s.SystemID)
	}
	if s.BatchSize != 50 {
		t.Errorf("wrong batch_size: %d", s.BatchSize)
	}
	if s.Buffered != 0 {
		t.Errorf("expected 0 buffered, got %d", s.Buffered)
	}
	if s.TotalRecorded != 0 {
		t.Errorf("expected 0 total, got %d", s.TotalRecorded)
	}
}

func TestStats_AfterRecords(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "s", BatchSize: 100})
	a.Record("m", "i", "o", 0.5, 0)
	a.Record("m", "i", "o", 0.5, 0)
	s := a.Stats()
	if s.Buffered != 2 {
		t.Errorf("expected 2 buffered, got %d", s.Buffered)
	}
	if s.TotalRecorded != 2 {
		t.Errorf("expected 2 total, got %d", s.TotalRecorded)
	}
}

func TestStats_TotalRecorded_IncludesFlushed(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "s", BatchSize: 100})
	a.Record("m", "i", "o", 0.5, 0)
	a.Flush()
	a.Record("m", "i", "o", 0.5, 0)
	s := a.Stats()
	if s.TotalRecorded != 2 {
		t.Errorf("expected total 2 (1 flushed + 1 pending), got %d", s.TotalRecorded)
	}
	if s.Buffered != 1 {
		t.Errorf("expected 1 buffered, got %d", s.Buffered)
	}
}

func TestReset_ClearsBuffer(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "s", BatchSize: 100})
	a.Record("m", "i", "o", 0.5, 0)
	a.Record("m", "i", "o", 0.5, 0)
	a.Reset()
	if a.Len() != 0 {
		t.Errorf("expected 0 after Reset, got %d", a.Len())
	}
}

func TestReset_DoesNotCallOnFlush(t *testing.T) {
	var called bool
	a := auditor.New(auditor.Config{
		SystemID:  "s",
		BatchSize: 100,
		OnFlush: func(_ []*auditor.Record) error {
			called = true
			return nil
		},
	})
	a.Record("m", "i", "o", 0.5, 0)
	a.Reset()
	if called {
		t.Error("Reset should not call OnFlush")
	}
}

func TestStats_TotalRecorded_preserved_after_Reset(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "s", BatchSize: 100})
	a.Record("m", "i", "o", 0.5, 0)
	a.Record("m", "i", "o", 0.5, 0)
	a.Reset()
	s := a.Stats()
	if s.TotalRecorded != 2 {
		t.Errorf("TotalRecorded should persist after Reset, got %d", s.TotalRecorded)
	}
}

func TestSetOnFlush_UpdatesCallback(t *testing.T) {
	var first, second bool
	a := auditor.New(auditor.Config{
		SystemID:  "s",
		BatchSize: 100,
		OnFlush: func(_ []*auditor.Record) error {
			first = true
			return nil
		},
	})
	a.Record("m", "i", "o", 0.5, 0)
	a.Flush()
	if !first {
		t.Error("first flush callback not called")
	}

	a.SetOnFlush(func(_ []*auditor.Record) error {
		second = true
		return nil
	})
	a.Record("m", "i", "o", 0.5, 0)
	a.Flush()
	if !second {
		t.Error("second flush callback not called after SetOnFlush")
	}
}

func TestRecordWithMeta_MetadataAttached(t *testing.T) {
	var rec *auditor.Record
	a := auditor.New(auditor.Config{
		SystemID:  "s",
		BatchSize: 100,
		OnFlush: func(recs []*auditor.Record) error {
			rec = recs[0]
			return nil
		},
	})
	meta := map[string]interface{}{"session": "abc", "user": "alice"}
	a.RecordWithMeta("m", "in", "out", 0.9, 10, meta)
	a.Flush()
	if rec == nil {
		t.Fatal("no record flushed")
	}
	if rec.Metadata["session"] != "abc" {
		t.Errorf("metadata not attached: %v", rec.Metadata)
	}
}

func TestRecordWithMeta_EmptyMetaOk(t *testing.T) {
	a := auditor.New(auditor.Config{SystemID: "s", BatchSize: 100})
	id, err := a.RecordWithMeta("m", "i", "o", 0.5, 0, nil)
	if err != nil || id == "" {
		t.Errorf("unexpected error or empty id: %v %s", err, id)
	}
}
