package meter

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/yash/gatewayllm/internal/config"
)

// fakeSink captures batches and can block to simulate a slow database.
type fakeSink struct {
	mu      sync.Mutex
	rows    []Record
	batches int
	block   chan struct{}
	err     error
}

func (f *fakeSink) WriteBatch(_ context.Context, rows []Record) error {
	if f.block != nil {
		<-f.block
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	f.rows = append(f.rows, rows...)
	f.batches++
	return nil
}

func (f *fakeSink) Close() error { return nil }

func (f *fakeSink) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.rows)
}

// TestMeter_FlushesOnClose asserts buffered rows are not lost at shutdown.
func TestMeter_FlushesOnClose(t *testing.T) {
	sink := &fakeSink{}
	m := New(Options{Sink: sink, BatchSize: 100, FlushInterval: time.Hour})

	for i := 0; i < 5; i++ {
		m.Record(Record{RequestID: "r", TenantID: "t"})
	}
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := sink.count(); got != 5 {
		t.Errorf("wrote %d rows, want 5: Close must drain the buffer", got)
	}
}

// TestMeter_FlushesOnBatchSize asserts a full batch is written without waiting
// for the interval.
func TestMeter_FlushesOnBatchSize(t *testing.T) {
	sink := &fakeSink{}
	m := New(Options{Sink: sink, BatchSize: 3, FlushInterval: time.Hour})
	defer m.Close()

	for i := 0; i < 3; i++ {
		m.Record(Record{RequestID: "r", TenantID: "t"})
	}

	waitFor(t, func() bool { return sink.count() == 3 })
}

// TestMeter_FlushesOnInterval asserts a partial batch still lands promptly, so
// a low-traffic gateway's dashboards do not lag until a batch fills.
func TestMeter_FlushesOnInterval(t *testing.T) {
	sink := &fakeSink{}
	m := New(Options{Sink: sink, BatchSize: 1000, FlushInterval: 20 * time.Millisecond})
	defer m.Close()

	m.Record(Record{RequestID: "r", TenantID: "t"})
	waitFor(t, func() bool { return sink.count() == 1 })
}

// TestMeter_DropsRatherThanBlocks is the meter's defining trade-off: when the
// sink stalls, live requests must not stall with it.
func TestMeter_DropsRatherThanBlocks(t *testing.T) {
	sink := &fakeSink{block: make(chan struct{})}
	m := New(Options{Sink: sink, BufferSize: 4, BatchSize: 1, FlushInterval: time.Hour})

	// The writer goroutine is stuck in WriteBatch; the buffer fills and then
	// overflows. Record must stay fast regardless.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 200; i++ {
			m.Record(Record{RequestID: "r", TenantID: "t"})
		}
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Record blocked: usage metering must never apply backpressure to live requests")
	}

	if m.Stats().Dropped.Load() == 0 {
		t.Error("expected drops to be counted so the gap is visible rather than silent")
	}

	close(sink.block)
	_ = m.Close()
}

// TestMeter_RecordAfterClose asserts a late record does not panic on a closed
// channel, which would take the process down during shutdown.
func TestMeter_RecordAfterClose(t *testing.T) {
	m := New(Options{Sink: &fakeSink{}, BatchSize: 10, FlushInterval: time.Hour})
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	m.Record(Record{RequestID: "late"}) // must not panic
}

// TestMeter_SetsCreatedAt asserts rows get a timestamp when the caller omits it.
func TestMeter_SetsCreatedAt(t *testing.T) {
	sink := &fakeSink{}
	m := New(Options{Sink: sink, BatchSize: 1, FlushInterval: time.Hour})

	m.Record(Record{RequestID: "r"})
	waitFor(t, func() bool { return sink.count() == 1 })
	_ = m.Close()

	sink.mu.Lock()
	defer sink.mu.Unlock()
	if sink.rows[0].CreatedAt.IsZero() {
		t.Error("CreatedAt must be stamped when unset")
	}
}

// --- pricing ---

// TestPricing_Cost asserts per-million-token arithmetic.
func TestPricing_Cost(t *testing.T) {
	p := NewPricing(map[string]config.Price{
		"gpt-4o": {InputPerMillion: 2.50, OutputPerMillion: 10.00},
	})

	// 1M input at $2.50 + 1M output at $10.00.
	cost, ok := p.Cost("openai", "gpt-4o", 1_000_000, 1_000_000)
	if !ok {
		t.Fatal("price must be found")
	}
	if cost != 12.50 {
		t.Errorf("cost = %v, want 12.50", cost)
	}

	// 1000 in / 500 out.
	cost, _ = p.Cost("openai", "gpt-4o", 1000, 500)
	want := 1000.0/1e6*2.50 + 500.0/1e6*10.00
	if cost != want {
		t.Errorf("cost = %v, want %v", cost, want)
	}
}

// TestPricing_ProviderQualifiedWins asserts the same model served by two
// providers can be priced differently — the reason routing saves money.
func TestPricing_ProviderQualifiedWins(t *testing.T) {
	p := NewPricing(map[string]config.Price{
		"llama-3.3-70b":      {InputPerMillion: 10, OutputPerMillion: 10},
		"groq/llama-3.3-70b": {InputPerMillion: 0.59, OutputPerMillion: 0.79},
	})

	groqCost, _ := p.Cost("groq", "llama-3.3-70b", 1_000_000, 0)
	if groqCost != 0.59 {
		t.Errorf("groq cost = %v, want the provider-qualified price 0.59", groqCost)
	}

	otherCost, _ := p.Cost("other", "llama-3.3-70b", 1_000_000, 0)
	if otherCost != 10 {
		t.Errorf("fallback cost = %v, want the bare-model price 10", otherCost)
	}
}

// TestPricing_UnknownModel asserts a missing price is reported, not silently
// treated as free — which would understate spend on the cost dashboard.
func TestPricing_UnknownModel(t *testing.T) {
	p := NewPricing(nil)
	cost, ok := p.Cost("openai", "unpriced-model", 1000, 1000)
	if ok {
		t.Error("ok must be false for an unpriced model")
	}
	if cost != 0 {
		t.Errorf("cost = %v, want 0", cost)
	}
}

// TestPricing_CaseInsensitive asserts price keys tolerate casing differences.
func TestPricing_CaseInsensitive(t *testing.T) {
	p := NewPricing(map[string]config.Price{"GPT-4o": {InputPerMillion: 1}})
	if !p.Has("openai", "gpt-4o") {
		t.Error("price lookup must be case-insensitive")
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("timed out waiting for the meter to flush")
}
