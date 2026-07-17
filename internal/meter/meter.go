// Package meter records per-request usage and cost. Writes are asynchronous and
// batched: the ledger must never add latency to, or fail, a request the client
// is waiting on.
package meter

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// Record is one row of the usage ledger.
type Record struct {
	RequestID        string
	TenantID         string
	KeyID            string
	Provider         string
	Model            string
	ModelAlias       string
	PromptTokens     int
	CompletionTokens int
	CostUSD          float64
	// SavedUSD is what this request would have cost had it not been a cache hit.
	// It is the number the "cost saved" dashboard panel sums.
	SavedUSD    float64
	CacheStatus string
	Streamed    bool
	StatusCode  int
	LatencyMS   int64
	Attempts    int
	ErrorKind   string
	CreatedAt   time.Time
}

// Sink persists batches of records.
type Sink interface {
	WriteBatch(ctx context.Context, rows []Record) error
	Close() error
}

// Stats reports the meter's own health, exposed as metrics.
type Stats struct {
	// Enqueued is the total records accepted.
	Enqueued atomic.Int64
	// Dropped is records discarded because the buffer was full.
	Dropped atomic.Int64
	// Written is records successfully persisted.
	Written atomic.Int64
	// Failed is records lost to write errors.
	Failed atomic.Int64
}

// Meter buffers records and flushes them to a Sink in batches.
type Meter struct {
	sink    Sink
	ch      chan Record
	batch   int
	flush   time.Duration
	log     *slog.Logger
	stats   Stats
	wg      sync.WaitGroup
	closing atomic.Bool
}

// Options configures a Meter.
type Options struct {
	Sink          Sink
	BufferSize    int
	BatchSize     int
	FlushInterval time.Duration
	Logger        *slog.Logger
}

// New builds a Meter and starts its writer goroutine.
func New(o Options) *Meter {
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	if o.BufferSize <= 0 {
		o.BufferSize = 4096
	}
	if o.BatchSize <= 0 {
		o.BatchSize = 128
	}
	if o.FlushInterval <= 0 {
		o.FlushInterval = 2 * time.Second
	}

	m := &Meter{
		sink:  o.Sink,
		ch:    make(chan Record, o.BufferSize),
		batch: o.BatchSize,
		flush: o.FlushInterval,
		log:   o.Logger,
	}
	m.wg.Add(1)
	go m.run()
	return m
}

// Record enqueues a usage row. It never blocks.
//
// When the buffer is full the row is dropped rather than applying backpressure:
// the ledger is for billing and dashboards, and stalling live traffic to protect
// its completeness would be the wrong trade. Drops are counted so the gap is
// visible rather than silent.
func (m *Meter) Record(r Record) {
	if m == nil || m.closing.Load() {
		return
	}
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now()
	}
	select {
	case m.ch <- r:
		m.stats.Enqueued.Add(1)
	default:
		n := m.stats.Dropped.Add(1)
		// Log on a ramp, not per drop: a saturated buffer would otherwise turn
		// one incident into a second one in the logging pipeline.
		if n == 1 || n%1000 == 0 {
			m.log.Warn("usage record dropped: meter buffer full", "dropped_total", n)
		}
	}
}

// Stats exposes counters for metrics reporting.
func (m *Meter) Stats() *Stats { return &m.stats }

// run drains the channel, flushing on batch size or interval.
func (m *Meter) run() {
	defer m.wg.Done()

	ticker := time.NewTicker(m.flush)
	defer ticker.Stop()

	buf := make([]Record, 0, m.batch)
	for {
		select {
		case r, ok := <-m.ch:
			if !ok {
				m.writeBatch(buf) // final drain on close
				return
			}
			buf = append(buf, r)
			if len(buf) >= m.batch {
				m.writeBatch(buf)
				buf = buf[:0]
			}
		case <-ticker.C:
			// Time-based flush bounds how long a low-traffic gateway's rows sit
			// unwritten, so dashboards stay near-live instead of lagging until
			// a batch happens to fill.
			if len(buf) > 0 {
				m.writeBatch(buf)
				buf = buf[:0]
			}
		}
	}
}

func (m *Meter) writeBatch(rows []Record) {
	if len(rows) == 0 {
		return
	}
	// Detached from any request context: a client disconnecting must not cancel
	// the write of usage that has already been incurred.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Copy: buf is reused by the caller as soon as this returns.
	batch := make([]Record, len(rows))
	copy(batch, rows)

	if err := m.sink.WriteBatch(ctx, batch); err != nil {
		m.stats.Failed.Add(int64(len(batch)))
		m.log.Error("usage batch write failed", "rows", len(batch), "err", err)
		return
	}
	m.stats.Written.Add(int64(len(batch)))
}

// Close stops accepting records and flushes what remains.
func (m *Meter) Close() error {
	if m == nil || !m.closing.CompareAndSwap(false, true) {
		return nil
	}
	close(m.ch)
	m.wg.Wait()
	return m.sink.Close()
}
