// Package obs wires OpenTelemetry tracing and Prometheus metrics. Other
// packages report events through narrow interfaces they define themselves, so
// nothing in the core depends on a metrics library.
package obs

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/yash/gatewayllm/internal/cache"
)

// Metrics holds every Prometheus collector the gateway exports.
type Metrics struct {
	RequestsTotal   *prometheus.CounterVec
	RequestDuration *prometheus.HistogramVec

	CacheLookups    *prometheus.CounterVec
	CacheLookupTime *prometheus.HistogramVec
	CacheWrites     *prometheus.CounterVec
	// Similarity records the score of every semantic candidate considered,
	// split by whether it was accepted. Comparing the two distributions is how
	// the threshold gets tuned with evidence instead of guesswork.
	Similarity *prometheus.HistogramVec
	// FalseHits counts hits an operator or an eval marked wrong. Hit rate alone
	// is a vanity metric: it is only meaningful next to this.
	FalseHits prometheus.Counter

	ProviderAttempts *prometheus.CounterVec
	ProviderDuration *prometheus.HistogramVec
	BreakerState     *prometheus.GaugeVec

	TokensTotal   *prometheus.CounterVec
	CostUSDTotal  *prometheus.CounterVec
	SavedUSDTotal *prometheus.CounterVec

	MeterDropped prometheus.Counter
	MeterFailed  prometheus.Counter
}

// latencyBuckets span a cache hit (sub-millisecond) to a slow completion (a
// minute). The default Prometheus buckets top out at 10s and would put every
// real LLM call in +Inf, hiding exactly the tail worth watching.
var latencyBuckets = []float64{
	0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60,
}

// NewMetrics registers all collectors.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	f := promauto.With(reg)
	return &Metrics{
		RequestsTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Total API requests.",
		}, []string{"path", "status", "cache"}),

		RequestDuration: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "End-to-end request latency.",
			Buckets: latencyBuckets,
		}, []string{"path", "cache"}),

		CacheLookups: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cache_lookups_total",
			Help: "Cache lookups by outcome and tier.",
		}, []string{"status", "tier"}),

		CacheLookupTime: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_cache_lookup_duration_seconds",
			Help:    "Cache lookup latency by tier.",
			Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1},
		}, []string{"tier"}),

		CacheWrites: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cache_writes_total",
			Help: "Cache writes by tier and result.",
		}, []string{"tier", "result"}),

		Similarity: f.NewHistogramVec(prometheus.HistogramOpts{
			Name: "gateway_cache_similarity",
			Help: "Semantic similarity scores of candidate matches.",
			// Tight buckets near 1: everything interesting happens between 0.8
			// and 1.0, and uniform buckets would compress it into two bars.
			Buckets: []float64{0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0},
		}, []string{"accepted"}),

		FalseHits: f.NewCounter(prometheus.CounterOpts{
			Name: "gateway_cache_false_hits_total",
			Help: "Cache hits reported as semantically wrong. Track alongside hit rate.",
		}),

		ProviderAttempts: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_provider_attempts_total",
			Help: "Provider call attempts by outcome.",
		}, []string{"provider", "model", "result"}),

		ProviderDuration: f.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_provider_duration_seconds",
			Help:    "Provider call latency.",
			Buckets: latencyBuckets,
		}, []string{"provider", "model"}),

		BreakerState: f.NewGaugeVec(prometheus.GaugeOpts{
			Name: "gateway_breaker_state",
			Help: "Circuit breaker state per provider (0=closed, 1=half_open, 2=open).",
		}, []string{"provider"}),

		TokensTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_tokens_total",
			Help: "Tokens processed.",
		}, []string{"provider", "model", "kind"}),

		CostUSDTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cost_usd_total",
			Help: "Cumulative provider spend in USD.",
		}, []string{"provider", "model"}),

		SavedUSDTotal: f.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_saved_usd_total",
			Help: "Cumulative USD not spent because of cache hits.",
		}, []string{"model", "tier"}),

		MeterDropped: f.NewCounter(prometheus.CounterOpts{
			Name: "gateway_meter_dropped_total",
			Help: "Usage records dropped because the meter buffer was full.",
		}),

		MeterFailed: f.NewCounter(prometheus.CounterOpts{
			Name: "gateway_meter_failed_total",
			Help: "Usage records lost to write errors.",
		}),
	}
}

// --- cache.Metrics implementation ---

// RecordLookup implements cache.Metrics.
func (m *Metrics) RecordLookup(status cache.Status, tier string, dur time.Duration) {
	m.CacheLookups.WithLabelValues(string(status), tier).Inc()
	m.CacheLookupTime.WithLabelValues(tier).Observe(dur.Seconds())
}

// RecordSimilarity implements cache.Metrics.
func (m *Metrics) RecordSimilarity(score float64, accepted bool) {
	m.Similarity.WithLabelValues(boolLabel(accepted)).Observe(score)
}

// RecordWrite implements cache.Metrics.
func (m *Metrics) RecordWrite(tier string, err error) {
	result := "ok"
	if err != nil {
		result = "error"
	}
	m.CacheWrites.WithLabelValues(tier, result).Inc()
}

// BreakerStateValue maps a breaker state onto its gauge value.
func BreakerStateValue(s string) float64 {
	switch s {
	case "open":
		return 2
	case "half_open":
		return 1
	default:
		return 0
	}
}

func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

var _ cache.Metrics = (*Metrics)(nil)
