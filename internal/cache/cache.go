package cache

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/embed"
	"github.com/yash/gatewayllm/internal/provider"
)

// Status is the outcome of a cache lookup, reported to the client via the
// X-Cache header and recorded in the usage log.
type Status string

const (
	// StatusExactHit means the request's digest matched a stored entry.
	StatusExactHit Status = "exact_hit"
	// StatusSemanticHit means a sufficiently similar prompt was found.
	StatusSemanticHit Status = "semantic_hit"
	// StatusMiss means nothing matched and the provider was called.
	StatusMiss Status = "miss"
	// StatusBypass means caching was deliberately skipped for this request.
	StatusBypass Status = "bypass"
	// StatusError means the cache failed and the request was served as a miss.
	StatusError Status = "error"
)

// Entry is a cached completion plus the parameters it was produced under. The
// parameters are stored, not just the text, because serving an entry requires
// proving it was generated under compatible settings.
type Entry struct {
	Completion string `json:"completion"`
	// Model is the client-facing alias this entry answers for. The
	// compatibility guard and the Qdrant tenant/model filter both match on it,
	// so it must be the alias, not the upstream model.
	Model string `json:"model"`
	// UpstreamModel is the provider's actual model, retained so a cache hit can
	// price the spend it avoided. Pricing keys on the upstream model, so using
	// the alias here would miss the price table and book every hit's savings as
	// zero.
	UpstreamModel string  `json:"upstream_model"`
	Provider      string  `json:"provider"`
	Temp          float64 `json:"temperature"`
	MaxTokens     int     `json:"max_tokens"`
	// PromptTokens and CompletionTokens are the original call's usage, replayed
	// so a cache hit reports the tokens it saved rather than zeros.
	PromptTokens     int   `json:"prompt_tokens"`
	CompletionTokens int   `json:"completion_tokens"`
	CreatedAt        int64 `json:"created_at"`
	// Prompt is the normalized text this entry was keyed on. Stored only in the
	// semantic tier, where it is needed to audit false hits.
	Prompt string `json:"prompt,omitempty"`
}

// Age reports how long ago the entry was written.
func (e *Entry) Age() time.Duration {
	return time.Since(time.Unix(e.CreatedAt, 0))
}

// Result is a lookup outcome.
type Result struct {
	Status Status
	Entry  *Entry
	// Similarity is the score that produced a semantic hit; 0 otherwise.
	Similarity float64
	// Vector is the prompt embedding, retained on a semantic miss so the write
	// path can reuse it instead of embedding the same prompt twice.
	Vector []float32
}

// Hit reports whether the result can be served without calling a provider.
func (r Result) Hit() bool {
	return r.Status == StatusExactHit || r.Status == StatusSemanticHit
}

// Metrics receives cache events. The cache reports outcomes rather than
// importing a metrics package, keeping the dependency pointing outward.
type Metrics interface {
	RecordLookup(status Status, tier string, dur time.Duration)
	RecordSimilarity(score float64, accepted bool)
	RecordWrite(tier string, err error)
}

// Cache orchestrates the exact and semantic tiers.
type Cache struct {
	cfg      config.Cache
	exact    *ExactTier
	semantic *SemanticTier
	embedder embed.Embedder
	metrics  Metrics
	log      *slog.Logger
}

// Options configures a Cache.
type Options struct {
	Config   config.Cache
	Exact    *ExactTier
	Semantic *SemanticTier
	Embedder embed.Embedder
	Metrics  Metrics
	Logger   *slog.Logger
}

// New builds a Cache. A nil semantic tier disables that tier, which is how the
// gateway runs with only the Redis tier.
func New(o Options) *Cache {
	log := o.Logger
	if log == nil {
		log = slog.Default()
	}
	return &Cache{
		cfg:      o.Config,
		exact:    o.Exact,
		semantic: o.Semantic,
		embedder: o.Embedder,
		metrics:  o.Metrics,
		log:      log,
	}
}

// BypassReason explains why a request was not cached, for logs and debugging.
type BypassReason string

const (
	// BypassDisabled means caching is off in config.
	BypassDisabled BypassReason = "cache_disabled"
	// BypassHeader means the client asked to skip the cache.
	BypassHeader BypassReason = "no_cache_header"
	// BypassTemperature means the request asked for a non-deterministic answer.
	BypassTemperature BypassReason = "temperature_too_high"
	// BypassStopSequences means the request carries settings the cache does not
	// model precisely enough to match on.
	BypassStopSequences BypassReason = "unsupported_params"
)

// Cacheable reports whether a request may be served from or written to cache,
// and why not when it may not.
//
// This is the cache's most important guard. A cache that answers a question
// nobody asked is worse than no cache: the latency win is invisible, but a wrong
// answer is not, and it destroys trust in the whole gateway.
func (c *Cache) Cacheable(req *provider.ChatRequest, noCacheHeader bool) (bool, BypassReason) {
	if !c.cfg.Enabled {
		return false, BypassDisabled
	}
	// An explicit opt-out is absolute: a caller asking for a fresh answer has a
	// reason, and overriding them would make the gateway untrustworthy.
	if noCacheHeader {
		return false, BypassHeader
	}
	// High temperature is a request for variety. Returning the same completion
	// every time would technically satisfy the API contract while defeating the
	// caller's actual intent — the one case where a correct cache hit is wrong.
	if req.Temp() > c.cfg.MaxTemperature {
		return false, BypassTemperature
	}
	return true, ""
}

// Lookup checks both tiers. It never returns an error: a broken cache degrades
// to a miss, because the provider call is always a correct fallback and failing
// the request would make the cache a liability rather than an optimization.
func (c *Cache) Lookup(ctx context.Context, tenantID string, req *provider.ChatRequest) Result {
	start := time.Now()

	// Tier 1: exact. No embedding call, so a repeat costs one Redis round trip.
	key := ExactKey(tenantID, req)
	entry, err := c.exact.Get(ctx, key)
	switch {
	case err != nil:
		c.log.Warn("exact cache lookup failed, serving as miss", "err", err)
		c.record(StatusError, "exact", start)
	case entry != nil:
		c.record(StatusExactHit, "exact", start)
		return Result{Status: StatusExactHit, Entry: entry}
	}

	// Tier 2: semantic. Only reached on an exact miss, because embedding costs
	// far more than the Redis lookup that would have made it unnecessary.
	if c.semantic == nil || !c.cfg.Semantic.Enabled {
		c.record(StatusMiss, "exact", start)
		return Result{Status: StatusMiss}
	}

	vec, err := embed.EmbedOne(ctx, c.embedder, PromptText(req))
	if err != nil {
		// The embedder being down must not fail the request; it just means this
		// request cannot use or populate the semantic tier.
		c.log.Warn("embedding failed, serving as miss", "err", err)
		c.record(StatusError, "semantic", start)
		return Result{Status: StatusMiss}
	}

	match, score, err := c.semantic.Search(ctx, tenantID, req, vec, c.cfg.Semantic.Threshold)
	if err != nil {
		c.log.Warn("semantic cache lookup failed, serving as miss", "err", err)
		c.record(StatusError, "semantic", start)
		return Result{Status: StatusMiss, Vector: vec}
	}
	if match == nil {
		c.record(StatusMiss, "semantic", start)
		return Result{Status: StatusMiss, Vector: vec}
	}

	// A vector match means the prompts are similar. It does not mean the entry
	// is safe to serve: the parameters must also be compatible, or a
	// max_tokens=50 request could be answered with a 2000-token completion.
	if !c.compatible(req, match) {
		c.recordSimilarity(score, false)
		c.record(StatusMiss, "semantic", start)
		return Result{Status: StatusMiss, Vector: vec}
	}

	c.recordSimilarity(score, true)
	c.record(StatusSemanticHit, "semantic", start)
	return Result{Status: StatusSemanticHit, Entry: match, Similarity: score, Vector: vec}
}

// compatible reports whether a semantically matched entry was produced under
// parameters close enough to the current request to stand in for it.
func (c *Cache) compatible(req *provider.ChatRequest, e *Entry) bool {
	// Different alias means a different contract: the caller chose a model, and
	// handing back another model's answer silently breaks that choice.
	if !strings.EqualFold(e.Model, req.Model) {
		return false
	}
	// Temperature shapes the answer's character. A near-equal value is fine;
	// a deterministic request answered by a creative one's output is not.
	if math.Abs(e.Temp-req.Temp()) > c.cfg.Semantic.MaxTempDelta {
		return false
	}
	// A cached completion longer than this request's ceiling would exceed a
	// limit the caller set deliberately.
	if want := req.MaxTok(); want > 0 && e.CompletionTokens > want {
		return false
	}
	// Stale entries are rejected on read rather than swept: it costs nothing on
	// the hot path and needs no background job.
	if c.cfg.Semantic.TTL > 0 && e.Age() > c.cfg.Semantic.TTL {
		return false
	}
	return true
}

// Store writes a completed response into both tiers. It is called after the
// response has been sent, so an error here costs nothing but a future miss.
func (c *Cache) Store(ctx context.Context, tenantID string, req *provider.ChatRequest, entry *Entry, vec []float32) {
	entry.CreatedAt = time.Now().Unix()

	key := ExactKey(tenantID, req)
	if err := c.exact.Set(ctx, key, entry, c.cfg.ExactTTL); err != nil {
		c.log.Warn("exact cache write failed", "err", err)
		c.recordWrite("exact", err)
	} else {
		c.recordWrite("exact", nil)
	}

	if c.semantic == nil || !c.cfg.Semantic.Enabled {
		return
	}
	// Reuse the lookup's vector when present: re-embedding the same prompt would
	// double the embedding cost of every miss.
	if vec == nil {
		var err error
		vec, err = embed.EmbedOne(ctx, c.embedder, PromptText(req))
		if err != nil {
			c.log.Warn("embedding failed on cache write", "err", err)
			c.recordWrite("semantic", err)
			return
		}
	}
	entry.Prompt = PromptText(req)
	if err := c.semantic.Upsert(ctx, tenantID, req, entry, vec); err != nil {
		c.log.Warn("semantic cache write failed", "err", err)
		c.recordWrite("semantic", err)
		return
	}
	c.recordWrite("semantic", nil)
}

// Ping reports whether the cache's stores are reachable.
func (c *Cache) Ping(ctx context.Context) error {
	if c.exact != nil {
		if err := c.exact.Ping(ctx); err != nil {
			return err
		}
	}
	if c.semantic != nil && c.cfg.Semantic.Enabled {
		if err := c.semantic.Ping(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (c *Cache) record(s Status, tier string, start time.Time) {
	if c.metrics != nil {
		c.metrics.RecordLookup(s, tier, time.Since(start))
	}
}

func (c *Cache) recordWrite(tier string, err error) {
	if c.metrics != nil {
		c.metrics.RecordWrite(tier, err)
	}
}

func (c *Cache) recordSimilarity(score float64, accepted bool) {
	if c.metrics != nil {
		c.metrics.RecordSimilarity(score, accepted)
	}
}

// ErrNotFound reports a missing entry.
var ErrNotFound = errors.New("cache: entry not found")
