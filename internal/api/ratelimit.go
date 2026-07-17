package api

import (
	"context"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// LimitResult is the outcome of a rate-limit check.
type LimitResult struct {
	Allowed bool
	// Limit is the tenant's ceiling, in requests per minute.
	Limit int
	// Remaining is the tokens left in the bucket after this check.
	Remaining int
	// RetryAfter is how long until a token is available; zero when allowed.
	RetryAfter time.Duration
}

// writeHeaders emits the conventional rate-limit headers so clients can back off
// on their own rather than discovering the limit by getting rejected.
func (r LimitResult) writeHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Set("X-RateLimit-Limit", strconv.Itoa(r.Limit))
	h.Set("X-RateLimit-Remaining", strconv.Itoa(max(0, r.Remaining)))
	if !r.Allowed && r.RetryAfter > 0 {
		// Round up: rounding down would advise a retry that is still too early.
		secs := int(r.RetryAfter.Seconds())
		if r.RetryAfter%time.Second != 0 {
			secs++
		}
		h.Set("Retry-After", strconv.Itoa(max(1, secs)))
	}
}

// Limiter decides whether a tenant's request may proceed.
type Limiter interface {
	Allow(ctx context.Context, t *Tenant) (LimitResult, error)
}

// LocalLimiter is an in-memory token bucket. Correct for a single replica; a
// fleet enforces N times the intended limit, so it exists as a fallback for when
// Redis is not configured, not as the production path.
type LocalLimiter struct {
	defaultRPM int
	burst      int
	now        func() time.Time

	mu      sync.Mutex
	buckets map[string]*bucket
}

type bucket struct {
	tokens float64
	last   time.Time
}

// NewLocalLimiter builds an in-memory limiter.
func NewLocalLimiter(defaultRPM, burst int) *LocalLimiter {
	if burst <= 0 {
		burst = defaultRPM
	}
	return &LocalLimiter{
		defaultRPM: defaultRPM,
		burst:      burst,
		now:        time.Now,
		buckets:    map[string]*bucket{},
	}
}

// limitFor returns the tenant's effective ceiling, honoring a per-key override.
func (l *LocalLimiter) limitFor(t *Tenant) int {
	if t.RPM > 0 {
		return t.RPM
	}
	return l.defaultRPM
}

func (l *LocalLimiter) Allow(_ context.Context, t *Tenant) (LimitResult, error) {
	rpm := l.limitFor(t)
	if rpm <= 0 {
		return LimitResult{Allowed: true, Limit: 0, Remaining: 0}, nil
	}
	capacity := float64(l.burst)
	if t.RPM > 0 {
		capacity = float64(t.RPM)
	}
	refillPerSec := float64(rpm) / 60

	// Bucket per key, not per tenant: a tenant's keys are independently limited
	// so one runaway integration cannot starve the others.
	key := t.KeyID
	if key == "" {
		key = t.ID
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	now := l.now()
	b, ok := l.buckets[key]
	if !ok {
		b = &bucket{tokens: capacity, last: now}
		l.buckets[key] = b
	}

	// Refill for elapsed time, capped at capacity.
	elapsed := now.Sub(b.last).Seconds()
	if elapsed > 0 {
		b.tokens = minF(capacity, b.tokens+elapsed*refillPerSec)
		b.last = now
	}

	if b.tokens < 1 {
		deficit := 1 - b.tokens
		return LimitResult{
			Allowed:    false,
			Limit:      rpm,
			Remaining:  0,
			RetryAfter: time.Duration(deficit / refillPerSec * float64(time.Second)),
		}, nil
	}
	b.tokens--
	return LimitResult{Allowed: true, Limit: rpm, Remaining: int(b.tokens)}, nil
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
