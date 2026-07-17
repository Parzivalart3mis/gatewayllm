package resilience

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/router"
)

// Config tunes the executor.
type Config struct {
	// MaxAttempts counts the initial try plus retries against a single provider.
	MaxAttempts int
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
	Breaker     BreakerConfig
}

// Executor runs a call across an ordered list of targets, retrying each and
// failing over to the next until one succeeds or all are exhausted.
type Executor struct {
	cfg     Config
	breaker *Breaker
	// OnAttempt, when set, is notified of every attempt outcome. Observability
	// hooks in here rather than the executor importing a metrics package.
	OnAttempt func(AttemptInfo)

	rngMu sync.Mutex
	rng   *rand.Rand
}

// AttemptInfo describes one attempt against one provider.
type AttemptInfo struct {
	Provider string
	Model    string
	Attempt  int
	Err      error
	Duration time.Duration
	// BreakerState is the circuit state observed before the attempt.
	BreakerState State
	// Skipped is true when the attempt never ran because the circuit was open.
	Skipped bool
}

// New builds an Executor.
func New(cfg Config, breaker *Breaker) *Executor {
	return &Executor{
		cfg:     cfg,
		breaker: breaker,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Breaker exposes the underlying breaker, for metrics reporting.
func (e *Executor) Breaker() *Breaker { return e.breaker }

// ErrNoTargets reports that routing produced nothing to call.
var ErrNoTargets = errors.New("resilience: no targets to try")

// Chat runs a non-streaming completion across targets, returning the first
// success along with the target that served it.
func (e *Executor) Chat(ctx context.Context, targets []router.Target, req *provider.ChatRequest) (*provider.ChatResponse, router.Target, error) {
	var resp *provider.ChatResponse
	target, err := e.do(ctx, targets, req, func(ctx context.Context, t router.Target, r *provider.ChatRequest) error {
		out, err := t.Provider.Chat(ctx, r)
		if err != nil {
			return err
		}
		resp = out
		return nil
	})
	return resp, target, err
}

// ChatStream runs a streaming completion across targets.
//
// Failover is only attempted while no bytes have reached the client. Once the
// first chunk is forwarded, the response has begun and switching providers
// mid-stream would splice two different completions together, so a later failure
// is returned to the caller rather than retried.
func (e *Executor) ChatStream(ctx context.Context, targets []router.Target, req *provider.ChatRequest, onChunk func(*provider.ChatChunk) error) (router.Target, error) {
	var started bool
	return e.do(ctx, targets, req, func(ctx context.Context, t router.Target, r *provider.ChatRequest) error {
		return t.Provider.ChatStream(ctx, r, func(c *provider.ChatChunk) error {
			started = true
			return onChunk(c)
		})
	}, func() bool { return started })
}

// do drives the failover loop. committed, when supplied, reports whether the
// operation has become non-retryable because output already reached the client.
func (e *Executor) do(
	ctx context.Context,
	targets []router.Target,
	req *provider.ChatRequest,
	call func(context.Context, router.Target, *provider.ChatRequest) error,
	committed ...func() bool,
) (router.Target, error) {
	if len(targets) == 0 {
		return router.Target{}, ErrNoTargets
	}
	isCommitted := func() bool {
		return len(committed) > 0 && committed[0]()
	}

	var errs []error
	for _, t := range targets {
		name := t.Provider.Name()

		allowed, state := e.breaker.Allow(ctx, name)
		if !allowed {
			e.report(AttemptInfo{Provider: name, Model: t.Model, BreakerState: state, Skipped: true})
			errs = append(errs, provider.Errorf(provider.KindUnavailable, name, "circuit open"))
			continue // fail straight over; the whole point is not to call it
		}

		err := e.attemptWithRetries(ctx, t, req, call, isCommitted, state)
		if err == nil {
			return t, nil
		}
		errs = append(errs, err)

		// The client is gone or output is already in flight: stop.
		if ctx.Err() != nil || isCommitted() {
			return t, err
		}
		pe := provider.AsError(err)
		if pe != nil && !pe.Failoverable() {
			return t, err // deterministic across providers; failing over is pointless
		}
	}
	return router.Target{}, joinErrors(errs)
}

// attemptWithRetries retries one target until the budget is spent.
func (e *Executor) attemptWithRetries(
	ctx context.Context,
	t router.Target,
	req *provider.ChatRequest,
	call func(context.Context, router.Target, *provider.ChatRequest) error,
	isCommitted func() bool,
	state State,
) error {
	name := t.Provider.Name()
	attempts := e.cfg.MaxAttempts
	if attempts < 1 {
		attempts = 1
	}

	// Give the provider its resolved model without mutating the caller's request,
	// which is shared with the cache and meter.
	local := *req
	local.ResolvedModel = t.Model

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		start := time.Now()
		err := call(ctx, t, &local)
		dur := time.Since(start)

		e.report(AttemptInfo{
			Provider: name, Model: t.Model, Attempt: attempt,
			Err: err, Duration: dur, BreakerState: state,
		})

		if err == nil {
			e.breaker.Record(ctx, name, true)
			return nil
		}
		lastErr = err

		pe := provider.AsError(err)
		// Only upstream faults count against the circuit. A client disconnect or
		// a malformed request says nothing about provider health, and counting
		// it would let bad callers trip the breaker for everyone.
		if pe != nil && countsAgainstProvider(pe.Kind) {
			e.breaker.Record(ctx, name, false)
		}

		if ctx.Err() != nil || isCommitted() {
			return err
		}
		if pe == nil || !pe.Retryable() || attempt == attempts {
			return err
		}
		if err := e.sleepBackoff(ctx, attempt, pe.RetryAfter); err != nil {
			return err
		}
	}
	return lastErr
}

// countsAgainstProvider reports whether a failure kind reflects provider health.
func countsAgainstProvider(k provider.Kind) bool {
	switch k {
	case provider.KindInvalidRequest, provider.KindContentFilter, provider.KindContextLength, provider.KindInternal:
		return false
	default:
		return true
	}
}

// sleepBackoff waits before the next attempt: exponential with full jitter, or
// the server-directed Retry-After when the upstream supplied one.
func (e *Executor) sleepBackoff(ctx context.Context, attempt int, retryAfter time.Duration) error {
	delay := retryAfter
	if delay <= 0 {
		backoff := e.cfg.BaseBackoff << (attempt - 1)
		if backoff > e.cfg.MaxBackoff || backoff <= 0 {
			backoff = e.cfg.MaxBackoff
		}
		// Full jitter: without it, concurrent failures retry in lockstep and
		// hammer a recovering provider at exactly the same instant.
		e.rngMu.Lock()
		delay = time.Duration(e.rng.Int63n(int64(backoff) + 1))
		e.rngMu.Unlock()
	}
	if delay > e.cfg.MaxBackoff {
		delay = e.cfg.MaxBackoff
	}

	t := time.NewTimer(delay)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (e *Executor) report(info AttemptInfo) {
	if e.OnAttempt != nil {
		e.OnAttempt(info)
	}
}

// joinErrors collapses per-target failures into one error. It preserves the most
// specific *provider.Error so the API surface can pick a sensible status code
// instead of flattening every failover exhaustion into a 500.
func joinErrors(errs []error) error {
	if len(errs) == 0 {
		return ErrNoTargets
	}
	if len(errs) == 1 {
		return errs[0]
	}

	// Prefer the first error a client can act on; otherwise the first upstream
	// fault. Rate limits rank highest so an exhausted quota surfaces as a 429.
	best := errs[0]
	bestRank := -1
	var msgs []string
	for _, err := range errs {
		msgs = append(msgs, err.Error())
		if pe := provider.AsError(err); pe != nil {
			if r := kindRank(pe.Kind); r > bestRank {
				bestRank, best = r, err
			}
		}
	}

	if pe := provider.AsError(best); pe != nil {
		return &provider.Error{
			Kind:     pe.Kind,
			Provider: pe.Provider,
			Status:   pe.Status,
			Message:  fmt.Sprintf("all providers failed: %s", strings.Join(msgs, "; ")),
			Err:      best,
		}
	}
	return fmt.Errorf("all providers failed: %s", strings.Join(msgs, "; "))
}

// kindRank orders failure kinds by how informative they are to the client.
func kindRank(k provider.Kind) int {
	switch k {
	case provider.KindRateLimit:
		return 5
	case provider.KindInvalidRequest, provider.KindContextLength, provider.KindContentFilter:
		return 4
	case provider.KindAuth:
		return 3
	case provider.KindTimeout:
		return 2
	case provider.KindUnavailable:
		return 1
	default:
		return 0
	}
}
