// Package resilience wraps every provider call with timeouts, retries, and a
// circuit breaker, and drives failover across the router's ordered targets.
package resilience

import (
	"context"
	"sync"
	"time"
)

// State is a circuit breaker's position in its lifecycle.
type State string

const (
	// StateClosed passes calls through; the provider is considered healthy.
	StateClosed State = "closed"
	// StateOpen rejects calls immediately; the provider is considered down.
	StateOpen State = "open"
	// StateHalfOpen admits a limited number of probes to test recovery.
	StateHalfOpen State = "half_open"
)

// BreakerStore persists breaker state. Backing it with Redis makes the state
// shared across replicas, so one replica discovering a dead provider spares the
// others from rediscovering it. A local implementation is used when Redis is
// not configured.
type BreakerStore interface {
	// Snapshot returns the current state for a provider.
	Snapshot(ctx context.Context, provider string) (State, error)
	// RecordSuccess reports a successful call and returns the resulting state.
	RecordSuccess(ctx context.Context, provider string) (State, error)
	// RecordFailure reports a failed call and returns the resulting state.
	RecordFailure(ctx context.Context, provider string) (State, error)
}

// BreakerConfig tunes breaker behaviour.
type BreakerConfig struct {
	// FailureThreshold is the consecutive-failure count that opens the circuit.
	FailureThreshold int
	// OpenDuration is how long a circuit stays open before admitting probes.
	OpenDuration time.Duration
	// HalfOpenProbes is the number of consecutive half-open successes that close
	// the circuit.
	HalfOpenProbes int
}

// LocalBreakerStore keeps breaker state in memory. Used when Redis is absent and
// in tests. State is per-replica, so a fleet converges more slowly than with the
// shared store, but each replica still protects itself.
type LocalBreakerStore struct {
	cfg BreakerConfig
	mu  sync.Mutex
	m   map[string]*breakerEntry
	now func() time.Time
}

type breakerEntry struct {
	state        State
	failures     int
	successes    int
	openedAt     time.Time
	probesInFlgt int
}

// NewLocalBreakerStore builds an in-memory breaker store.
func NewLocalBreakerStore(cfg BreakerConfig) *LocalBreakerStore {
	return &LocalBreakerStore{cfg: cfg, m: map[string]*breakerEntry{}, now: time.Now}
}

func (s *LocalBreakerStore) entry(p string) *breakerEntry {
	e, ok := s.m[p]
	if !ok {
		e = &breakerEntry{state: StateClosed}
		s.m[p] = e
	}
	return e
}

func (s *LocalBreakerStore) Snapshot(_ context.Context, p string) (State, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.entry(p)
	// An open circuit becomes half-open once its cooldown elapses. Evaluating
	// this on read means no background timer is needed.
	if e.state == StateOpen && s.now().Sub(e.openedAt) >= s.cfg.OpenDuration {
		e.state = StateHalfOpen
		e.successes = 0
	}
	return e.state, nil
}

func (s *LocalBreakerStore) RecordSuccess(_ context.Context, p string) (State, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.entry(p)
	switch e.state {
	case StateHalfOpen:
		e.successes++
		if e.successes >= s.cfg.HalfOpenProbes {
			e.state = StateClosed
			e.failures = 0
			e.successes = 0
		}
	default:
		e.state = StateClosed
		e.failures = 0
	}
	return e.state, nil
}

func (s *LocalBreakerStore) RecordFailure(_ context.Context, p string) (State, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	e := s.entry(p)
	// A failure during a probe means the provider is still down: reopen
	// immediately rather than burning the remaining probe budget.
	if e.state == StateHalfOpen {
		e.state = StateOpen
		e.openedAt = s.now()
		e.successes = 0
		return e.state, nil
	}
	e.failures++
	if e.failures >= s.cfg.FailureThreshold {
		e.state = StateOpen
		e.openedAt = s.now()
	}
	return e.state, nil
}

// Breaker guards calls to one provider using a BreakerStore.
type Breaker struct {
	store   BreakerStore
	enabled bool
}

// NewBreaker builds a breaker over the given store. When enabled is false, every
// call is allowed and outcomes are not recorded.
func NewBreaker(store BreakerStore, enabled bool) *Breaker {
	return &Breaker{store: store, enabled: enabled}
}

// Allow reports whether a call to the provider may proceed.
func (b *Breaker) Allow(ctx context.Context, p string) (bool, State) {
	if !b.enabled {
		return true, StateClosed
	}
	st, err := b.store.Snapshot(ctx, p)
	if err != nil {
		// The breaker is a safety device, not a gate: if its own store is
		// unreachable, failing the request would turn a Redis outage into a
		// total outage. Allow the call and let the provider decide.
		return true, StateClosed
	}
	return st != StateOpen, st
}

// Record reports a call outcome to the breaker.
func (b *Breaker) Record(ctx context.Context, p string, success bool) {
	if !b.enabled {
		return
	}
	if success {
		_, _ = b.store.RecordSuccess(ctx, p)
		return
	}
	_, _ = b.store.RecordFailure(ctx, p)
}

// State reports the current state of a provider's circuit, for metrics.
func (b *Breaker) State(ctx context.Context, p string) State {
	if !b.enabled {
		return StateClosed
	}
	st, err := b.store.Snapshot(ctx, p)
	if err != nil {
		return StateClosed
	}
	return st
}
