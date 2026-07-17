// Package router resolves a client-facing model alias into an ordered list of
// provider targets. The order it returns is the failover order: resilience tries
// them in sequence until one succeeds.
package router

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/provider"
)

// Target is a resolved routing destination.
type Target struct {
	// Provider is the live adapter to call.
	Provider provider.Provider
	// Model is the upstream model ID to send, which may differ from the alias
	// the client requested.
	Model string
	// Alias is the client-facing name that resolved to this target, retained for
	// metrics and the usage log.
	Alias string
}

// Router maps aliases onto targets.
type Router struct {
	aliases map[string]resolvedAlias
	// defaultAlias serves unknown models; empty means reject them.
	defaultAlias string
	rng          *rand.Rand
	mu           sync.Mutex // guards rng, which is not safe for concurrent use
}

type resolvedAlias struct {
	name     string
	strategy string
	targets  []Target
	// weights parallels targets, used only by the weighted strategy.
	weights []int
	// totalWeight is precomputed for the weighted strategy.
	totalWeight int
}

// randSeed draws a seed from the OS entropy source so replicas do not share an
// identical weighted-selection sequence.
func randSeed() int64 {
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		return time.Now().UnixNano()
	}
	return int64(binary.LittleEndian.Uint64(b[:]))
}

// New builds a Router from config, resolving every provider reference against
// the registry. Config validation has already checked these references, so a
// failure here means config and registry disagree — a programming error worth
// failing startup over.
func New(cfg config.Router, reg *provider.Registry) (*Router, error) {
	r := &Router{
		aliases:      make(map[string]resolvedAlias, len(cfg.Aliases)),
		defaultAlias: cfg.DefaultAlias,
		rng:          rand.New(rand.NewSource(randSeed())),
	}
	for name, a := range cfg.Aliases {
		ra := resolvedAlias{name: name, strategy: a.Strategy}
		for _, t := range a.Targets {
			p, ok := reg.Get(t.Provider)
			if !ok {
				return nil, fmt.Errorf("router: alias %q references unregistered provider %q", name, t.Provider)
			}
			ra.targets = append(ra.targets, Target{Provider: p, Model: t.Model, Alias: name})
			ra.totalWeight += t.Weight
		}
		if a.Strategy == "weighted" {
			// Keep the weights alongside the targets for selection.
			ra.weights = make([]int, len(a.Targets))
			for i, t := range a.Targets {
				ra.weights[i] = t.Weight
			}
		}
		r.aliases[name] = ra
	}
	return r, nil
}

// Route resolves a requested model into an ordered failover list. The first
// element is the primary; later elements are fallbacks tried in order.
func (r *Router) Route(model string) ([]Target, error) {
	a, ok := r.aliases[model]
	if !ok {
		if r.defaultAlias == "" {
			return nil, fmt.Errorf("model %q is not a configured alias", model)
		}
		a, ok = r.aliases[r.defaultAlias]
		if !ok {
			return nil, fmt.Errorf("default alias %q is not configured", r.defaultAlias)
		}
	}
	if len(a.targets) == 0 {
		return nil, fmt.Errorf("alias %q has no targets", a.name)
	}

	switch a.strategy {
	case "weighted":
		return r.weightedOrder(a), nil
	default: // priority: config order is the failover order
		out := make([]Target, len(a.targets))
		copy(out, a.targets)
		return out, nil
	}
}

// weightedOrder returns targets ordered by weighted random selection without
// replacement: the primary is chosen by weight, and the remainder stay available
// as fallbacks so a weighted alias still fails over rather than giving up.
func (r *Router) weightedOrder(a resolvedAlias) []Target {
	remaining := make([]Target, len(a.targets))
	copy(remaining, a.targets)
	weights := make([]int, len(a.weights))
	copy(weights, a.weights)

	total := a.totalWeight
	out := make([]Target, 0, len(remaining))

	r.mu.Lock()
	defer r.mu.Unlock()

	for len(remaining) > 0 {
		if total <= 0 {
			// Weights exhausted or misconfigured; append the rest in order.
			out = append(out, remaining...)
			break
		}
		pick := r.rng.Intn(total)
		idx := 0
		for i, w := range weights {
			if pick < w {
				idx = i
				break
			}
			pick -= w
		}
		out = append(out, remaining[idx])
		total -= weights[idx]
		remaining = append(remaining[:idx], remaining[idx+1:]...)
		weights = append(weights[:idx], weights[idx+1:]...)
	}
	return out
}

// Aliases lists configured alias names, sorted, for the /v1/models surface.
func (r *Router) Aliases() []string {
	out := make([]string, 0, len(r.aliases))
	for n := range r.aliases {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Has reports whether model resolves to a configured alias.
func (r *Router) Has(model string) bool {
	_, ok := r.aliases[model]
	return ok
}
