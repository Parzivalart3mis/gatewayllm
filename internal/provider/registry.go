package provider

import (
	"fmt"

	"github.com/yash/gatewayllm/internal/config"
)

// Registry holds the live providers built from config, keyed by name.
type Registry struct {
	byName map[string]Provider
}

// Build constructs one Provider per enabled config entry. An unknown kind is a
// startup error rather than a runtime surprise on the first routed request.
func Build(cfgs []config.ProviderConfig) (*Registry, error) {
	r := &Registry{byName: make(map[string]Provider, len(cfgs))}
	for _, c := range cfgs {
		if !c.IsEnabled() {
			continue
		}
		opts := Options{
			Name:    c.Name,
			BaseURL: c.BaseURL,
			APIKey:  c.APIKey,
			Models:  c.Models,
			Timeout: c.Timeout,
		}
		var p Provider
		switch c.Kind {
		case "openai":
			p = NewOpenAI(opts)
		case "groq":
			p = NewGroq(opts)
		case "gemini":
			p = NewGemini(opts)
		case "mock":
			p = &Mock{ProviderName: c.Name, AvailableModels: c.Models, VectorSize: 384}
		default:
			return nil, fmt.Errorf("provider %q: unknown kind %q (want openai, groq, gemini, or mock)", c.Name, c.Kind)
		}
		r.byName[c.Name] = p
	}
	return r, nil
}

// Get returns the provider registered under name.
func (r *Registry) Get(name string) (Provider, bool) {
	p, ok := r.byName[name]
	return p, ok
}

// Names lists every registered provider name.
func (r *Registry) Names() []string {
	out := make([]string, 0, len(r.byName))
	for n := range r.byName {
		out = append(out, n)
	}
	return out
}

// Len reports how many providers are registered.
func (r *Registry) Len() int { return len(r.byName) }

// NewRegistry builds a Registry directly from providers, for tests.
func NewRegistry(ps ...Provider) *Registry {
	r := &Registry{byName: make(map[string]Provider, len(ps))}
	for _, p := range ps {
		r.byName[p.Name()] = p
	}
	return r
}
