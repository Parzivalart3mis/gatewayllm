package meter

import (
	"strings"
	"sync"

	"github.com/yash/gatewayllm/internal/config"
)

// Pricing computes request cost from token counts.
type Pricing struct {
	mu     sync.RWMutex
	prices map[string]config.Price
}

// NewPricing builds a price table from config. Keys are matched case-insensitively
// as "provider/model", falling back to bare "model".
func NewPricing(prices map[string]config.Price) *Pricing {
	p := &Pricing{prices: make(map[string]config.Price, len(prices))}
	for k, v := range prices {
		p.prices[strings.ToLower(k)] = v
	}
	return p
}

// Cost returns the USD cost of a call.
//
// The second return reports whether a price was found. An unpriced model costs
// 0, and without this flag a missing price would be indistinguishable from a
// free call — quietly understating spend on exactly the dashboard built to track
// it.
func (p *Pricing) Cost(providerName, model string, promptTokens, completionTokens int) (float64, bool) {
	price, ok := p.lookup(providerName, model)
	if !ok {
		return 0, false
	}
	const perMillion = 1_000_000.0
	cost := float64(promptTokens)/perMillion*price.InputPerMillion +
		float64(completionTokens)/perMillion*price.OutputPerMillion
	return cost, true
}

// lookup resolves a price, preferring the provider-qualified entry so the same
// model served by two providers can be priced differently — which is the whole
// reason routing across providers saves money.
func (p *Pricing) lookup(providerName, model string) (config.Price, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if v, ok := p.prices[strings.ToLower(providerName+"/"+model)]; ok {
		return v, true
	}
	if v, ok := p.prices[strings.ToLower(model)]; ok {
		return v, true
	}
	return config.Price{}, false
}

// Has reports whether a price is configured for a provider/model pair.
func (p *Pricing) Has(providerName, model string) bool {
	_, ok := p.lookup(providerName, model)
	return ok
}
