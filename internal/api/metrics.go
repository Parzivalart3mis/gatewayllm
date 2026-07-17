package api

import "time"

// Metrics receives request and usage events.
//
// Declared here as a narrow interface rather than importing the obs package, so
// the HTTP layer depends on the events it emits rather than on Prometheus. A nil
// Metrics disables reporting.
type Metrics interface {
	// RecordRequest reports a completed request.
	RecordRequest(path string, status int, cacheStatus string, dur time.Duration)
	// RecordUsage reports tokens and money for one request. savedUSD is non-zero
	// only for cache hits, where it is what the call would have cost.
	RecordUsage(providerName, model string, promptTokens, completionTokens int, costUSD, savedUSD float64, cacheStatus string)
}
