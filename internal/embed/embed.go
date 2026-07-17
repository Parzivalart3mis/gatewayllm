// Package embed defines the Embedder abstraction used by the semantic cache and
// provides two implementations: a local Python sidecar and a hosted API.
// Which one runs is a config choice, not a code change.
package embed

import (
	"context"
	"errors"
	"fmt"
)

// Embedder turns text into vectors.
//
// It is deliberately separate from provider.Provider even though some backends
// implement both: the embedder sits on the cache lookup path, where latency and
// cost are charged to every semantic-tier miss, so it is swapped independently
// of which LLM serves completions.
type Embedder interface {
	// Embed returns one vector per input, in input order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimensions reports the vector size, which must match the Qdrant collection.
	Dimensions() int
	// Name identifies the embedder in logs and metrics.
	Name() string
}

// ErrEmbedderUnavailable reports that the embedding backend could not be reached.
var ErrEmbedderUnavailable = errors.New("embed: embedder unavailable")

// EmbedOne is a convenience wrapper for the single-text case, which is what the
// cache lookup path always needs.
func EmbedOne(ctx context.Context, e Embedder, text string) ([]float32, error) {
	vecs, err := e.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embed: expected 1 vector, got %d", len(vecs))
	}
	return vecs[0], nil
}
