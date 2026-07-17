package cache

import (
	"context"
	"crypto/sha256"
	"fmt"

	"github.com/google/uuid"
	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/store"
)

// SemanticTier is the Qdrant-backed similarity cache: one point per cached
// prompt, vector = embedding, payload = completion + the params it was made
// under.
type SemanticTier struct {
	q   *store.Qdrant
	cfg config.SemanticCache
}

// NewSemanticTier builds the Qdrant tier.
func NewSemanticTier(q *store.Qdrant, cfg config.SemanticCache) *SemanticTier {
	return &SemanticTier{q: q, cfg: cfg}
}

// EnsureCollection creates the backing collection if needed.
func (t *SemanticTier) EnsureCollection(ctx context.Context, dims int) error {
	return t.q.EnsureCollection(ctx, t.cfg.Collection, dims, store.Cosine)
}

// pointID derives a stable ID for a cached prompt.
//
// Qdrant requires a UUID or unsigned integer ID, so the digest is folded into a
// UUID. Deriving it from tenant + model + prompt rather than using a random ID
// means re-caching the same prompt overwrites its entry instead of accumulating
// near-duplicate points that all match and slow every future search.
func pointID(tenantID, model, prompt string) string {
	h := sha256.Sum256([]byte(tenantID + "\x00" + model + "\x00" + prompt))
	return uuid.NewSHA1(uuid.NameSpaceOID, h[:]).String()
}

// tenantFilter restricts a search to one tenant's entries.
//
// This is a correctness boundary, not an optimization: without it, one tenant's
// cached completions would be served to another, leaking prompt content across
// customers. It is applied as a pre-filter so Qdrant never scores foreign points.
func tenantFilter(tenantID, model string) map[string]any {
	return map[string]any{
		"must": []map[string]any{
			{"key": "tenant", "match": map[string]any{"value": tenantID}},
			{"key": "model", "match": map[string]any{"value": model}},
		},
	}
}

// Search finds the closest cached prompt above threshold. It returns nil when
// nothing qualifies.
func (t *SemanticTier) Search(ctx context.Context, tenantID string, req *provider.ChatRequest, vec []float32, threshold float64) (*Entry, float64, error) {
	res, err := t.q.Search(ctx, t.cfg.Collection, store.SearchRequest{
		Vector: vec,
		// One result is enough: the top match is the best candidate, and if it
		// fails the compatibility guard, weaker matches are unlikely to pass a
		// check the strongest one failed.
		Limit: 1,
		// Push the threshold into Qdrant so weak matches never cross the wire.
		ScoreThreshold: threshold,
		Filter:         tenantFilter(tenantID, req.Model),
	})
	if err != nil {
		return nil, 0, fmt.Errorf("semantic search: %w", err)
	}
	if len(res) == 0 {
		return nil, 0, nil
	}

	top := res[0]
	// Defense in depth: trust the score we compare against, not just the one the
	// server was asked to filter on.
	if top.Score < threshold {
		return nil, top.Score, nil
	}

	entry, err := entryFromPayload(top.Payload)
	if err != nil {
		return nil, top.Score, fmt.Errorf("semantic search: %w", err)
	}
	return entry, top.Score, nil
}

// Upsert writes a cached completion into the collection.
func (t *SemanticTier) Upsert(ctx context.Context, tenantID string, req *provider.ChatRequest, e *Entry, vec []float32) error {
	prompt := PromptText(req)
	return t.q.Upsert(ctx, t.cfg.Collection, []store.Point{{
		ID:     pointID(tenantID, req.Model, prompt),
		Vector: vec,
		Payload: map[string]any{
			"tenant":            tenantID,
			"model":             req.Model,
			"upstream_model":    e.UpstreamModel,
			"provider":          e.Provider,
			"completion":        e.Completion,
			"prompt":            prompt,
			"temperature":       e.Temp,
			"max_tokens":        e.MaxTokens,
			"prompt_tokens":     e.PromptTokens,
			"completion_tokens": e.CompletionTokens,
			"created_at":        e.CreatedAt,
		},
	}})
}

// entryFromPayload rebuilds an Entry from a Qdrant payload. JSON numbers decode
// as float64, so numeric fields are converted rather than type-asserted to int.
func entryFromPayload(p map[string]any) (*Entry, error) {
	completion, ok := p["completion"].(string)
	if !ok {
		return nil, fmt.Errorf("payload missing completion")
	}
	e := &Entry{
		Completion:       completion,
		Model:            str(p["model"]),
		UpstreamModel:    str(p["upstream_model"]),
		Provider:         str(p["provider"]),
		Prompt:           str(p["prompt"]),
		Temp:             num(p["temperature"]),
		MaxTokens:        int(num(p["max_tokens"])),
		PromptTokens:     int(num(p["prompt_tokens"])),
		CompletionTokens: int(num(p["completion_tokens"])),
		CreatedAt:        int64(num(p["created_at"])),
	}
	return e, nil
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func num(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	default:
		return 0
	}
}

// Ping reports whether Qdrant is reachable.
func (t *SemanticTier) Ping(ctx context.Context) error {
	return t.q.Ping(ctx)
}
