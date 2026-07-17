package cache

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/store"
)

// fakeQdrant stands in for a real Qdrant. It records what the tier sends and
// returns a scripted result, which is enough to pin down the two things that
// matter: that the tenant filter is applied, and that the threshold is enforced.
type fakeQdrant struct {
	searchBody map[string]any
	upsertBody map[string]any
	result     []store.ScoredPoint
}

func (f *fakeQdrant) server(t *testing.T) *store.Qdrant {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]any{}
		_ = json.NewDecoder(r.Body).Decode(&body)

		switch {
		case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/points/search"):
			f.searchBody = body
			_ = json.NewEncoder(w).Encode(map[string]any{"result": f.result})
		case r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/points"):
			f.upsertBody = body
			_ = json.NewEncoder(w).Encode(map[string]any{"result": map[string]any{"status": "ok"}})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{"result": map[string]any{}})
		}
	}))
	t.Cleanup(srv.Close)
	return store.NewQdrant(srv.URL, "", 2*time.Second)
}

func testSemanticTier(t *testing.T, f *fakeQdrant) *SemanticTier {
	t.Helper()
	return NewSemanticTier(f.server(t), config.SemanticCache{
		Collection: "test_collection",
		Threshold:  0.95,
		TTL:        24 * time.Hour,
	})
}

// TestSemantic_SearchAppliesTenantFilter guards a security boundary: without the
// filter, one tenant's cached completions would be scored against another's
// prompts and could be served across customers.
func TestSemantic_SearchAppliesTenantFilter(t *testing.T) {
	fq := &fakeQdrant{}
	tier := testSemanticTier(t, fq)

	r := req("gpt-4o", f(0.2), nil, user("hello"))
	_, _, err := tier.Search(context.Background(), "tenant-a", r, []float32{0.1, 0.2}, 0.95)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	filter, ok := fq.searchBody["filter"].(map[string]any)
	if !ok {
		t.Fatal("search must carry a filter: without it tenants share cached completions")
	}
	must, ok := filter["must"].([]any)
	if !ok || len(must) != 2 {
		t.Fatalf("filter.must = %v, want tenant and model conditions", filter["must"])
	}

	// The tenant condition must be present and carry the right value.
	var sawTenant bool
	for _, c := range must {
		cond := c.(map[string]any)
		if cond["key"] == "tenant" {
			sawTenant = true
			match := cond["match"].(map[string]any)
			if match["value"] != "tenant-a" {
				t.Errorf("tenant filter value = %v, want tenant-a", match["value"])
			}
		}
	}
	if !sawTenant {
		t.Error("search filter must constrain on tenant")
	}

	// The threshold is pushed server-side so weak matches never cross the wire.
	if fq.searchBody["score_threshold"] != 0.95 {
		t.Errorf("score_threshold = %v, want 0.95 pushed into Qdrant", fq.searchBody["score_threshold"])
	}
}

// TestSemantic_BelowThresholdIsNotAHit asserts the client-side threshold check
// holds even if the server returns a weak match anyway.
func TestSemantic_BelowThresholdIsNotAHit(t *testing.T) {
	fq := &fakeQdrant{
		result: []store.ScoredPoint{{
			ID:    "p1",
			Score: 0.80, // below the 0.95 threshold
			Payload: map[string]any{
				"completion": "a stale answer to a different question",
				"model":      "gpt-4o",
			},
		}},
	}
	tier := testSemanticTier(t, fq)

	r := req("gpt-4o", f(0.2), nil, user("hello"))
	entry, score, err := tier.Search(context.Background(), "t", r, []float32{0.1}, 0.95)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if entry != nil {
		t.Error("a match below the threshold must not be returned, even if the server sends it")
	}
	if score != 0.80 {
		t.Errorf("score = %v, want the observed 0.80 reported for metrics", score)
	}
}

// TestSemantic_AboveThresholdReturnsEntry asserts a strong match decodes fully.
func TestSemantic_AboveThresholdReturnsEntry(t *testing.T) {
	now := time.Now().Unix()
	fq := &fakeQdrant{
		result: []store.ScoredPoint{{
			ID:    "p1",
			Score: 0.97,
			Payload: map[string]any{
				"completion":        "Paris",
				"model":             "gpt-4o",
				"provider":          "openai",
				"temperature":       0.2,
				"prompt_tokens":     float64(10),
				"completion_tokens": float64(3),
				"created_at":        float64(now),
			},
		}},
	}
	tier := testSemanticTier(t, fq)

	r := req("gpt-4o", f(0.2), nil, user("capital of France?"))
	entry, score, err := tier.Search(context.Background(), "t", r, []float32{0.1}, 0.95)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if entry == nil {
		t.Fatal("a match above the threshold must be returned")
	}
	if entry.Completion != "Paris" {
		t.Errorf("completion = %q", entry.Completion)
	}
	if entry.Provider != "openai" {
		t.Errorf("provider = %q, want openai", entry.Provider)
	}
	// JSON numbers arrive as float64; they must survive conversion to int.
	if entry.CompletionTokens != 3 {
		t.Errorf("completion_tokens = %d, want 3", entry.CompletionTokens)
	}
	if score != 0.97 {
		t.Errorf("score = %v, want 0.97", score)
	}
}

// TestSemantic_NoMatch asserts an empty result is a clean miss, not an error.
func TestSemantic_NoMatch(t *testing.T) {
	fq := &fakeQdrant{result: nil}
	tier := testSemanticTier(t, fq)

	entry, _, err := tier.Search(context.Background(), "t", req("gpt-4o", f(0.2), nil, user("hi")), []float32{0.1}, 0.95)
	if err != nil {
		t.Fatalf("an empty result must not be an error: %v", err)
	}
	if entry != nil {
		t.Error("entry must be nil when nothing matches")
	}
}

// TestSemantic_UpsertPayload asserts the stored payload carries everything the
// compatibility guard later needs to decide whether the entry may be served.
func TestSemantic_UpsertPayload(t *testing.T) {
	fq := &fakeQdrant{}
	tier := testSemanticTier(t, fq)

	r := req("gpt-4o", f(0.2), nil, user("what is 2+2?"))
	entry := &Entry{
		Completion: "4", Model: "gpt-4o", Provider: "openai",
		Temp: 0.2, PromptTokens: 5, CompletionTokens: 1, CreatedAt: time.Now().Unix(),
	}
	if err := tier.Upsert(context.Background(), "tenant-x", r, entry, []float32{0.1, 0.2}); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	points := fq.upsertBody["points"].([]any)
	if len(points) != 1 {
		t.Fatalf("upserted %d points, want 1", len(points))
	}
	payload := points[0].(map[string]any)["payload"].(map[string]any)

	for _, key := range []string{"tenant", "model", "provider", "completion", "temperature", "completion_tokens", "created_at"} {
		if _, ok := payload[key]; !ok {
			t.Errorf("payload is missing %q, which the compatibility guard needs", key)
		}
	}
	if payload["tenant"] != "tenant-x" {
		t.Errorf("payload tenant = %v, want tenant-x", payload["tenant"])
	}
}

// TestPointID_StableAndScoped asserts IDs are deterministic per (tenant, model,
// prompt), so re-caching overwrites instead of accumulating duplicate points
// that would all match and slow every future search.
func TestPointID_StableAndScoped(t *testing.T) {
	a := pointID("t1", "gpt-4o", "hello")
	if a != pointID("t1", "gpt-4o", "hello") {
		t.Error("point ID must be stable for the same inputs")
	}
	if a == pointID("t2", "gpt-4o", "hello") {
		t.Error("point ID must differ per tenant")
	}
	if a == pointID("t1", "gpt-4o-mini", "hello") {
		t.Error("point ID must differ per model")
	}
	if a == pointID("t1", "gpt-4o", "goodbye") {
		t.Error("point ID must differ per prompt")
	}
}
