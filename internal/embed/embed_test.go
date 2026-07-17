package embed

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/yash/gatewayllm/internal/config"
)

// TestSidecar_Embed asserts the sidecar contract: vectors come back in input
// order, one per text.
func TestSidecar_Embed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embed" {
			t.Errorf("path = %q, want /embed", r.URL.Path)
		}
		var req struct {
			Texts []string `json:"texts"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)

		vecs := make([][]float32, len(req.Texts))
		for i := range req.Texts {
			vecs[i] = []float32{float32(i), 0.5}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"vectors": vecs, "model": "test", "dims": 2})
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Model: "test", Timeout: time.Second, Dimensions: 2})
	vecs, err := e.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	if vecs[2][0] != 2 {
		t.Errorf("vectors must come back in input order, got %v", vecs)
	}
}

// TestSidecar_VectorCountMismatch guards a silent corruption path: a short
// vector list would misalign vectors with prompts and poison the cache with
// entries keyed on the wrong text.
func TestSidecar_VectorCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Two texts requested, one vector returned.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"vectors": [][]float32{{0.1, 0.2}}, "model": "test", "dims": 2,
		})
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Timeout: time.Second, Dimensions: 2})
	if _, err := e.Embed(context.Background(), []string{"a", "b"}); err == nil {
		t.Fatal("a vector/text count mismatch must be an error, not silently misaligned")
	}
}

// TestSidecar_Unavailable asserts a down sidecar is reported as unavailable, so
// the cache can degrade to a miss rather than failing the request.
func TestSidecar_Unavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Timeout: time.Second, Dimensions: 2})
	_, err := e.Embed(context.Background(), []string{"a"})
	if !errors.Is(err, ErrEmbedderUnavailable) {
		t.Errorf("err = %v, want ErrEmbedderUnavailable", err)
	}
}

// TestSidecar_Info asserts dimensions are discoverable, which is what lets the
// gateway catch a model/collection mismatch at startup.
func TestSidecar_Info(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "all-MiniLM-L6-v2", "dims": 384})
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Timeout: time.Second})
	if got := e.Dimensions(); got != 384 {
		t.Errorf("Dimensions() = %d, want 384 probed from /info", got)
	}
}

// TestSidecar_ConfiguredDimsSkipProbe asserts an explicit size avoids the probe.
func TestSidecar_ConfiguredDimsSkipProbe(t *testing.T) {
	probed := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probed = true
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "x", "dims": 999})
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Dimensions: 384})
	if got := e.Dimensions(); got != 384 {
		t.Errorf("Dimensions() = %d, want the configured 384", got)
	}
	if probed {
		t.Error("a configured dimension must not trigger a probe")
	}
}

// TestAPI_Embed asserts the hosted embedder speaks the OpenAI shape.
func TestAPI_Embed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("Authorization = %q", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 0, "embedding": []float32{0.1, 0.2}},
			},
			"model": "text-embedding-3-small",
		})
	}))
	defer srv.Close()

	e := NewAPI(APIOptions{BaseURL: srv.URL, Model: "text-embedding-3-small", APIKey: "sk-test", Timeout: time.Second})
	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 || vecs[0][1] != 0.2 {
		t.Errorf("vectors = %v", vecs)
	}
}

// TestAPI_ReordersByIndex asserts vectors are placed by their index field.
// The API documents index ordering but does not guarantee array order; trusting
// position would pair a vector with the wrong prompt.
func TestAPI_ReordersByIndex(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Deliberately out of order.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"index": 1, "embedding": []float32{9, 9}},
				{"index": 0, "embedding": []float32{1, 1}},
			},
		})
	}))
	defer srv.Close()

	e := NewAPI(APIOptions{BaseURL: srv.URL, Model: "text-embedding-3-small", APIKey: "k", Timeout: time.Second})
	vecs, err := e.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if vecs[0][0] != 1 || vecs[1][0] != 9 {
		t.Errorf("vectors must be placed by index, got %v", vecs)
	}
}

// TestAPI_KnownDimensions asserts common models get their size without config.
func TestAPI_KnownDimensions(t *testing.T) {
	e := NewAPI(APIOptions{Model: "text-embedding-3-small", APIKey: "k"})
	if got := e.Dimensions(); got != 1536 {
		t.Errorf("Dimensions() = %d, want 1536", got)
	}
}

// TestEmbedOne asserts the single-text convenience wrapper.
func TestEmbedOne(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"vectors": [][]float32{{0.5, 0.5}}, "model": "t", "dims": 2,
		})
	}))
	defer srv.Close()

	e := NewSidecar(SidecarOptions{BaseURL: srv.URL, Timeout: time.Second, Dimensions: 2})
	vec, err := EmbedOne(context.Background(), e, "hello")
	if err != nil {
		t.Fatalf("EmbedOne: %v", err)
	}
	if len(vec) != 2 {
		t.Errorf("vector = %v, want 2 dimensions", vec)
	}
}

// TestBuild asserts the factory honors the config switch, which is what makes
// the embedder swappable without a code change.
func TestBuild(t *testing.T) {
	sidecar, err := Build(cfgEmbed("sidecar", "http://localhost:8000", ""), 384)
	if err != nil {
		t.Fatalf("Build sidecar: %v", err)
	}
	if sidecar.Dimensions() != 384 {
		t.Errorf("sidecar dims = %d, want 384", sidecar.Dimensions())
	}

	api, err := Build(cfgEmbed("api", "", "sk-key"), 0)
	if err != nil {
		t.Fatalf("Build api: %v", err)
	}
	if api.Dimensions() != 1536 {
		t.Errorf("api dims = %d, want the known 1536", api.Dimensions())
	}

	if _, err := Build(cfgEmbed("nonsense", "", ""), 384); err == nil {
		t.Error("an unknown embedder kind must fail at construction")
	}
}

// cfgEmbed builds an embed config for the factory tests.
func cfgEmbed(kind, baseURL, apiKey string) config.Embed {
	model := "sentence-transformers/all-MiniLM-L6-v2"
	if kind == "api" {
		model = "text-embedding-3-small"
	}
	return config.Embed{Kind: kind, BaseURL: baseURL, Model: model, APIKey: apiKey, Timeout: time.Second}
}
