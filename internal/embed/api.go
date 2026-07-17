package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// API embeds via a hosted OpenAI-compatible embeddings endpoint. It trades the
// sidecar's zero marginal cost for zero operational overhead — no model to ship,
// no memory to budget — at the price of a network hop and a per-token charge on
// the cache lookup path.
type API struct {
	baseURL string
	model   string
	apiKey  string
	dims    int
	http    *http.Client
}

// APIOptions configures the hosted embedder.
type APIOptions struct {
	BaseURL string
	Model   string
	APIKey  string
	Timeout time.Duration
	// Dimensions is the expected vector size; it must match the collection.
	Dimensions int
}

// knownDims records output sizes for common embedding models, so a config that
// omits dimensions still validates against the Qdrant collection at startup.
var knownDims = map[string]int{
	"text-embedding-3-small": 1536,
	"text-embedding-3-large": 3072,
	"text-embedding-ada-002": 1536,
}

// NewAPI builds a hosted embedder.
func NewAPI(o APIOptions) *API {
	if o.BaseURL == "" {
		o.BaseURL = "https://api.openai.com/v1"
	}
	if o.Timeout <= 0 {
		o.Timeout = 10 * time.Second
	}
	dims := o.Dimensions
	if dims == 0 {
		dims = knownDims[o.Model]
	}
	return &API{
		baseURL: strings.TrimRight(o.BaseURL, "/"),
		model:   o.Model,
		apiKey:  o.APIKey,
		dims:    dims,
		http:    &http.Client{Timeout: o.Timeout},
	}
}

func (a *API) Name() string    { return "api:" + a.model }
func (a *API) Dimensions() int { return a.dims }

type apiRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
	// Dimensions asks the API to truncate output, supported by the v3 models.
	Dimensions *int `json:"dimensions,omitempty"`
}

type apiResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
}

func (a *API) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	reqBody := apiRequest{Model: a.model, Input: texts}
	if a.dims > 0 && strings.HasPrefix(a.model, "text-embedding-3") {
		reqBody.Dimensions = &a.dims
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("embed api: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embed api: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.apiKey)

	resp, err := a.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrEmbedderUnavailable, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("%w: status %d: %s", ErrEmbedderUnavailable, resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var out apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed api: decode: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embed api: got %d vectors for %d texts", len(out.Data), len(texts))
	}

	// The API documents index ordering but does not guarantee array order;
	// placing by index rather than trusting position avoids pairing a vector
	// with the wrong prompt.
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("embed api: vector index %d out of range", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	for i, v := range vecs {
		if v == nil {
			return nil, fmt.Errorf("embed api: missing vector for input %d", i)
		}
	}
	return vecs, nil
}
