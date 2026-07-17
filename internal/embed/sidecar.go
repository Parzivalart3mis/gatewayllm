package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Sidecar embeds via a local Python service running a sentence-transformers
// model. Keeping the model out of the Go binary is the one deliberate exception
// to the single-binary rule: the mature embedding models ship as Python, and
// reimplementing inference in Go to preserve purity would be a bad trade.
//
// Because it is local, there is no per-token cost and no data leaves the host,
// which is what makes the semantic cache viable on a cheap VPS.
type Sidecar struct {
	baseURL string
	model   string
	timeout time.Duration
	http    *http.Client

	// dims is discovered from the sidecar on first use rather than configured,
	// so a model swap cannot silently disagree with the collection's vector size.
	dimsOnce sync.Once
	dims     int
	dimsErr  error
	// configuredDims, when positive, is used instead of probing.
	configuredDims int
}

// SidecarOptions configures the sidecar embedder.
type SidecarOptions struct {
	BaseURL string
	Model   string
	Timeout time.Duration
	// Dimensions skips the startup probe when known.
	Dimensions int
}

// NewSidecar builds a sidecar embedder.
func NewSidecar(o SidecarOptions) *Sidecar {
	if o.Timeout <= 0 {
		o.Timeout = 10 * time.Second
	}
	return &Sidecar{
		baseURL:        strings.TrimRight(o.BaseURL, "/"),
		model:          o.Model,
		timeout:        o.Timeout,
		configuredDims: o.Dimensions,
		http: &http.Client{
			Timeout: o.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        20,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

func (s *Sidecar) Name() string { return "sidecar:" + s.model }

type sidecarRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

type sidecarResponse struct {
	Vectors [][]float32 `json:"vectors"`
	Model   string      `json:"model"`
	Dims    int         `json:"dims"`
}

type sidecarInfo struct {
	Model string `json:"model"`
	Dims  int    `json:"dims"`
}

func (s *Sidecar) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(sidecarRequest{Texts: texts, Model: s.model})
	if err != nil {
		return nil, fmt.Errorf("embed sidecar: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embed sidecar: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.http.Do(req)
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

	var out sidecarResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("embed sidecar: decode: %w", err)
	}
	// A short vector list would silently misalign vectors with their prompts and
	// poison the cache with wrong-prompt entries.
	if len(out.Vectors) != len(texts) {
		return nil, fmt.Errorf("embed sidecar: got %d vectors for %d texts", len(out.Vectors), len(texts))
	}
	return out.Vectors, nil
}

// Dimensions reports the vector size, probing the sidecar once if not configured.
func (s *Sidecar) Dimensions() int {
	if s.configuredDims > 0 {
		return s.configuredDims
	}
	s.dimsOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
		defer cancel()
		info, err := s.info(ctx)
		if err != nil {
			s.dimsErr = err
			return
		}
		s.dims = info.Dims
	})
	return s.dims
}

func (s *Sidecar) info(ctx context.Context) (*sidecarInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.baseURL+"/info", nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrEmbedderUnavailable, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: info status %d", ErrEmbedderUnavailable, resp.StatusCode)
	}
	var out sidecarInfo
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Ping checks the sidecar is reachable and reports its model, so a broken
// embedder surfaces at startup rather than as a stream of cache misses.
func (s *Sidecar) Ping(ctx context.Context) error {
	_, err := s.info(ctx)
	return err
}
