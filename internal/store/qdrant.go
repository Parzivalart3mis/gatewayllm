package store

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

// Qdrant is a minimal REST client for the vector operations the semantic cache
// needs: ensure a collection, upsert a point, search by vector.
//
// It speaks REST rather than gRPC deliberately: the official gRPC client pulls a
// large dependency tree to serve three call sites, and at the semantic tier's
// request rate the difference in wire efficiency is not measurable.
type Qdrant struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewQdrant builds a Qdrant client.
func NewQdrant(baseURL, apiKey string, timeout time.Duration) *Qdrant {
	if timeout <= 0 {
		timeout = 3 * time.Second
	}
	return &Qdrant{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        50,
				MaxIdleConnsPerHost: 50,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// Distance is a vector similarity metric.
type Distance string

// Cosine compares direction while ignoring magnitude, which is what "these two
// prompts mean the same thing" requires.
const Cosine Distance = "Cosine"

// Point is one cached vector plus its payload.
type Point struct {
	ID      string         `json:"id"`
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload,omitempty"`
}

// ScoredPoint is a search result.
type ScoredPoint struct {
	ID      string         `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}

func (q *Qdrant) do(ctx context.Context, method, path string, body, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("qdrant: marshal: %w", err)
		}
		rdr = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, q.baseURL+path, rdr)
	if err != nil {
		return fmt.Errorf("qdrant: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := q.http.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant: %s %s: %w", method, path, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("qdrant: %s %s: status %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return fmt.Errorf("qdrant: decode %s: %w", path, err)
		}
	}
	return nil
}

type collectionInfoResponse struct {
	Result struct {
		Config struct {
			Params struct {
				Vectors struct {
					Size     int    `json:"size"`
					Distance string `json:"distance"`
				} `json:"vectors"`
			} `json:"params"`
		} `json:"config"`
	} `json:"result"`
}

// EnsureCollection creates the collection if absent, and verifies the vector
// size if present.
//
// The size check matters: a collection built for a 384-dim model will accept
// neither reads nor writes from a 1536-dim one, and discovering that through
// per-request errors is far worse than refusing to start.
func (q *Qdrant) EnsureCollection(ctx context.Context, name string, size int, dist Distance) error {
	var info collectionInfoResponse
	err := q.do(ctx, http.MethodGet, "/collections/"+name, nil, &info)
	if err == nil {
		got := info.Result.Config.Params.Vectors.Size
		if got != 0 && got != size {
			return fmt.Errorf("qdrant: collection %q has vector size %d but the embedder produces %d: "+
				"recreate the collection or switch the embedding model", name, got, size)
		}
		return nil
	}
	if !strings.Contains(err.Error(), "status 404") {
		return err
	}

	body := map[string]any{
		"vectors": map[string]any{"size": size, "distance": string(dist)},
	}
	// A concurrent replica may have created it between the GET and the PUT; that
	// is a benign race, not a startup failure.
	if err := q.do(ctx, http.MethodPut, "/collections/"+name, body, nil); err != nil {
		if strings.Contains(err.Error(), "already exists") {
			return nil
		}
		return err
	}
	return nil
}

// Upsert writes points into a collection.
func (q *Qdrant) Upsert(ctx context.Context, collection string, points []Point) error {
	if len(points) == 0 {
		return nil
	}
	// wait=false: the cache write is a side effect of a request the client is
	// waiting on, so it must not block on the index being durable.
	return q.do(ctx, http.MethodPut, "/collections/"+collection+"/points?wait=false",
		map[string]any{"points": points}, nil)
}

type searchResponse struct {
	Result []ScoredPoint `json:"result"`
}

// SearchRequest is a vector similarity query.
type SearchRequest struct {
	Vector []float32 `json:"vector"`
	Limit  int       `json:"limit"`
	// ScoreThreshold lets Qdrant discard weak matches server-side.
	ScoreThreshold float64 `json:"score_threshold,omitempty"`
	WithPayload    bool    `json:"with_payload"`
	// Filter narrows candidates before scoring.
	Filter map[string]any `json:"filter,omitempty"`
}

// Search runs a vector query and returns scored matches.
func (q *Qdrant) Search(ctx context.Context, collection string, req SearchRequest) ([]ScoredPoint, error) {
	req.WithPayload = true
	var out searchResponse
	if err := q.do(ctx, http.MethodPost, "/collections/"+collection+"/points/search", req, &out); err != nil {
		return nil, err
	}
	return out.Result, nil
}

// Delete removes points by ID.
func (q *Qdrant) Delete(ctx context.Context, collection string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	return q.do(ctx, http.MethodPost, "/collections/"+collection+"/points/delete?wait=false",
		map[string]any{"points": ids}, nil)
}

// Ping reports whether Qdrant is reachable.
func (q *Qdrant) Ping(ctx context.Context) error {
	return q.do(ctx, http.MethodGet, "/healthz", nil, nil)
}
