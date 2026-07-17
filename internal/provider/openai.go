package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// openAICompat implements Provider against any backend speaking the OpenAI wire
// protocol. OpenAI and Groq differ only in base URL and model list, so both are
// constructed from this type rather than duplicating the transport and SSE
// parsing. A backend with a distinct wire format (see gemini.go) gets its own.
type openAICompat struct {
	name    string
	baseURL string
	apiKey  string
	models  []string
	timeout time.Duration
	http    *http.Client
}

// Options configures an adapter constructor.
type Options struct {
	Name    string
	BaseURL string
	APIKey  string
	Models  []string
	Timeout time.Duration
}

// defaultTransport is shared so connections pool across providers instead of
// each adapter opening its own pool.
var defaultTransport = &http.Transport{
	MaxIdleConns:        200,
	MaxIdleConnsPerHost: 50,
	IdleConnTimeout:     90 * time.Second,
}

// newHTTPClient builds a client with no client-level Timeout: that deadline
// covers reading the whole body and would sever long streams mid-response.
// Per-call deadlines are applied to the context instead (see withTimeout).
func newHTTPClient() *http.Client {
	return &http.Client{Transport: defaultTransport}
}

// NewOpenAI builds an adapter for the OpenAI API.
func NewOpenAI(o Options) Provider {
	if o.BaseURL == "" {
		o.BaseURL = "https://api.openai.com/v1"
	}
	return &openAICompat{
		name:    orDefault(o.Name, "openai"),
		baseURL: strings.TrimRight(o.BaseURL, "/"),
		apiKey:  o.APIKey,
		models:  o.Models,
		timeout: o.Timeout,
		http:    newHTTPClient(),
	}
}

// withTimeout applies the provider's configured timeout to ctx. For streaming it
// is deliberately not applied: the timeout bounds how long a single call may
// take, but a stream legitimately stays open far longer, so streams are bounded
// by the client's context and the server write timeout instead.
func (p *openAICompat) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if p.timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, p.timeout)
}

func (p *openAICompat) Name() string     { return p.name }
func (p *openAICompat) Models() []string { return p.models }

// upstreamModel returns the model the adapter should send: the router's resolved
// model when set, otherwise whatever the client asked for.
func upstreamModel(req *ChatRequest) string {
	if req.ResolvedModel != "" {
		return req.ResolvedModel
	}
	return req.Model
}

// wireRequest is the body sent upstream. It is built explicitly rather than
// re-marshalling ChatRequest so gateway-internal fields never leak upstream.
type wireRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
	User        string    `json:"user,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	// StreamOptions asks OpenAI to append a usage-bearing final chunk, without
	// which streamed requests report no token counts and the meter undercounts.
	StreamOptions *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

func (p *openAICompat) buildBody(req *ChatRequest, stream bool) *wireRequest {
	w := &wireRequest{
		Model:       upstreamModel(req),
		Messages:    req.Messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		User:        req.User,
		Stream:      stream,
	}
	if stream {
		w.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return w
}

func (p *openAICompat) newRequest(ctx context.Context, path string, body any, stream bool) (*http.Request, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, Wrap(KindInternal, p.name, err, "marshal request")
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(buf))
	if err != nil {
		return nil, Wrap(KindInternal, p.name, err, "build request")
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+p.apiKey)
	if stream {
		r.Header.Set("Accept", "text/event-stream")
	}
	return r, nil
}

func (p *openAICompat) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	ctx, cancel := p.withTimeout(ctx)
	defer cancel()

	httpReq, err := p.newRequest(ctx, "/chat/completions", p.buildBody(req, false), false)
	if err != nil {
		return nil, err
	}
	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, p.errorFromResponse(resp)
	}
	var out ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, Wrap(KindUnavailable, p.name, err, "decode response")
	}
	if len(out.Choices) == 0 {
		return nil, Errorf(KindUnavailable, p.name, "upstream returned no choices")
	}
	return &out, nil
}

func (p *openAICompat) ChatStream(ctx context.Context, req *ChatRequest, onChunk func(*ChatChunk) error) error {
	httpReq, err := p.newRequest(ctx, "/chat/completions", p.buildBody(req, true), true)
	if err != nil {
		return err
	}
	resp, err := p.http.Do(httpReq)
	if err != nil {
		return p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return p.errorFromResponse(resp)
	}
	return p.parseSSE(ctx, resp.Body, onChunk)
}

// parseSSE reads an OpenAI-style event stream. Chunks arrive as `data: {json}`
// lines terminated by `data: [DONE]`.
func (p *openAICompat) parseSSE(ctx context.Context, body io.Reader, onChunk func(*ChatChunk) error) error {
	sc := bufio.NewScanner(body)
	// Chunks are small, but a single SSE line carrying a large tool-call payload
	// can exceed the 64KB default and would otherwise abort the stream.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, ":") { // blank or SSE comment/keepalive
			continue
		}
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue // ignore `event:`/`id:` fields; only data carries chunks
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			return nil
		}
		var chunk ChatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			// A malformed chunk mid-stream means the rest is untrustworthy.
			return Wrap(KindUnavailable, p.name, err, "decode stream chunk")
		}
		if err := onChunk(&chunk); err != nil {
			return err // caller aborted (client disconnect, write failure)
		}
	}
	if err := sc.Err(); err != nil {
		return p.transportError(ctx, err)
	}
	// Stream ended without [DONE]: the upstream connection dropped early.
	return Errorf(KindUnavailable, p.name, "stream ended without terminator")
}

func (p *openAICompat) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	ctx, cancel := p.withTimeout(ctx)
	defer cancel()

	httpReq, err := p.newRequest(ctx, "/embeddings", req, false)
	if err != nil {
		return nil, err
	}
	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, p.errorFromResponse(resp)
	}
	var out EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, Wrap(KindUnavailable, p.name, err, "decode embeddings response")
	}
	return &out, nil
}

// apiError is the OpenAI error envelope.
type apiError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// errorFromResponse classifies a non-200. It starts from the HTTP status and
// refines using the error code, which distinguishes causes that share a status:
// a 400 may be a malformed body, an oversized context, or a safety refusal.
func (p *openAICompat) errorFromResponse(resp *http.Response) *Error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024))
	kind := ClassifyStatus(resp.StatusCode)
	msg := strings.TrimSpace(string(body))

	var ae apiError
	if err := json.Unmarshal(body, &ae); err == nil && ae.Error.Message != "" {
		msg = ae.Error.Message
		switch ae.Error.Code {
		case "context_length_exceeded":
			kind = KindContextLength
		case "content_filter":
			kind = KindContentFilter
		case "insufficient_quota":
			// Billing exhaustion, not throughput throttling: retrying the same
			// provider cannot help, but failing over to another can.
			kind = KindUnavailable
		}
		if ae.Error.Type == "invalid_request_error" && kind == KindInternal {
			kind = KindInvalidRequest
		}
	}
	if msg == "" {
		msg = http.StatusText(resp.StatusCode)
	}
	return &Error{
		Kind:       kind,
		Provider:   p.name,
		Status:     resp.StatusCode,
		Message:    truncate(msg, 512),
		RetryAfter: ParseRetryAfter(resp.Header),
	}
}

// transportError classifies a failure that produced no HTTP response.
func (p *openAICompat) transportError(ctx context.Context, err error) *Error {
	switch {
	case errors.Is(err, context.Canceled) && ctx.Err() != nil:
		// The caller went away (client disconnected); not an upstream fault, so
		// it must not count against the provider's circuit breaker.
		return Wrap(KindInternal, p.name, err, "request canceled")
	case errors.Is(err, context.DeadlineExceeded):
		return Wrap(KindTimeout, p.name, err, "request timed out")
	default:
		return Wrap(KindUnavailable, p.name, err, "transport error")
	}
}

// drainClose consumes any remaining body before closing so the keep-alive
// connection returns to the pool instead of being discarded.
func drainClose(rc io.ReadCloser) {
	_, _ = io.Copy(io.Discard, io.LimitReader(rc, 64*1024))
	_ = rc.Close()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
