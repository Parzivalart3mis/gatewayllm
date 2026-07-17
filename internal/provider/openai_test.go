package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testOpenAI(t *testing.T, h http.HandlerFunc) Provider {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return NewOpenAI(Options{Name: "test", BaseURL: srv.URL, APIKey: "sk-test", Timeout: 5 * time.Second})
}

func testChatReq() *ChatRequest {
	return &ChatRequest{Model: "gpt-4o", Messages: []Message{{Role: RoleUser, Content: "hi"}}}
}

// TestOpenAI_Chat asserts the happy path and that auth is sent correctly.
func TestOpenAI_Chat(t *testing.T) {
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("Authorization = %q, want the bearer key", got)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		fmt.Fprint(w, `{"id":"cmpl-1","object":"chat.completion","model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	})

	resp, err := p.Chat(context.Background(), testChatReq())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Text() != "hello" {
		t.Errorf("text = %q", resp.Text())
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("total tokens = %d, want 7", resp.Usage.TotalTokens)
	}
}

// TestOpenAI_ResolvedModelIsSent asserts the router's resolved upstream model
// reaches the provider, not the client-facing alias.
func TestOpenAI_ResolvedModelIsSent(t *testing.T) {
	var gotModel string
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		decodeBody(t, r, &body)
		gotModel = body.Model
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"x"}}]}`)
	})

	req := testChatReq()
	req.Model = "my-alias"
	req.ResolvedModel = "gpt-4o-2024-08-06"
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotModel != "gpt-4o-2024-08-06" {
		t.Errorf("upstream model = %q, want the resolved model, not the alias", gotModel)
	}
}

// TestOpenAI_ErrorClassification asserts upstream failures map onto the kinds
// resilience uses to decide retry and failover. Getting these wrong means either
// hammering a dead provider or giving up on a recoverable one.
func TestOpenAI_ErrorClassification(t *testing.T) {
	cases := []struct {
		name         string
		status       int
		body         string
		wantKind     Kind
		wantRetry    bool
		wantFailover bool
	}{
		{
			name: "429 is a retryable rate limit", status: http.StatusTooManyRequests,
			body:     `{"error":{"message":"slow down","type":"rate_limit_error"}}`,
			wantKind: KindRateLimit, wantRetry: true, wantFailover: true,
		},
		{
			name: "503 is retryable and failoverable", status: http.StatusServiceUnavailable,
			body:     `{"error":{"message":"overloaded"}}`,
			wantKind: KindUnavailable, wantRetry: true, wantFailover: true,
		},
		{
			name:     "401 is an auth failure: not retryable, but failover may help",
			status:   http.StatusUnauthorized,
			body:     `{"error":{"message":"bad key"}}`,
			wantKind: KindAuth, wantRetry: false, wantFailover: true,
		},
		{
			name:     "400 invalid request is neither retried nor failed over",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"bad param","type":"invalid_request_error"}}`,
			wantKind: KindInvalidRequest, wantRetry: false, wantFailover: false,
		},
		{
			name:     "context_length_exceeded is not retried but may fit another provider",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"too long","code":"context_length_exceeded"}}`,
			wantKind: KindContextLength, wantRetry: false, wantFailover: true,
		},
		{
			name:     "content_filter is deterministic: no retry, no failover",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"refused","code":"content_filter"}}`,
			wantKind: KindContentFilter, wantRetry: false, wantFailover: false,
		},
		{
			name:     "insufficient_quota cannot be retried but another provider can serve",
			status:   http.StatusTooManyRequests,
			body:     `{"error":{"message":"billing","code":"insufficient_quota"}}`,
			wantKind: KindUnavailable, wantRetry: true, wantFailover: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			})

			_, err := p.Chat(context.Background(), testChatReq())
			pe := AsError(err)
			if pe == nil {
				t.Fatalf("error must be a *Error, got %T: %v", err, err)
			}
			if pe.Kind != tc.wantKind {
				t.Errorf("kind = %q, want %q", pe.Kind, tc.wantKind)
			}
			if pe.Retryable() != tc.wantRetry {
				t.Errorf("Retryable() = %v, want %v", pe.Retryable(), tc.wantRetry)
			}
			if pe.Failoverable() != tc.wantFailover {
				t.Errorf("Failoverable() = %v, want %v", pe.Failoverable(), tc.wantFailover)
			}
		})
	}
}

// TestOpenAI_RetryAfterIsParsed asserts a server-directed delay is honored
// rather than overridden by the gateway's own backoff.
func TestOpenAI_RetryAfterIsParsed(t *testing.T) {
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
	})

	_, err := p.Chat(context.Background(), testChatReq())
	pe := AsError(err)
	if pe == nil {
		t.Fatalf("want *Error, got %T", err)
	}
	if pe.RetryAfter != 7*time.Second {
		t.Errorf("RetryAfter = %v, want 7s", pe.RetryAfter)
	}
}

// TestOpenAI_ChatStream asserts SSE parsing assembles the deltas in order.
func TestOpenAI_ChatStream(t *testing.T) {
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []string{
			`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"lo"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	var text strings.Builder
	var usage *Usage
	err := p.ChatStream(context.Background(), testChatReq(), func(c *ChatChunk) error {
		for _, ch := range c.Choices {
			text.WriteString(ch.Delta.Content)
		}
		if c.Usage != nil {
			usage = c.Usage
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if text.String() != "Hello" {
		t.Errorf("assembled = %q, want Hello", text.String())
	}
	// Without stream_options.include_usage the final chunk carries no usage and
	// the meter would undercount every streamed request.
	if usage == nil || usage.TotalTokens != 3 {
		t.Error("the final chunk's usage must be surfaced")
	}
}

// TestOpenAI_StreamRequestsUsage asserts the gateway asks for usage on streams.
func TestOpenAI_StreamRequestsUsage(t *testing.T) {
	var body struct {
		Stream        bool `json:"stream"`
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		decodeBody(t, r, &body)
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	_ = p.ChatStream(context.Background(), testChatReq(), func(*ChatChunk) error { return nil })

	if !body.Stream {
		t.Error("stream must be set on a streaming request")
	}
	if body.StreamOptions == nil || !body.StreamOptions.IncludeUsage {
		t.Error("stream_options.include_usage must be requested or token counts are lost")
	}
}

// TestOpenAI_StreamIgnoresKeepalives asserts SSE comments and blank lines do not
// break parsing: providers send them to hold the connection open.
func TestOpenAI_StreamIgnoresKeepalives(t *testing.T) {
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, ": keepalive\n\n")
		fmt.Fprint(w, "\n")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	var text strings.Builder
	err := p.ChatStream(context.Background(), testChatReq(), func(c *ChatChunk) error {
		for _, ch := range c.Choices {
			text.WriteString(ch.Delta.Content)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream: %v", err)
	}
	if text.String() != "ok" {
		t.Errorf("text = %q, want ok", text.String())
	}
}

// TestOpenAI_StreamTruncated asserts a stream that ends without [DONE] is an
// error: the client must not be told a truncated answer is complete.
func TestOpenAI_StreamTruncated(t *testing.T) {
	p := testOpenAI(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"partial"}}]}`+"\n\n")
		// Connection ends here with no terminator.
	})

	err := p.ChatStream(context.Background(), testChatReq(), func(*ChatChunk) error { return nil })
	if err == nil {
		t.Fatal("a stream ending without [DONE] must be reported as an error")
	}
	if pe := AsError(err); pe == nil || pe.Kind != KindUnavailable {
		t.Errorf("kind = %v, want unavailable", err)
	}
}

// TestGroq_EmbedUnsupported asserts Groq reports embeddings as unsupported
// rather than surfacing a confusing upstream 404.
func TestGroq_EmbedUnsupported(t *testing.T) {
	p := NewGroq(Options{Name: "groq", APIKey: "k"})
	if _, err := p.Embed(context.Background(), &EmbeddingRequest{Input: Input{"x"}}); err != ErrUnsupported {
		t.Errorf("err = %v, want ErrUnsupported", err)
	}
}

// TestInputUnmarshal asserts the polymorphic embeddings input field decodes both
// documented forms.
func TestInputUnmarshal(t *testing.T) {
	var single Input
	if err := single.UnmarshalJSON([]byte(`"one"`)); err != nil {
		t.Fatalf("string input: %v", err)
	}
	if len(single) != 1 || single[0] != "one" {
		t.Errorf("single = %v, want [one]", single)
	}

	var many Input
	if err := many.UnmarshalJSON([]byte(`["a","b"]`)); err != nil {
		t.Fatalf("array input: %v", err)
	}
	if len(many) != 2 {
		t.Errorf("many = %v, want 2 items", many)
	}

	var bad Input
	if err := bad.UnmarshalJSON([]byte(`123`)); err == nil {
		t.Error("a numeric input must be rejected")
	}
}

func decodeBody(t *testing.T, r *http.Request, v any) {
	t.Helper()
	if err := decodeJSONBody(r, v); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
}

// decodeJSONBody reads a request body as JSON.
func decodeJSONBody(r *http.Request, v any) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}
