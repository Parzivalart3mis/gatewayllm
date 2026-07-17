package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// Provider is the load-bearing abstraction: one implementation per LLM backend.
// Implementations must be safe for concurrent use and must translate upstream
// failures into *Error so resilience can classify them.
type Provider interface {
	// Name is the stable identifier used in config, metrics, and the usage log.
	Name() string

	// Chat performs a non-streaming completion.
	Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error)

	// ChatStream performs a streaming completion, invoking onChunk for each
	// chunk in order. Implementations must not call onChunk after returning.
	// An error returned by onChunk aborts the stream and is returned as-is.
	ChatStream(ctx context.Context, req *ChatRequest, onChunk func(*ChatChunk) error) error

	// Embed produces embedding vectors. Providers without an embeddings API
	// return ErrUnsupported.
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)

	// Models lists the upstream model IDs this provider can serve. The router
	// uses it to validate config at startup.
	Models() []string
}

// ErrUnsupported reports that a provider does not implement a capability.
var ErrUnsupported = errors.New("provider: capability not supported")

// Kind classifies a provider failure. Resilience uses it to decide whether to
// retry the same provider, fail over to another, or surface the error verbatim.
type Kind string

const (
	// KindInvalidRequest is a client error: the request is malformed or violates
	// the upstream contract. Never retried, never failed over — every provider
	// would reject it identically.
	KindInvalidRequest Kind = "invalid_request"
	// KindAuth is a bad or missing upstream credential. Not retried; failover to
	// a provider with working credentials is worthwhile.
	KindAuth Kind = "auth"
	// KindRateLimit is upstream throttling. Retried with backoff, honoring
	// Retry-After, and eligible for failover.
	KindRateLimit Kind = "rate_limit"
	// KindTimeout is a deadline or context expiry. Retried and failed over.
	KindTimeout Kind = "timeout"
	// KindUnavailable is a 5xx, connection failure, or open circuit. Retried and
	// failed over.
	KindUnavailable Kind = "unavailable"
	// KindContentFilter is an upstream safety refusal. Deterministic for a given
	// prompt, so never retried; failover is pointless and is not attempted.
	KindContentFilter Kind = "content_filter"
	// KindContextLength means the prompt exceeds the model's window. Retrying is
	// futile, but a provider with a larger window may succeed, so failover runs.
	KindContextLength Kind = "context_length"
	// KindInternal is an unclassified gateway-side failure.
	KindInternal Kind = "internal"
)

// Error is a provider failure carrying enough classification for resilience to
// act on and enough detail to render an OpenAI-compatible error body.
type Error struct {
	Kind     Kind
	Provider string
	// Status is the upstream HTTP status, or 0 if the failure was not an HTTP
	// response (for example, a dial error).
	Status int
	// Message is the human-readable cause, surfaced to the client.
	Message string
	// RetryAfter is a server-directed delay parsed from the Retry-After header.
	RetryAfter time.Duration
	// Err is the wrapped underlying cause, if any.
	Err error
}

func (e *Error) Error() string {
	if e.Provider == "" {
		return fmt.Sprintf("%s: %s", e.Kind, e.Message)
	}
	return fmt.Sprintf("%s: %s: %s", e.Provider, e.Kind, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// Retryable reports whether retrying the same provider could plausibly succeed.
func (e *Error) Retryable() bool {
	switch e.Kind {
	case KindRateLimit, KindTimeout, KindUnavailable:
		return true
	default:
		return false
	}
}

// Failoverable reports whether a different provider could plausibly succeed.
// Invalid requests and content filters are deterministic across providers, so
// failing over only burns latency and quota.
func (e *Error) Failoverable() bool {
	switch e.Kind {
	case KindInvalidRequest, KindContentFilter:
		return false
	default:
		return true
	}
}

// HTTPStatus maps the failure onto the status the gateway returns to its client.
func (e *Error) HTTPStatus() int {
	switch e.Kind {
	case KindInvalidRequest, KindContextLength, KindContentFilter:
		return http.StatusBadRequest
	case KindAuth:
		return http.StatusBadGateway // upstream credential problem, not the caller's
	case KindRateLimit:
		return http.StatusTooManyRequests
	case KindTimeout:
		return http.StatusGatewayTimeout
	case KindUnavailable:
		return http.StatusBadGateway
	default:
		return http.StatusInternalServerError
	}
}

// AsError extracts a *Error from err, or nil if err is not one.
func AsError(err error) *Error {
	var pe *Error
	if errors.As(err, &pe) {
		return pe
	}
	return nil
}

// Errorf builds a *Error.
func Errorf(kind Kind, prov string, format string, args ...any) *Error {
	return &Error{Kind: kind, Provider: prov, Message: fmt.Sprintf(format, args...)}
}

// Wrap builds a *Error around an underlying cause.
func Wrap(kind Kind, prov string, err error, format string, args ...any) *Error {
	return &Error{Kind: kind, Provider: prov, Message: fmt.Sprintf(format, args...), Err: err}
}

// ClassifyStatus maps an upstream HTTP status onto a Kind. Adapters use it as a
// baseline and override when the body carries a more specific cause.
func ClassifyStatus(status int) Kind {
	switch {
	case status == http.StatusUnauthorized, status == http.StatusForbidden:
		return KindAuth
	case status == http.StatusRequestTimeout, status == http.StatusGatewayTimeout:
		return KindTimeout
	case status == http.StatusTooManyRequests:
		return KindRateLimit
	case status >= 500:
		return KindUnavailable
	case status >= 400:
		return KindInvalidRequest
	default:
		return KindInternal
	}
}

// ParseRetryAfter reads a Retry-After header in either delay-seconds or HTTP-date
// form. It returns 0 when the header is absent or unparseable.
func ParseRetryAfter(h http.Header) time.Duration {
	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}
	if secs, err := time.ParseDuration(v + "s"); err == nil && secs > 0 {
		return secs
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}
