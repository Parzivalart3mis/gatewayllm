package api

import (
	"context"
	"errors"
	"net/http"
	"runtime/debug"
	"time"

	"github.com/yash/gatewayllm/internal/provider"
)

// ctxKey is the private key type for request-scoped values.
type ctxKey int

const (
	ctxKeyTenant ctxKey = iota
	ctxKeyRequestID
	ctxKeyCacheStatus
)

// TenantFrom returns the authenticated tenant, if any.
func TenantFrom(ctx context.Context) *Tenant {
	t, _ := ctx.Value(ctxKeyTenant).(*Tenant)
	return t
}

// RequestIDFrom returns the request's correlation ID.
func RequestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(ctxKeyRequestID).(string)
	return id
}

// withRequestID assigns each request a correlation ID, honoring an inbound
// X-Request-Id so a trace can be followed across an upstream proxy.
func (s *Server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = provider.NewID()
		}
		w.Header().Set("X-Request-Id", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKeyRequestID, id)))
	})
}

// statusRecorder captures the response status and byte count for logging.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written int64
	// wroteHeader guards against a double WriteHeader panic when a handler
	// errors after streaming has begun.
	wroteHeader bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if r.wroteHeader {
		return
	}
	r.wroteHeader = true
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

// Flush forwards to the underlying writer. Without this, wrapping the writer
// would break streaming: chunks would buffer instead of reaching the client.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// withAccessLog logs one line per request after it completes.
func (s *Server) withAccessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)

		dur := time.Since(start)
		cacheStatus := ""
		if cs, ok := r.Context().Value(ctxKeyCacheStatus).(*string); ok {
			cacheStatus = *cs
		}

		attrs := []any{
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"bytes", rec.written,
			"duration_ms", dur.Milliseconds(),
			"request_id", RequestIDFrom(r.Context()),
		}
		if t := TenantFrom(r.Context()); t != nil {
			attrs = append(attrs, "tenant", t.ID)
		}
		if cacheStatus != "" {
			attrs = append(attrs, "cache", cacheStatus)
		}
		s.log.Info("request", attrs...)

		if s.deps.Metrics != nil {
			// Label on the route pattern, never the raw path: a per-path label
			// would be unbounded cardinality if any route ever carries an ID.
			s.deps.Metrics.RecordRequest(routePattern(r), rec.status, cacheStatus, dur)
		}
	})
}

// withRecovery converts a handler panic into a 500 instead of killing the
// process and every in-flight request with it.
func (s *Server) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				// A client disconnecting mid-write surfaces as this sentinel
				// panic; it is normal, not a bug, so it is not logged as one.
				if v == http.ErrAbortHandler {
					panic(v)
				}
				s.log.Error("panic recovered",
					"err", v,
					"path", r.URL.Path,
					"request_id", RequestIDFrom(r.Context()),
					"stack", string(debug.Stack()),
				)
				writeError(w, http.StatusInternalServerError, "api_error", "internal server error", "")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// withAuth resolves the bearer token into a tenant.
func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := bearerToken(r)
		if token == "" {
			writeError(w, http.StatusUnauthorized, "authentication_error",
				"missing API key: pass it as 'Authorization: Bearer <key>'", "missing_api_key")
			return
		}
		tenant, err := s.deps.Auth.Authenticate(r.Context(), token)
		if err != nil {
			if errors.Is(err, ErrUnauthorized) {
				writeError(w, http.StatusUnauthorized, "authentication_error",
					"invalid API key", "invalid_api_key")
				return
			}
			// The auth store is unreachable. This is the gateway's fault, not
			// the caller's, so it must not read as a rejected credential.
			s.log.Error("authentication backend failed", "err", err,
				"request_id", RequestIDFrom(r.Context()))
			writeError(w, http.StatusServiceUnavailable, "api_error",
				"authentication temporarily unavailable", "")
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKeyTenant, tenant)))
	})
}

// withRateLimit enforces the tenant's token bucket.
func (s *Server) withRateLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.deps.Limit == nil {
			next.ServeHTTP(w, r)
			return
		}
		t := TenantFrom(r.Context())
		if t == nil {
			next.ServeHTTP(w, r)
			return
		}

		res, err := s.deps.Limit.Allow(r.Context(), t)
		if err != nil {
			// Fail open: the limiter's own store being down is not a reason to
			// reject traffic the gateway can otherwise serve.
			s.log.Warn("rate limiter unavailable, allowing request", "err", err, "tenant", t.ID)
			next.ServeHTTP(w, r)
			return
		}

		res.writeHeaders(w)
		if !res.Allowed {
			writeError(w, http.StatusTooManyRequests, "rate_limit_error",
				"rate limit exceeded, retry later", "rate_limit_exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// routePattern returns the matched route pattern for metric labels. Falling back
// to a constant rather than the raw path keeps an unmatched request (a 404 probe
// from a scanner) from minting a new label series per URL.
func routePattern(r *http.Request) string {
	if p := r.Pattern; p != "" {
		return p
	}
	return "unmatched"
}
