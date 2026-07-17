// Package api is the HTTP surface: an OpenAI-compatible router plus auth and
// rate-limit middleware. It owns request/response translation and delegates all
// decisions to the cache, router, resilience, and meter packages.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/yash/gatewayllm/internal/cache"
	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/meter"
	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/resilience"
	"github.com/yash/gatewayllm/internal/router"
)

// Deps are the collaborators the server needs. Cache and Meter are optional:
// a nil value disables that concern, which is what keeps the build order's
// early stages runnable before those layers exist.
type Deps struct {
	Config  *config.Config
	Router  *router.Router
	Exec    *resilience.Executor
	Auth    Authenticator
	Limit   Limiter
	Cache   *cache.Cache
	Meter   *meter.Meter
	Metrics Metrics
	// MetricsHandler, when set, is mounted at Obs.MetricsPath on the main API
	// server. Used on single-port hosts (Cloud Run) where a dedicated metrics
	// listener is unreachable; left nil when metrics run on their own port.
	MetricsHandler http.Handler
	Logger         *slog.Logger
}

// Server serves the OpenAI-compatible API.
type Server struct {
	deps   Deps
	log    *slog.Logger
	mux    *http.ServeMux
	prices *meter.Pricing
}

// NewServer builds the server and registers routes.
func NewServer(d Deps) *Server {
	if d.Logger == nil {
		d.Logger = slog.Default()
	}
	s := &Server{
		deps: d,
		log:  d.Logger,
		mux:  http.NewServeMux(),
		// Built once: the table is read on every request and never mutated, so
		// building it lazily would only add a data race.
		prices: meter.NewPricing(d.Config.Pricing),
	}
	s.routes()
	return s
}

func (s *Server) routes() {
	// Authenticated, rate-limited API surface.
	s.mux.Handle("POST /v1/chat/completions", s.protected(http.HandlerFunc(s.handleChatCompletions)))
	s.mux.Handle("POST /v1/embeddings", s.protected(http.HandlerFunc(s.handleEmbeddings)))
	s.mux.Handle("GET /v1/models", s.protected(http.HandlerFunc(s.handleModels)))

	// Operational surface: unauthenticated by design so orchestrators and
	// scrapers can reach it without holding a tenant credential.
	s.mux.HandleFunc("GET /healthz", s.handleHealth)
	s.mux.HandleFunc("GET /readyz", s.handleReady)

	// Inline metrics for single-port hosts. Unauthenticated like the other
	// operational routes: a Cloud Run service is expected to gate scrape access
	// at the platform (IAM) rather than in-process.
	if s.deps.MetricsHandler != nil {
		s.mux.Handle("GET "+s.deps.Config.Obs.MetricsPath, s.deps.MetricsHandler)
	}
}

// Handler returns the root handler with global middleware applied.
//
// Order is outermost-last: request ID wraps everything, so both the access log
// and any recovered panic can report the ID. Recovery sits inside the access
// log so a panicking request is still logged with its 500.
func (s *Server) Handler() http.Handler {
	var h http.Handler = s.mux
	h = s.withRecovery(h)
	h = s.withAccessLog(h)
	h = s.withRequestID(h)
	return h
}

// protected applies auth then rate limiting. Order matters: rate limits are
// per-tenant, so the tenant must be known before the limiter can be consulted.
func (s *Server) protected(h http.Handler) http.Handler {
	return s.withAuth(s.withRateLimit(h))
}

// ListenAndServe runs the server until ctx is canceled, then drains in-flight
// requests within the configured shutdown timeout.
func (s *Server) ListenAndServe(ctx context.Context) error {
	cfg := s.deps.Config.Server
	srv := &http.Server{
		Addr:        cfg.Addr,
		Handler:     s.Handler(),
		ReadTimeout: cfg.ReadTimeout,
		// WriteTimeout must exceed the longest expected stream; a short value
		// silently truncates long completions.
		WriteTimeout: cfg.WriteTimeout,
		BaseContext:  func(net.Listener) context.Context { return ctx },
	}

	errCh := make(chan error, 1)
	go func() {
		s.log.Info("gateway listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		s.log.Info("shutting down", "timeout", cfg.ShutdownTimeout)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}

// handleHealth reports process liveness. It never checks dependencies: a failing
// store should not cause the orchestrator to kill an otherwise healthy process.
func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleReady reports whether the gateway can serve traffic. Stores are checked
// because a gateway that cannot reach its providers should leave the load
// balancer pool.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	status := map[string]string{"status": "ok"}
	code := http.StatusOK

	if s.deps.Cache != nil {
		if err := s.deps.Cache.Ping(ctx); err != nil {
			// Degraded, not down: the cache is an optimization, and refusing
			// traffic over it would turn a Redis blip into a full outage.
			status["cache"] = "degraded: " + err.Error()
		}
	}
	writeJSON(w, code, status)
}

// handleModels lists configured aliases in the OpenAI /v1/models shape.
func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	out := struct {
		Object string  `json:"object"`
		Data   []model `json:"data"`
	}{Object: "list"}

	for _, a := range s.deps.Router.Aliases() {
		out.Data = append(out.Data, model{ID: a, Object: "model", Created: 0, OwnedBy: "gatewayllm"})
	}
	writeJSON(w, http.StatusOK, out)
}

// handleEmbeddings proxies an embeddings request to a routed provider. It shares
// the router so an embeddings alias can fail over exactly like a chat alias.
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req provider.EmbeddingRequest
	if !decodeJSON(w, r, s.deps.Config.Server.MaxBodyBytes, &req) {
		return
	}
	if req.Model == "" {
		writeBadRequest(w, "model is required")
		return
	}
	if len(req.Input) == 0 {
		writeBadRequest(w, "input is required")
		return
	}

	targets, err := s.deps.Router.Route(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, "invalid_request_error", err.Error(), "model_not_found")
		return
	}

	// Embeddings are not cached: they are cheap relative to a completion, and a
	// vector cache keyed on text would duplicate the semantic tier's own store.
	var lastErr error
	for _, t := range targets {
		local := req
		local.Model = t.Model
		resp, err := t.Provider.Embed(r.Context(), &local)
		if err == nil {
			resp.Model = req.Model // report the alias the client asked for
			writeJSON(w, http.StatusOK, resp)
			return
		}
		lastErr = err
		if errors.Is(err, provider.ErrUnsupported) {
			continue // this backend has no embeddings API; try the next
		}
		if pe := provider.AsError(err); pe != nil && !pe.Failoverable() {
			break
		}
	}
	if errors.Is(lastErr, provider.ErrUnsupported) {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"no configured provider for this model supports embeddings", "unsupported")
		return
	}
	writeProviderError(w, lastErr)
}

// writeJSON renders v as a JSON response.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Debug("write json response failed", "err", err)
	}
}

// decodeJSON reads and validates a JSON body, writing an error response and
// returning false if it cannot. The body is size-capped so an oversized upload
// cannot exhaust memory.
func decodeJSON(w http.ResponseWriter, r *http.Request, maxBytes int64, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	dec := json.NewDecoder(r.Body)
	// Reject unknown fields: silently ignoring a misspelled parameter would let
	// a caller believe a setting took effect when it never reached the provider.
	dec.DisallowUnknownFields()

	if err := dec.Decode(v); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error",
				"request body too large", "")
			return false
		}
		writeBadRequest(w, "invalid JSON body: "+err.Error())
		return false
	}
	return true
}
