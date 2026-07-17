package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/yash/gatewayllm/internal/cache"
	"github.com/yash/gatewayllm/internal/meter"
	"github.com/yash/gatewayllm/internal/obs"
	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/router"
)

// handleChatCompletions serves POST /v1/chat/completions.
//
// The flow is: authenticate (middleware) → cache lookup → route → provider call
// with retries → cache write → meter. A cache hit returns before routing, so no
// provider is contacted at all.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var req provider.ChatRequest
	if !decodeJSON(w, r, s.deps.Config.Server.MaxBodyBytes, &req) {
		return
	}
	if err := req.Validate(); err != nil {
		writeBadRequest(w, err.Error())
		return
	}

	tenant := TenantFrom(r.Context())

	// Root span for the request. Every stage below opens a child, producing the
	// auth → cache-lookup → route → provider-call → cache-write waterfall; a
	// cache hit returns before the provider span opens, so the hit visibly
	// skips it in the trace.
	ctx, endSpan := obs.StartSpan(r.Context(), "chat.completions",
		obs.AttrTenant.String(tenant.ID),
		obs.AttrModelAlias.String(req.Model),
		obs.AttrStreamed.Bool(req.Stream),
	)
	defer endSpan()

	// Expose the cache outcome to the access log, which runs after this handler.
	cacheStatus := new(string)
	ctx = context.WithValue(ctx, ctxKeyCacheStatus, cacheStatus)
	r = r.WithContext(ctx)

	rc := &requestContext{
		server:   s,
		tenant:   tenant,
		req:      &req,
		start:    start,
		reqID:    RequestIDFrom(ctx),
		status:   cacheStatus,
		streamed: req.Stream,
	}

	// --- cache lookup ---
	var lookup cache.Result
	cacheable := false
	if s.deps.Cache != nil {
		var reason cache.BypassReason
		cacheable, reason = s.deps.Cache.Cacheable(&req, noCacheRequested(r))
		if cacheable {
			lookupCtx, endLookup := obs.StartSpan(ctx, "cache.lookup")
			lookup = s.deps.Cache.Lookup(lookupCtx, tenant.ID, &req)
			endLookup()
		} else {
			lookup = cache.Result{Status: cache.StatusBypass}
			s.log.Debug("cache bypassed", "reason", reason, "request_id", rc.reqID)
		}
	} else {
		lookup = cache.Result{Status: cache.StatusBypass}
	}
	*cacheStatus = string(lookup.Status)
	obs.SpanFrom(ctx).SetAttributes(obs.AttrCacheStatus.String(string(lookup.Status)))

	if lookup.Hit() {
		rc.serveFromCache(w, lookup)
		return
	}

	// --- route ---
	targets, err := s.deps.Router.Route(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, "invalid_request_error", err.Error(), "model_not_found")
		rc.meterFailure(http.StatusNotFound, "model_not_found")
		return
	}

	// --- provider call ---
	if req.Stream {
		rc.serveStream(w, r, targets, lookup, cacheable)
		return
	}
	rc.serveUnary(w, r, targets, lookup, cacheable)
}

// requestContext carries per-request state through the handler's stages, so the
// stages do not each need six parameters threaded through them.
type requestContext struct {
	server   *Server
	tenant   *Tenant
	req      *provider.ChatRequest
	start    time.Time
	reqID    string
	status   *string
	streamed bool
}

// noCacheRequested reports whether the client opted out of the cache. Both the
// gateway's own header and the standard Cache-Control directive are honored.
func noCacheRequested(r *http.Request) bool {
	if strings.EqualFold(r.Header.Get("X-No-Cache"), "true") {
		return true
	}
	return strings.Contains(strings.ToLower(r.Header.Get("Cache-Control")), "no-cache")
}

// serveFromCache returns a cached completion without contacting any provider.
func (rc *requestContext) serveFromCache(w http.ResponseWriter, res cache.Result) {
	e := res.Entry
	resp := &provider.ChatResponse{
		ID:      "chatcmpl-" + provider.NewID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   rc.req.Model,
		Choices: []provider.Choice{{
			Index:        0,
			Message:      provider.Message{Role: provider.RoleAssistant, Content: e.Completion},
			FinishReason: "stop",
		}},
		// Replay the original usage: the tokens were genuinely required to
		// produce this text, and reporting zeros would make the savings the
		// cache produced invisible to the client.
		Usage: provider.Usage{
			PromptTokens:     e.PromptTokens,
			CompletionTokens: e.CompletionTokens,
			TotalTokens:      e.PromptTokens + e.CompletionTokens,
		},
	}

	rc.writeCacheHeaders(w, res)
	if rc.streamed {
		// A cached hit still honors stream=true: the client asked for an event
		// stream, and the response shape must not depend on cache state.
		rc.replayAsStream(w, resp)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
	rc.meterCacheHit(res, e)
}

// writeCacheHeaders surfaces cache state to the client for debugging.
func (rc *requestContext) writeCacheHeaders(w http.ResponseWriter, res cache.Result) {
	w.Header().Set("X-Cache", string(res.Status))
	if res.Status == cache.StatusSemanticHit {
		w.Header().Set("X-Cache-Similarity", fmt.Sprintf("%.4f", res.Similarity))
	}
}

// replayAsStream emits a complete response as a single-delta event stream.
func (rc *requestContext) replayAsStream(w http.ResponseWriter, resp *provider.ChatResponse) {
	sw, err := newStreamWriter(w)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error(), "")
		return
	}
	stop := "stop"
	_ = sw.send(&provider.ChatChunk{
		ID: resp.ID, Object: "chat.completion.chunk", Created: resp.Created, Model: resp.Model,
		Choices: []provider.StreamChoice{{
			Index: 0,
			Delta: provider.StreamDelta{Role: provider.RoleAssistant, Content: resp.Text()},
		}},
	})
	_ = sw.send(&provider.ChatChunk{
		ID: resp.ID, Object: "chat.completion.chunk", Created: resp.Created, Model: resp.Model,
		Choices: []provider.StreamChoice{{Index: 0, FinishReason: &stop}},
		Usage:   &resp.Usage,
	})
	sw.done()
}

// serveUnary handles a non-streaming completion.
func (rc *requestContext) serveUnary(w http.ResponseWriter, r *http.Request, targets []router.Target, lookup cache.Result, cacheable bool) {
	s := rc.server
	callCtx, endCall := obs.StartSpan(r.Context(), "provider.call")
	resp, target, err := s.deps.Exec.Chat(callCtx, targets, rc.req)
	endCall()
	if err == nil {
		obs.SpanFrom(r.Context()).SetAttributes(
			obs.AttrProvider.String(target.Provider.Name()),
			obs.AttrModel.String(target.Model),
			obs.AttrTokensIn.Int(resp.Usage.PromptTokens),
			obs.AttrTokensOut.Int(resp.Usage.CompletionTokens),
		)
	}
	if err != nil {
		writeProviderError(w, err)
		rc.meterProviderFailure(err)
		return
	}

	// The response reports the alias the client requested, not the upstream
	// model that served it: which backend answered is the gateway's business,
	// and leaking it would make failover a visible behaviour change.
	resp.Model = rc.req.Model

	w.Header().Set("X-Cache", string(lookup.Status))
	w.Header().Set("X-Provider", target.Provider.Name())
	writeJSON(w, http.StatusOK, resp)

	// Attribute usage to the router's resolved model, not to whatever the
	// provider echoed back. Providers routinely return a more specific ID than
	// the one they were called with (gpt-4o -> gpt-4o-2024-08-06), which would
	// miss the price table and silently book the request as free.
	if cacheable {
		rc.storeAsync(target, target.Model, resp.Text(), resp.Usage, lookup.Vector)
	}
	rc.meterSuccess(target, target.Model, resp.Usage, lookup.Status)
}

// serveStream handles a streaming completion, forwarding chunks as they arrive
// and accumulating the full text so the result can still be cached.
func (rc *requestContext) serveStream(w http.ResponseWriter, r *http.Request, targets []router.Target, lookup cache.Result, cacheable bool) {
	s := rc.server

	// Headers must be set before newStreamWriter, which writes the status line;
	// anything set afterwards is silently discarded.
	w.Header().Set("X-Cache", string(lookup.Status))

	sw, err := newStreamWriter(w)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error(), "")
		return
	}

	var (
		full  strings.Builder
		usage provider.Usage
		wrote bool
	)

	callCtx, endCall := obs.StartSpan(r.Context(), "provider.call")
	defer endCall()
	target, err := s.deps.Exec.ChatStream(callCtx, targets, rc.req, func(c *provider.ChatChunk) error {
		c.Model = rc.req.Model // report the alias, as the unary path does
		if c.Usage != nil {
			usage = *c.Usage
		}
		for _, ch := range c.Choices {
			full.WriteString(ch.Delta.Content)
		}
		wrote = true
		return sw.send(c)
	})

	if err != nil {
		// Once bytes are on the wire the status line is already sent, so the
		// error cannot become an HTTP status. Emit it as a terminal SSE event
		// instead, which is the only way a streaming client can learn the
		// completion is truncated rather than finished.
		if wrote {
			sw.sendError(err)
			sw.done()
		} else {
			writeProviderError(w, err)
		}
		rc.meterProviderFailure(err)
		return
	}
	sw.done()

	if cacheable && full.Len() > 0 {
		rc.storeAsync(target, target.Model, full.String(), usage, lookup.Vector)
	}
	rc.meterSuccess(target, target.Model, usage, lookup.Status)
}

// storeAsync writes the completion to cache in the background.
//
// It is detached from the request context deliberately: the client's context is
// canceled the moment the response finishes, and inheriting it would cancel
// nearly every cache write before it landed.
func (rc *requestContext) storeAsync(target router.Target, upstreamModel, text string, usage provider.Usage, vec []float32) {
	if rc.server.deps.Cache == nil || text == "" {
		return
	}
	entry := &cache.Entry{
		Completion:       text,
		Model:            rc.req.Model,  // the alias, for the compatibility guard
		UpstreamModel:    upstreamModel, // the real model, for pricing a future hit
		Provider:         target.Provider.Name(),
		Temp:             rc.req.Temp(),
		MaxTokens:        rc.req.MaxTok(),
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
	}
	// Copy the request: the caller's may be reused, and the goroutine outlives
	// this handler.
	reqCopy := *rc.req
	tenantID := rc.tenant.ID

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		rc.server.deps.Cache.Store(ctx, tenantID, &reqCopy, entry, vec)
	}()
}

// --- metering ---

func (rc *requestContext) meterSuccess(target router.Target, upstreamModel string, usage provider.Usage, status cache.Status) {
	providerName := ""
	if target.Provider != nil {
		providerName = target.Provider.Name()
	}
	cost, priced := rc.server.pricing().Cost(providerName, upstreamModel, usage.PromptTokens, usage.CompletionTokens)
	if !priced && usage.TotalTokens > 0 {
		// Without this the model silently books as free and the cost dashboard
		// understates real spend.
		rc.server.log.Warn("no price configured: this request is booked at zero cost",
			"provider", providerName, "model", upstreamModel)
	}

	if mx := rc.server.deps.Metrics; mx != nil {
		mx.RecordUsage(providerName, upstreamModel, usage.PromptTokens, usage.CompletionTokens, cost, 0, string(status))
	}

	m := rc.server.deps.Meter
	if m == nil {
		return
	}
	m.Record(meter.Record{
		RequestID:        rc.reqID,
		TenantID:         rc.tenant.ID,
		KeyID:            rc.tenant.KeyID,
		Provider:         providerName,
		Model:            upstreamModel,
		ModelAlias:       rc.req.Model,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		CostUSD:          cost,
		CacheStatus:      string(status),
		Streamed:         rc.streamed,
		StatusCode:       http.StatusOK,
		LatencyMS:        time.Since(rc.start).Milliseconds(),
		// A served response took at least one attempt. Per-request retry and
		// failover counts are carried in aggregate by gateway_provider_attempts_total
		// rather than duplicated per row.
		Attempts: 1,
	})
}

// meterCacheHit records a hit. Cost is zero and the money not spent is recorded
// as savings, which is what makes the "cost saved" panel possible.
func (rc *requestContext) meterCacheHit(res cache.Result, e *cache.Entry) {
	// Price on the upstream model, not the alias e.Model: the price table is
	// keyed on the real model, so the alias would miss it and report zero saved.
	saved, _ := rc.server.pricing().Cost(e.Provider, e.UpstreamModel, e.PromptTokens, e.CompletionTokens)

	if mx := rc.server.deps.Metrics; mx != nil {
		// Cost is zero and the avoided spend is booked as savings: this is the
		// pair of numbers the "cost saved" panel is built from.
		mx.RecordUsage(e.Provider, e.Model, e.PromptTokens, e.CompletionTokens, 0, saved, string(res.Status))
	}

	m := rc.server.deps.Meter
	if m == nil {
		return
	}
	m.Record(meter.Record{
		RequestID:        rc.reqID,
		TenantID:         rc.tenant.ID,
		KeyID:            rc.tenant.KeyID,
		Provider:         e.Provider,
		Model:            e.Model,
		ModelAlias:       rc.req.Model,
		PromptTokens:     e.PromptTokens,
		CompletionTokens: e.CompletionTokens,
		CostUSD:          0,
		SavedUSD:         saved,
		CacheStatus:      string(res.Status),
		Streamed:         rc.streamed,
		StatusCode:       http.StatusOK,
		LatencyMS:        time.Since(rc.start).Milliseconds(),
	})
}

func (rc *requestContext) meterProviderFailure(err error) {
	kind := "internal"
	code := http.StatusInternalServerError
	if pe := provider.AsError(err); pe != nil {
		kind = string(pe.Kind)
		code = pe.HTTPStatus()
	}
	rc.meterFailure(code, kind)
}

func (rc *requestContext) meterFailure(code int, kind string) {
	m := rc.server.deps.Meter
	if m == nil {
		return
	}
	m.Record(meter.Record{
		RequestID:   rc.reqID,
		TenantID:    rc.tenant.ID,
		KeyID:       rc.tenant.KeyID,
		ModelAlias:  rc.req.Model,
		Model:       rc.req.Model,
		CacheStatus: derefStatus(rc.status),
		Streamed:    rc.streamed,
		StatusCode:  code,
		LatencyMS:   time.Since(rc.start).Milliseconds(),
		ErrorKind:   kind,
	})
}

func derefStatus(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// --- SSE writing ---

// streamWriter emits Server-Sent Events.
type streamWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
}

// newStreamWriter prepares the response for streaming.
func newStreamWriter(w http.ResponseWriter) (*streamWriter, error) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// Proxies that buffer would defeat streaming entirely, holding every chunk
	// until the response completed.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	sw := &streamWriter{w: w, rc: http.NewResponseController(w)}
	// The server's WriteTimeout applies to the whole response; a long stream
	// would trip it. Clearing the deadline lets the client's context and the
	// provider timeout govern instead.
	if err := sw.rc.SetWriteDeadline(time.Time{}); err != nil {
		// Not fatal: without a deadline reset, long streams may be cut, but
		// short ones still work.
		_ = err
	}
	sw.flush()
	return sw, nil
}

func (s *streamWriter) send(c *provider.ChatChunk) error {
	b, err := json.Marshal(c)
	if err != nil {
		return fmt.Errorf("marshal chunk: %w", err)
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return fmt.Errorf("write chunk: %w", err)
	}
	s.flush()
	return nil
}

// sendError emits an error as a stream event, for failures after the headers.
func (s *streamWriter) sendError(err error) {
	detail := errorDetail{Message: err.Error(), Type: "api_error"}
	if pe := provider.AsError(err); pe != nil {
		detail.Message = pe.Message
		detail.Type = errorType(pe.Kind)
		code := string(pe.Kind)
		detail.Code = &code
	}
	b, mErr := json.Marshal(errorBody{Error: detail})
	if mErr != nil {
		return
	}
	fmt.Fprintf(s.w, "data: %s\n\n", b)
	s.flush()
}

// done emits the terminator OpenAI clients wait for.
func (s *streamWriter) done() {
	fmt.Fprint(s.w, "data: [DONE]\n\n")
	s.flush()
}

func (s *streamWriter) flush() {
	// Without flushing, chunks sit in the response buffer and the client sees
	// nothing until the handler returns, which is not streaming at all.
	_ = s.rc.Flush()
}

// pricing returns the price table, built once at startup.
func (s *Server) pricing() *meter.Pricing { return s.prices }
