package api

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"

	"github.com/yash/gatewayllm/internal/cache"
	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/resilience"
	"github.com/yash/gatewayllm/internal/router"
)

const testKey = "glm_test_key"

// harness is a fully wired gateway backed by miniredis and a mock provider, so
// the whole request path is exercised with no network and no containers.
type harness struct {
	server *httptest.Server
	mock   *provider.Mock
	redis  *miniredis.Miniredis
	cfg    *config.Config
}

func newHarness(t *testing.T, mutate func(*config.Config)) *harness {
	t.Helper()

	mr := miniredis.RunT(t)
	rdb := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })

	cfg := &config.Config{
		Server: config.Server{MaxBodyBytes: 1 << 20},
		Cache: config.Cache{
			Enabled:        true,
			ExactTTL:       time.Hour,
			MaxTemperature: 0.3,
		},
		Resilience: config.Resilience{MaxAttempts: 2, BaseBackoff: time.Millisecond, MaxBackoff: time.Millisecond},
		Pricing: map[string]config.Price{
			"mock-model": {InputPerMillion: 1, OutputPerMillion: 2},
		},
	}
	if mutate != nil {
		mutate(cfg)
	}

	mock := provider.NewMock("mock-a", "the answer is 42")
	mock.AvailableModels = []string{"mock-model"}
	reg := provider.NewRegistry(mock)

	rt, err := router.New(config.Router{
		Aliases: map[string]config.Alias{
			"test-model": {Strategy: "priority", Targets: []config.Target{{Provider: "mock-a", Model: "mock-model"}}},
		},
	}, reg)
	if err != nil {
		t.Fatalf("router: %v", err)
	}

	breakerCfg := resilience.BreakerConfig{FailureThreshold: 3, OpenDuration: time.Second, HalfOpenProbes: 1}
	exec := resilience.New(resilience.Config{
		MaxAttempts: cfg.Resilience.MaxAttempts,
		BaseBackoff: cfg.Resilience.BaseBackoff,
		MaxBackoff:  cfg.Resilience.MaxBackoff,
	}, resilience.NewBreaker(resilience.NewLocalBreakerStore(breakerCfg), false))

	var c *cache.Cache
	if cfg.Cache.Enabled {
		c = cache.New(cache.Options{Config: cfg.Cache, Exact: cache.NewExactTier(rdb)})
	}

	var limiter Limiter
	if cfg.RateLimit.Enabled {
		limiter = NewRedisLimiter(rdb, cfg.RateLimit.DefaultRPM, cfg.RateLimit.Burst)
	}

	srv := NewServer(Deps{
		Config: cfg,
		Router: rt,
		Exec:   exec,
		Auth:   NewStaticAuthenticator(map[string]*Tenant{testKey: {ID: "t1", Name: "test", KeyID: "k1"}}),
		Limit:  limiter,
		Cache:  c,
	})

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	return &harness{server: ts, mock: mock, redis: mr, cfg: cfg}
}

// post issues a chat request with the given body.
func (h *harness) post(t *testing.T, body string, headers ...string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, h.server.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testKey)
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	return resp
}

func chatBody(prompt string, temp float64) string {
	b, err := json.Marshal(provider.ChatRequest{
		Model:       "test-model",
		Temperature: &temp,
		Messages:    []provider.Message{{Role: provider.RoleUser, Content: prompt}},
	})
	if err != nil {
		panic(err)
	}
	return string(b)
}

func decodeChat(t *testing.T, resp *http.Response) *provider.ChatResponse {
	t.Helper()
	defer resp.Body.Close()
	var out provider.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return &out
}

// TestChatCompletions_Basic asserts the OpenAI-compatible happy path.
func TestChatCompletions_Basic(t *testing.T) {
	h := newHarness(t, nil)
	resp := h.post(t, chatBody("what is the answer", 0.0))

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	out := decodeChat(t, resp)
	if out.Text() != "the answer is 42" {
		t.Errorf("text = %q", out.Text())
	}
	// The client asked for the alias and must see the alias back, not the
	// upstream model that happened to serve it.
	if out.Model != "test-model" {
		t.Errorf("model = %q, want the requested alias 'test-model'", out.Model)
	}
	if out.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", out.Object)
	}
}

// TestExactCache_HitSkipsProvider is the headline claim of the exact tier: a
// repeat request must not reach the provider at all.
func TestExactCache_HitSkipsProvider(t *testing.T) {
	h := newHarness(t, nil)
	body := chatBody("cache me", 0.0)

	first := h.post(t, body)
	if got := first.Header.Get("X-Cache"); got != string(cache.StatusMiss) {
		t.Errorf("first request X-Cache = %q, want miss", got)
	}
	firstOut := decodeChat(t, first)

	// The write is async; wait for it to land rather than racing it.
	waitFor(t, func() bool { return h.mock.Calls() == 1 && len(h.redis.Keys()) > 0 })

	second := h.post(t, body)
	if got := second.Header.Get("X-Cache"); got != string(cache.StatusExactHit) {
		t.Errorf("second request X-Cache = %q, want exact_hit", got)
	}
	secondOut := decodeChat(t, second)

	if h.mock.Calls() != 1 {
		t.Errorf("provider called %d times; a cache hit must not reach the provider", h.mock.Calls())
	}
	if secondOut.Text() != firstOut.Text() {
		t.Errorf("cached text %q != original %q", secondOut.Text(), firstOut.Text())
	}
	// Usage is replayed so the saving stays visible to the client.
	if secondOut.Usage.CompletionTokens != firstOut.Usage.CompletionTokens {
		t.Errorf("cached usage %d != original %d; a hit must replay the original token counts",
			secondOut.Usage.CompletionTokens, firstOut.Usage.CompletionTokens)
	}
}

// TestExactCache_DifferentPromptMisses asserts the cache does not over-match.
func TestExactCache_DifferentPromptMisses(t *testing.T) {
	h := newHarness(t, nil)

	h.post(t, chatBody("first question", 0.0)).Body.Close()
	waitFor(t, func() bool { return h.mock.Calls() == 1 })

	resp := h.post(t, chatBody("a completely different question", 0.0))
	if got := resp.Header.Get("X-Cache"); got != string(cache.StatusMiss) {
		t.Errorf("X-Cache = %q, want miss for a different prompt", got)
	}
	resp.Body.Close()
	if h.mock.Calls() != 2 {
		t.Errorf("provider calls = %d, want 2: a different prompt must reach the provider", h.mock.Calls())
	}
}

// TestCache_HighTemperatureBypasses asserts a creative request is never served
// from cache, however many times it repeats.
func TestCache_HighTemperatureBypasses(t *testing.T) {
	h := newHarness(t, nil)
	body := chatBody("write me a poem", 0.9)

	for i := 1; i <= 2; i++ {
		resp := h.post(t, body)
		if got := resp.Header.Get("X-Cache"); got != string(cache.StatusBypass) {
			t.Errorf("request %d X-Cache = %q, want bypass at temperature 0.9", i, got)
		}
		resp.Body.Close()
	}
	if h.mock.Calls() != 2 {
		t.Errorf("provider calls = %d, want 2: high-temperature requests must always hit the provider", h.mock.Calls())
	}
}

// TestCache_NoCacheHeaderBypasses asserts an explicit opt-out is honored even
// when the request is otherwise perfectly cacheable.
func TestCache_NoCacheHeaderBypasses(t *testing.T) {
	h := newHarness(t, nil)
	body := chatBody("deterministic question", 0.0)

	h.post(t, body).Body.Close()
	waitFor(t, func() bool { return h.mock.Calls() == 1 })

	resp := h.post(t, body, "X-No-Cache", "true")
	if got := resp.Header.Get("X-Cache"); got != string(cache.StatusBypass) {
		t.Errorf("X-Cache = %q, want bypass when X-No-Cache is set", got)
	}
	resp.Body.Close()
	if h.mock.Calls() != 2 {
		t.Errorf("provider calls = %d, want 2: X-No-Cache must force a fresh call", h.mock.Calls())
	}
}

// TestCache_CacheControlNoCacheBypasses asserts the standard header works too.
func TestCache_CacheControlNoCacheBypasses(t *testing.T) {
	h := newHarness(t, nil)
	body := chatBody("another question", 0.0)

	h.post(t, body).Body.Close()
	waitFor(t, func() bool { return h.mock.Calls() == 1 })

	resp := h.post(t, body, "Cache-Control", "no-cache")
	if got := resp.Header.Get("X-Cache"); got != string(cache.StatusBypass) {
		t.Errorf("X-Cache = %q, want bypass for Cache-Control: no-cache", got)
	}
	resp.Body.Close()
}

// TestCache_TenantIsolation is a security property: one tenant must never be
// served another tenant's cached completion.
func TestCache_TenantIsolation(t *testing.T) {
	h := newHarness(t, nil)

	// Rebuild the server with two tenants sharing one cache.
	auth := NewStaticAuthenticator(map[string]*Tenant{
		"key-a": {ID: "tenant-a", KeyID: "ka"},
		"key-b": {ID: "tenant-b", KeyID: "kb"},
	})
	srvDeps := h.deps(t, auth)
	ts := httptest.NewServer(NewServer(srvDeps).Handler())
	defer ts.Close()

	post := func(key string) *http.Response {
		req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions",
			strings.NewReader(chatBody("shared prompt", 0.0)))
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		return resp
	}

	post("key-a").Body.Close()
	waitFor(t, func() bool { return h.mock.Calls() == 1 })

	// Same prompt, different tenant: must miss.
	resp := post("key-b")
	if got := resp.Header.Get("X-Cache"); got != string(cache.StatusMiss) {
		t.Errorf("X-Cache = %q, want miss: tenant-b must not read tenant-a's cache", got)
	}
	resp.Body.Close()
	if h.mock.Calls() != 2 {
		t.Errorf("provider calls = %d, want 2: a second tenant must get its own provider call", h.mock.Calls())
	}
}

// TestStreaming asserts the SSE contract: incremental deltas then [DONE].
func TestStreaming(t *testing.T) {
	h := newHarness(t, nil)
	resp := h.post(t, `{"model":"test-model","temperature":0,"stream":true,"messages":[{"role":"user","content":"stream this"}]}`)
	defer resp.Body.Close()

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	var content strings.Builder
	var sawDone bool
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk provider.ChatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("chunk is not valid JSON: %v", err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("object = %q, want chat.completion.chunk", chunk.Object)
		}
		for _, c := range chunk.Choices {
			content.WriteString(c.Delta.Content)
		}
	}

	if !sawDone {
		t.Error("stream must terminate with data: [DONE]")
	}
	if content.String() != "the answer is 42" {
		t.Errorf("assembled stream = %q, want the full reply", content.String())
	}
}

// TestStreaming_CacheHitReplaysAsStream asserts the response shape does not
// change with cache state: a client that asked for a stream gets a stream.
func TestStreaming_CacheHitReplaysAsStream(t *testing.T) {
	h := newHarness(t, nil)
	body := `{"model":"test-model","temperature":0,"stream":true,"messages":[{"role":"user","content":"stream cached"}]}`

	first := h.post(t, body)
	drain(first)
	waitFor(t, func() bool { return h.mock.Calls() == 1 && len(h.redis.Keys()) > 0 })

	second := h.post(t, body)
	defer second.Body.Close()

	if got := second.Header.Get("X-Cache"); got != string(cache.StatusExactHit) {
		t.Errorf("X-Cache = %q, want exact_hit", got)
	}
	if ct := second.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q: a cached hit must still stream when stream=true", ct)
	}

	var content strings.Builder
	sc := bufio.NewScanner(second.Body)
	for sc.Scan() {
		payload, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "data:")
		if !ok {
			continue
		}
		payload = strings.TrimSpace(payload)
		if payload == "[DONE]" {
			break
		}
		var chunk provider.ChatChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, c := range chunk.Choices {
			content.WriteString(c.Delta.Content)
		}
	}
	if content.String() != "the answer is 42" {
		t.Errorf("cached stream = %q, want the full reply", content.String())
	}
	if h.mock.Calls() != 1 {
		t.Errorf("provider calls = %d: a cached stream must not reach the provider", h.mock.Calls())
	}
}

// --- auth ---

func TestAuth_MissingKey(t *testing.T) {
	h := newHarness(t, nil)
	req, _ := http.NewRequest(http.MethodPost, h.server.URL+"/v1/chat/completions",
		strings.NewReader(chatBody("hi", 0)))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

func TestAuth_InvalidKey(t *testing.T) {
	h := newHarness(t, nil)
	req, _ := http.NewRequest(http.MethodPost, h.server.URL+"/v1/chat/completions",
		strings.NewReader(chatBody("hi", 0)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer wrong-key")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
	// The error body must stay OpenAI-shaped so SDK error handling works.
	var body errorBody
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("error body is not valid JSON: %v", err)
	}
	if body.Error.Type != "authentication_error" {
		t.Errorf("error type = %q, want authentication_error", body.Error.Type)
	}
}

// --- validation ---

func TestValidation(t *testing.T) {
	h := newHarness(t, nil)

	cases := []struct {
		name string
		body string
		want int
	}{
		{"missing model", `{"messages":[{"role":"user","content":"hi"}]}`, http.StatusBadRequest},
		{"empty messages", `{"model":"test-model","messages":[]}`, http.StatusBadRequest},
		{"malformed json", `{"model":`, http.StatusBadRequest},
		{"temperature out of range", `{"model":"test-model","temperature":5,"messages":[{"role":"user","content":"hi"}]}`, http.StatusBadRequest},
		{"unknown model", `{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}`, http.StatusNotFound},
		{"unknown field", `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"bogus":1}`, http.StatusBadRequest},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := h.post(t, tc.body)
			defer resp.Body.Close()
			if resp.StatusCode != tc.want {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.want)
			}
		})
	}
}

// --- rate limiting ---

// TestRateLimit asserts the bucket rejects once drained and advertises the limit.
func TestRateLimit(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.RateLimit = config.RateLimit{Enabled: true, DefaultRPM: 2, Burst: 2}
		// Disable caching so every request must pass the limiter rather than
		// being served from cache.
		c.Cache.Enabled = false
	})

	var codes []int
	for i := 0; i < 4; i++ {
		resp := h.post(t, chatBody("rate limited", 0.0))
		codes = append(codes, resp.StatusCode)
		resp.Body.Close()
	}

	// Burst of 2 allows the first two; the rest are rejected.
	if codes[0] != 200 || codes[1] != 200 {
		t.Errorf("first two requests = %v, want both 200 within the burst", codes[:2])
	}
	if codes[2] != http.StatusTooManyRequests || codes[3] != http.StatusTooManyRequests {
		t.Errorf("requests 3-4 = %v, want both 429 once the bucket is drained", codes[2:])
	}
}

func TestRateLimit_Headers(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.RateLimit = config.RateLimit{Enabled: true, DefaultRPM: 1, Burst: 1}
		c.Cache.Enabled = false
	})

	first := h.post(t, chatBody("hi", 0.0))
	first.Body.Close()
	if got := first.Header.Get("X-RateLimit-Limit"); got != "1" {
		t.Errorf("X-RateLimit-Limit = %q, want 1", got)
	}

	second := h.post(t, chatBody("hi", 0.0))
	defer second.Body.Close()
	if second.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", second.StatusCode)
	}
	// Retry-After lets a client back off correctly instead of hammering.
	if got := second.Header.Get("Retry-After"); got == "" {
		t.Error("a 429 must carry Retry-After so the client knows when to retry")
	}
}

// --- models ---

func TestModelsEndpoint(t *testing.T) {
	h := newHarness(t, nil)
	req, _ := http.NewRequest(http.MethodGet, h.server.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+testKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()

	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Object != "list" || len(out.Data) != 1 || out.Data[0].ID != "test-model" {
		t.Errorf("models = %+v, want the single alias 'test-model'", out)
	}
}

func TestHealthz(t *testing.T) {
	h := newHarness(t, nil)
	resp, err := http.Get(h.server.URL + "/healthz")
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200: health must not require auth", resp.StatusCode)
	}
}

// --- helpers ---

// deps rebuilds Deps from the harness with a different authenticator.
func (h *harness) deps(t *testing.T, auth Authenticator) Deps {
	t.Helper()
	rdb := redis.NewClient(&redis.Options{Addr: h.redis.Addr()})
	t.Cleanup(func() { _ = rdb.Close() })

	reg := provider.NewRegistry(h.mock)
	rt, err := router.New(config.Router{
		Aliases: map[string]config.Alias{
			"test-model": {Strategy: "priority", Targets: []config.Target{{Provider: "mock-a", Model: "mock-model"}}},
		},
	}, reg)
	if err != nil {
		t.Fatalf("router: %v", err)
	}
	exec := resilience.New(resilience.Config{MaxAttempts: 1, BaseBackoff: time.Millisecond, MaxBackoff: time.Millisecond},
		resilience.NewBreaker(resilience.NewLocalBreakerStore(resilience.BreakerConfig{FailureThreshold: 3}), false))

	return Deps{
		Config: h.cfg,
		Router: rt,
		Exec:   exec,
		Auth:   auth,
		Cache:  cache.New(cache.Options{Config: h.cfg.Cache, Exact: cache.NewExactTier(rdb)}),
	}
}

// waitFor polls until cond holds, so tests synchronize on the async cache write
// instead of sleeping a guessed duration.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("timed out waiting for the async cache write to land")
}

func drain(resp *http.Response) {
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
	}
}
