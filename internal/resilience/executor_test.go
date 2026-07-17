package resilience

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/router"
)

func testConfig() Config {
	return Config{
		MaxAttempts: 3,
		BaseBackoff: time.Millisecond, // keep tests fast
		MaxBackoff:  2 * time.Millisecond,
		Breaker:     BreakerConfig{FailureThreshold: 3, OpenDuration: 50 * time.Millisecond, HalfOpenProbes: 1},
	}
}

func targets(ps ...provider.Provider) []router.Target {
	out := make([]router.Target, len(ps))
	for i, p := range ps {
		out[i] = router.Target{Provider: p, Model: "m", Alias: "alias"}
	}
	return out
}

func chatReq() *provider.ChatRequest {
	return &provider.ChatRequest{
		Model:    "alias",
		Messages: []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
	}
}

func newExec(cfg Config, enabled bool) *Executor {
	return New(cfg, NewBreaker(NewLocalBreakerStore(cfg.Breaker), enabled))
}

// TestChat_Success is the baseline: one healthy provider, one call.
func TestChat_Success(t *testing.T) {
	p := provider.NewMock("primary", "hello")
	e := newExec(testConfig(), true)

	resp, target, err := e.Chat(context.Background(), targets(p), chatReq())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Text() != "hello" {
		t.Errorf("text = %q, want %q", resp.Text(), "hello")
	}
	if target.Provider.Name() != "primary" {
		t.Errorf("served by %q, want primary", target.Provider.Name())
	}
	if p.Calls() != 1 {
		t.Errorf("calls = %d, want 1", p.Calls())
	}
}

// TestChat_RetriesTransientFailure asserts a provider that fails then recovers
// is retried rather than abandoned.
func TestChat_RetriesTransientFailure(t *testing.T) {
	p := provider.NewMock("primary", "recovered")
	p.FailFirst = 2 // fail twice, succeed on the third attempt

	e := newExec(testConfig(), true)
	resp, _, err := e.Chat(context.Background(), targets(p), chatReq())
	if err != nil {
		t.Fatalf("Chat should have succeeded on retry: %v", err)
	}
	if resp.Text() != "recovered" {
		t.Errorf("text = %q, want %q", resp.Text(), "recovered")
	}
	if p.Calls() != 3 {
		t.Errorf("calls = %d, want 3 (initial + 2 retries)", p.Calls())
	}
}

// TestChat_ExhaustsRetriesThenFailsOver is the core resilience story: retries
// are spent on the primary, then the secondary takes over.
func TestChat_ExhaustsRetriesThenFailsOver(t *testing.T) {
	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindUnavailable, "primary", "down")
	secondary := provider.NewMock("secondary", "from secondary")

	e := newExec(testConfig(), true)
	resp, target, err := e.Chat(context.Background(), targets(primary, secondary), chatReq())
	if err != nil {
		t.Fatalf("Chat should have failed over: %v", err)
	}
	if resp.Text() != "from secondary" {
		t.Errorf("text = %q, want the secondary's reply", resp.Text())
	}
	if target.Provider.Name() != "secondary" {
		t.Errorf("served by %q, want secondary", target.Provider.Name())
	}
	if primary.Calls() != 3 {
		t.Errorf("primary calls = %d, want 3 (full retry budget)", primary.Calls())
	}
	if secondary.Calls() != 1 {
		t.Errorf("secondary calls = %d, want 1", secondary.Calls())
	}
}

// TestChat_NoRetryOnInvalidRequest asserts a client error is not retried and not
// failed over: every provider would reject it identically, so doing either just
// burns latency and quota.
func TestChat_NoRetryOnInvalidRequest(t *testing.T) {
	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindInvalidRequest, "primary", "bad messages")
	secondary := provider.NewMock("secondary", "should not be reached")

	e := newExec(testConfig(), true)
	_, _, err := e.Chat(context.Background(), targets(primary, secondary), chatReq())
	if err == nil {
		t.Fatal("expected an error for an invalid request")
	}
	if primary.Calls() != 1 {
		t.Errorf("primary calls = %d, want 1 (no retries on a client error)", primary.Calls())
	}
	if secondary.Calls() != 0 {
		t.Errorf("secondary calls = %d, want 0 (no failover on a client error)", secondary.Calls())
	}
}

// TestChat_NoFailoverOnContentFilter asserts a safety refusal is deterministic
// and not shopped around to other providers.
func TestChat_NoFailoverOnContentFilter(t *testing.T) {
	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindContentFilter, "primary", "refused")
	secondary := provider.NewMock("secondary", "should not be reached")

	e := newExec(testConfig(), true)
	_, _, err := e.Chat(context.Background(), targets(primary, secondary), chatReq())
	if err == nil {
		t.Fatal("expected a content filter error")
	}
	if secondary.Calls() != 0 {
		t.Errorf("secondary calls = %d, want 0: a refusal must not be retried elsewhere", secondary.Calls())
	}
}

// TestChat_AllProvidersFail asserts the joined error keeps the most actionable
// classification rather than flattening to a generic 500.
func TestChat_AllProvidersFail(t *testing.T) {
	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindUnavailable, "primary", "down")
	secondary := provider.NewMock("secondary", "")
	secondary.FailWith = provider.Errorf(provider.KindRateLimit, "secondary", "throttled")

	e := newExec(testConfig(), true)
	_, _, err := e.Chat(context.Background(), targets(primary, secondary), chatReq())
	if err == nil {
		t.Fatal("expected an error when every provider fails")
	}
	pe := provider.AsError(err)
	if pe == nil {
		t.Fatalf("error must remain a *provider.Error, got %T", err)
	}
	// Rate limit outranks unavailable: it tells the client to back off.
	if pe.Kind != provider.KindRateLimit {
		t.Errorf("kind = %q, want %q (the most actionable failure)", pe.Kind, provider.KindRateLimit)
	}
}

// TestBreaker_OpensAndSkipsProvider asserts the breaker's whole purpose: after
// repeated failures the provider stops being called at all.
func TestBreaker_OpensAndSkipsProvider(t *testing.T) {
	cfg := testConfig()
	cfg.MaxAttempts = 1 // one attempt per call, so failures map 1:1 to the breaker

	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindUnavailable, "primary", "down")
	secondary := provider.NewMock("secondary", "ok")

	e := newExec(cfg, true)
	ts := targets(primary, secondary)

	// Drive the primary to its failure threshold.
	for i := 0; i < cfg.Breaker.FailureThreshold; i++ {
		if _, _, err := e.Chat(context.Background(), ts, chatReq()); err != nil {
			t.Fatalf("call %d should have failed over to secondary: %v", i, err)
		}
	}
	callsBeforeOpen := primary.Calls()

	// The circuit is now open: further calls must skip the primary entirely.
	if _, _, err := e.Chat(context.Background(), ts, chatReq()); err != nil {
		t.Fatalf("call after open should still succeed via secondary: %v", err)
	}
	if primary.Calls() != callsBeforeOpen {
		t.Errorf("primary was called %d more times after the circuit opened; an open circuit must skip it",
			primary.Calls()-callsBeforeOpen)
	}
}

// TestBreaker_RecoversAfterCooldown asserts an open circuit probes and closes
// once the provider is healthy, so a transient outage does not permanently
// remove a provider.
func TestBreaker_RecoversAfterCooldown(t *testing.T) {
	cfg := testConfig()
	cfg.MaxAttempts = 1
	cfg.Breaker.OpenDuration = 30 * time.Millisecond

	p := provider.NewMock("primary", "ok")
	p.FailWith = provider.Errorf(provider.KindUnavailable, "primary", "down")

	e := newExec(cfg, true)
	ts := targets(p)

	for i := 0; i < cfg.Breaker.FailureThreshold; i++ {
		_, _, _ = e.Chat(context.Background(), ts, chatReq())
	}
	// Circuit open: the call is rejected without touching the provider.
	if _, _, err := e.Chat(context.Background(), ts, chatReq()); err == nil {
		t.Fatal("expected rejection while the circuit is open")
	}

	// Provider recovers, cooldown elapses.
	p.FailWith = nil
	time.Sleep(cfg.Breaker.OpenDuration + 10*time.Millisecond)

	if _, _, err := e.Chat(context.Background(), ts, chatReq()); err != nil {
		t.Fatalf("half-open probe should succeed once the provider recovers: %v", err)
	}
}

// TestBreaker_ClientErrorsDoNotTrip guards a subtle failure mode: if malformed
// requests counted against the circuit, one bad caller could trip the breaker
// and take the provider offline for every other tenant.
func TestBreaker_ClientErrorsDoNotTrip(t *testing.T) {
	cfg := testConfig()
	cfg.MaxAttempts = 1

	p := provider.NewMock("primary", "")
	p.FailWith = provider.Errorf(provider.KindInvalidRequest, "primary", "bad request")

	e := newExec(cfg, true)
	ts := targets(p)

	for i := 0; i < cfg.Breaker.FailureThreshold*2; i++ {
		_, _, _ = e.Chat(context.Background(), ts, chatReq())
	}

	allowed, state := e.Breaker().Allow(context.Background(), "primary")
	if !allowed || state == StateOpen {
		t.Error("client errors must not open the circuit: one bad caller would take the provider down for everyone")
	}
}

// TestChatStream_NoFailoverAfterFirstChunk asserts the streaming commit point.
// Once bytes reach the client, switching providers would splice two different
// completions into one response.
func TestChatStream_NoFailoverAfterFirstChunk(t *testing.T) {
	// A provider that emits a chunk and then fails.
	primary := &midStreamFailer{name: "primary"}
	secondary := provider.NewMock("secondary", "should not be reached")

	e := newExec(testConfig(), true)
	var got []string
	_, err := e.ChatStream(context.Background(), targets(primary, secondary), chatReq(),
		func(c *provider.ChatChunk) error {
			got = append(got, c.Choices[0].Delta.Content)
			return nil
		})

	if err == nil {
		t.Fatal("expected the mid-stream failure to surface")
	}
	if secondary.Calls() != 0 {
		t.Error("must not fail over after chunks reached the client: the response would splice two completions")
	}
	if len(got) != 1 || got[0] != "partial" {
		t.Errorf("chunks = %v, want the one chunk delivered before the failure", got)
	}
}

// TestChatStream_FailsOverBeforeFirstChunk asserts the inverse: a provider that
// fails before emitting anything is safely replaced.
func TestChatStream_FailsOverBeforeFirstChunk(t *testing.T) {
	primary := provider.NewMock("primary", "")
	primary.FailWith = provider.Errorf(provider.KindUnavailable, "primary", "down")
	secondary := provider.NewMock("secondary", "hello from secondary")

	e := newExec(testConfig(), true)
	var got string
	target, err := e.ChatStream(context.Background(), targets(primary, secondary), chatReq(),
		func(c *provider.ChatChunk) error {
			got += c.Choices[0].Delta.Content
			return nil
		})

	if err != nil {
		t.Fatalf("should have failed over before any bytes were sent: %v", err)
	}
	if target.Provider.Name() != "secondary" {
		t.Errorf("served by %q, want secondary", target.Provider.Name())
	}
	if got != "hello from secondary" {
		t.Errorf("text = %q, want the secondary's full reply", got)
	}
}

// TestChat_ContextCancellation asserts a canceled request stops immediately
// rather than working through the remaining targets.
func TestChat_ContextCancellation(t *testing.T) {
	p := provider.NewMock("slow", "never")
	p.Latency = time.Second

	e := newExec(testConfig(), true)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, _, err := e.Chat(ctx, targets(p), chatReq())
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Errorf("took %v: a canceled context must abort promptly, not run the retry budget", elapsed)
	}
}

// TestNoTargets asserts an empty routing result is reported clearly.
func TestNoTargets(t *testing.T) {
	e := newExec(testConfig(), true)
	_, _, err := e.Chat(context.Background(), nil, chatReq())
	if !errors.Is(err, ErrNoTargets) {
		t.Errorf("err = %v, want ErrNoTargets", err)
	}
}

// midStreamFailer emits one chunk, then fails: the case that decides whether
// streaming failover is safe.
type midStreamFailer struct{ name string }

func (m *midStreamFailer) Name() string     { return m.name }
func (m *midStreamFailer) Models() []string { return nil }

func (m *midStreamFailer) Chat(context.Context, *provider.ChatRequest) (*provider.ChatResponse, error) {
	return nil, provider.ErrUnsupported
}

func (m *midStreamFailer) Embed(context.Context, *provider.EmbeddingRequest) (*provider.EmbeddingResponse, error) {
	return nil, provider.ErrUnsupported
}

func (m *midStreamFailer) ChatStream(_ context.Context, req *provider.ChatRequest, onChunk func(*provider.ChatChunk) error) error {
	if err := onChunk(&provider.ChatChunk{
		Model:   req.Model,
		Choices: []provider.StreamChoice{{Delta: provider.StreamDelta{Content: "partial"}}},
	}); err != nil {
		return err
	}
	return provider.Errorf(provider.KindUnavailable, m.name, "connection dropped mid-stream")
}
