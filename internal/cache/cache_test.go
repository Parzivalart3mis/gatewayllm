package cache

import (
	"testing"
	"time"

	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/provider"
)

func testCache() *Cache {
	return New(Options{Config: config.Cache{
		Enabled:        true,
		MaxTemperature: 0.3,
		ExactTTL:       time.Hour,
		Semantic: config.SemanticCache{
			Enabled:      true,
			Threshold:    0.95,
			TTL:          24 * time.Hour,
			MaxTempDelta: 0.05,
		},
	}})
}

// TestCacheable covers the guards that decide whether a request may touch the
// cache at all. These are the rules that keep the cache from returning answers
// the caller did not ask for.
func TestCacheable(t *testing.T) {
	c := testCache()

	cases := []struct {
		name       string
		temp       *float64
		noCache    bool
		wantOK     bool
		wantReason BypassReason
	}{
		{"deterministic request is cacheable", f(0.0), false, true, ""},
		{"at the temperature ceiling is cacheable", f(0.3), false, true, ""},
		{
			name: "above the ceiling bypasses: the caller asked for variety",
			temp: f(0.31), wantOK: false, wantReason: BypassTemperature,
		},
		{
			name: "creative request bypasses",
			temp: f(1.2), wantOK: false, wantReason: BypassTemperature,
		},
		{
			// The default temperature is 1.0, so an omitted temperature is a
			// creative request and must not be cached.
			name: "omitted temperature defaults to 1.0 and bypasses",
			temp: nil, wantOK: false, wantReason: BypassTemperature,
		},
		{
			name: "explicit no-cache is honored even when otherwise cacheable",
			temp: f(0.0), noCache: true, wantOK: false, wantReason: BypassHeader,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := req("gpt-4o", tc.temp, i(100), user("hi"))
			ok, reason := c.Cacheable(r, tc.noCache)
			if ok != tc.wantOK {
				t.Fatalf("Cacheable = %v, want %v", ok, tc.wantOK)
			}
			if !ok && reason != tc.wantReason {
				t.Errorf("reason = %q, want %q", reason, tc.wantReason)
			}
		})
	}
}

// TestCacheable_Disabled asserts a disabled cache reports as such.
func TestCacheable_Disabled(t *testing.T) {
	c := New(Options{Config: config.Cache{Enabled: false}})
	ok, reason := c.Cacheable(req("gpt-4o", f(0), i(10), user("hi")), false)
	if ok || reason != BypassDisabled {
		t.Errorf("got (%v, %q), want (false, %q)", ok, reason, BypassDisabled)
	}
}

// TestCompatible is the second correctness gate: a vector match means the
// prompts are similar, not that the cached answer is safe to serve.
func TestCompatible(t *testing.T) {
	c := testCache()
	now := time.Now().Unix()

	cases := []struct {
		name  string
		req   *provider.ChatRequest
		entry *Entry
		want  bool
	}{
		{
			name:  "same model and temperature is compatible",
			req:   req("gpt-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 50, CreatedAt: now},
			want:  true,
		},
		{
			name:  "model alias casing does not block a hit",
			req:   req("GPT-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 50, CreatedAt: now},
			want:  true,
		},
		{
			// The caller chose a model; serving another's output silently
			// breaks that choice.
			name:  "different model is incompatible",
			req:   req("gpt-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "claude-sonnet-5", Temp: 0.2, CompletionTokens: 50, CreatedAt: now},
			want:  false,
		},
		{
			name:  "temperature within delta is compatible",
			req:   req("gpt-4o", f(0.20), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.24, CompletionTokens: 50, CreatedAt: now},
			want:  true,
		},
		{
			name:  "temperature beyond delta is incompatible",
			req:   req("gpt-4o", f(0.0), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.3, CompletionTokens: 50, CreatedAt: now},
			want:  false,
		},
		{
			// Serving 500 tokens to a request capped at 100 would exceed a
			// limit the caller set deliberately.
			name:  "completion longer than max_tokens is incompatible",
			req:   req("gpt-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 500, CreatedAt: now},
			want:  false,
		},
		{
			name:  "completion within max_tokens is compatible",
			req:   req("gpt-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 100, CreatedAt: now},
			want:  true,
		},
		{
			name:  "unset max_tokens accepts any length",
			req:   req("gpt-4o", f(0.2), nil, user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 5000, CreatedAt: now},
			want:  true,
		},
		{
			name:  "entry past its TTL is rejected on read",
			req:   req("gpt-4o", f(0.2), i(100), user("hi")),
			entry: &Entry{Model: "gpt-4o", Temp: 0.2, CompletionTokens: 50, CreatedAt: time.Now().Add(-48 * time.Hour).Unix()},
			want:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := c.compatible(tc.req, tc.entry); got != tc.want {
				t.Errorf("compatible = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestResultHit asserts which statuses count as servable hits.
func TestResultHit(t *testing.T) {
	cases := map[Status]bool{
		StatusExactHit:    true,
		StatusSemanticHit: true,
		StatusMiss:        false,
		StatusBypass:      false,
		StatusError:       false,
	}
	for status, want := range cases {
		if got := (Result{Status: status}).Hit(); got != want {
			t.Errorf("Result{%s}.Hit() = %v, want %v", status, got, want)
		}
	}
}
