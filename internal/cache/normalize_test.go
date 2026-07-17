package cache

import (
	"strings"
	"testing"

	"github.com/yash/gatewayllm/internal/provider"
)

func req(model string, temp *float64, maxTok *int, msgs ...provider.Message) *provider.ChatRequest {
	return &provider.ChatRequest{Model: model, Messages: msgs, Temperature: temp, MaxTokens: maxTok}
}

func user(s string) provider.Message {
	return provider.Message{Role: provider.RoleUser, Content: s}
}

func f(v float64) *float64 { return &v }
func i(v int) *int         { return &v }

// TestExactKey_Identical asserts the property the whole exact tier rests on:
// requests that must produce the same completion share a key.
func TestExactKey_Identical(t *testing.T) {
	a := req("gpt-4o", f(0.2), i(100), user("hello"))
	b := req("gpt-4o", f(0.2), i(100), user("hello"))

	if ExactKey("t1", a) != ExactKey("t1", b) {
		t.Fatal("identical requests must share a cache key")
	}
}

// TestExactKey_Distinguishes covers the inverse and more important property: any
// field that can change the completion must change the key. A miss here is a
// wrong answer served to a real user.
func TestExactKey_Distinguishes(t *testing.T) {
	base := req("gpt-4o", f(0.2), i(100), user("hello"))
	baseKey := ExactKey("tenant-1", base)

	cases := []struct {
		name string
		req  *provider.ChatRequest
		tenant string
	}{
		{"different prompt", req("gpt-4o", f(0.2), i(100), user("goodbye")), "tenant-1"},
		{"different model", req("gpt-4o-mini", f(0.2), i(100), user("hello")), "tenant-1"},
		{"different temperature", req("gpt-4o", f(0.9), i(100), user("hello")), "tenant-1"},
		{"different max_tokens", req("gpt-4o", f(0.2), i(50), user("hello")), "tenant-1"},
		{"different role", req("gpt-4o", f(0.2), i(100), provider.Message{Role: provider.RoleSystem, Content: "hello"}), "tenant-1"},
		{"extra message", req("gpt-4o", f(0.2), i(100), user("hello"), user("more")), "tenant-1"},
		// Tenant isolation is enforced in the key itself, so one tenant can
		// never read another's cached completion.
		{"different tenant", base, "tenant-2"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ExactKey(tc.tenant, tc.req); got == baseKey {
				t.Errorf("%s must produce a different cache key, got the same", tc.name)
			}
		})
	}
}

// TestExactKey_IgnoresIrrelevantFields asserts fields that cannot change the
// output do not fragment the cache.
func TestExactKey_IgnoresIrrelevantFields(t *testing.T) {
	base := req("gpt-4o", f(0.2), i(100), user("hello"))

	streamed := req("gpt-4o", f(0.2), i(100), user("hello"))
	streamed.Stream = true
	if ExactKey("t", streamed) != ExactKey("t", base) {
		t.Error("stream must not affect the key: a streamed and unary request yield the same completion")
	}

	tagged := req("gpt-4o", f(0.2), i(100), user("hello"))
	tagged.User = "end-user-42"
	if ExactKey("t", tagged) != ExactKey("t", base) {
		t.Error("user must not affect the key: it is an upstream abuse tag, not a generation input")
	}
}

// TestExactKey_NormalizesWhitespace asserts insignificant formatting collapses.
func TestExactKey_NormalizesWhitespace(t *testing.T) {
	a := req("gpt-4o", f(0.2), i(100), user("hello   world"))
	b := req("gpt-4o", f(0.2), i(100), user("  hello world\n"))

	if ExactKey("t", a) != ExactKey("t", b) {
		t.Error("whitespace-only differences must share a key")
	}
}

// TestExactKey_PreservesCase guards a deliberate decision: case carries meaning
// in prompts, so folding it would collide materially different requests.
func TestExactKey_PreservesCase(t *testing.T) {
	a := req("gpt-4o", f(0.2), i(100), user("Polish the text"))
	b := req("gpt-4o", f(0.2), i(100), user("polish the text"))

	if ExactKey("t", a) == ExactKey("t", b) {
		t.Error("case differences must not collide: 'Polish' and 'polish' are different words")
	}
}

// TestExactKey_ModelCaseInsensitive asserts alias casing does not fragment.
func TestExactKey_ModelCaseInsensitive(t *testing.T) {
	a := req("GPT-4o", f(0.2), i(100), user("hi"))
	b := req("gpt-4o", f(0.2), i(100), user("hi"))

	if ExactKey("t", a) != ExactKey("t", b) {
		t.Error("model alias casing must not fragment the cache")
	}
}

// TestExactKey_NoFieldBoundaryCollision proves the length-prefixing works: two
// requests whose fields concatenate to the same bytes must not collide.
func TestExactKey_NoFieldBoundaryCollision(t *testing.T) {
	a := req("gpt-4o", f(0.2), i(100), user("ab"), user("c"))
	b := req("gpt-4o", f(0.2), i(100), user("a"), user("bc"))

	if ExactKey("t", a) == ExactKey("t", b) {
		t.Error("field boundaries must be encoded: ('ab','c') and ('a','bc') must not collide")
	}
}

// TestExactKey_TemperatureDefault asserts an omitted temperature is treated as
// the API default rather than as a distinct value.
func TestExactKey_TemperatureDefault(t *testing.T) {
	explicit := req("gpt-4o", f(1.0), nil, user("hi"))
	implicit := req("gpt-4o", nil, nil, user("hi"))

	if ExactKey("t", explicit) != ExactKey("t", implicit) {
		t.Error("omitted temperature must hash as the default of 1.0")
	}
}

// TestPromptText_IncludesFullConversation guards against embedding only the last
// message, which would make every follow-up question match every other.
func TestPromptText_IncludesFullConversation(t *testing.T) {
	r := req("gpt-4o", nil, nil,
		user("What is the capital of France?"),
		provider.Message{Role: provider.RoleAssistant, Content: "Paris."},
		user("What about Germany?"),
	)
	got := PromptText(r)

	for _, want := range []string{"capital of France", "Paris.", "What about Germany?"} {
		if !strings.Contains(got, want) {
			t.Errorf("prompt text must include %q for context, got: %s", want, got)
		}
	}
}
