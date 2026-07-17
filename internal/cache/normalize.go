// Package cache implements the two-tier response cache: an exact-match tier in
// Redis and a semantic-similarity tier in Qdrant. It owns the decision of what
// may be cached, what counts as a hit, and what must never be served from cache.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/yash/gatewayllm/internal/provider"
)

// Key is the exact-tier cache key: a digest of everything that can change the
// completion.
type Key string

// String renders the key.
func (k Key) String() string { return string(k) }

// Redis returns the namespaced Redis key. The version prefix lets a change to
// the key derivation invalidate old entries wholesale instead of serving
// responses computed under different rules.
func (k Key) Redis() string { return "glm:v1:exact:" + string(k) }

// ExactKey derives the exact-match key for a request.
//
// Every field that can change the output must be in the digest, or the cache
// will serve one request's answer to a materially different one. Fields that
// cannot change the output (stream, user) must be excluded, or identical work
// will miss.
//
// Notably:
//   - The model alias is used, not the resolved upstream model: the alias is the
//     contract the client asked against, and which provider served it is a
//     routing detail that must not fragment the cache.
//   - stream is excluded: a streamed and a non-streamed request produce the same
//     completion, so a streaming call can legitimately be served from a cache
//     entry written by a non-streaming one.
//   - user is excluded: it is an abuse-tracking tag upstream, not an input to
//     generation. Tenant isolation is handled by namespacing, not by the digest.
func ExactKey(tenantID string, req *provider.ChatRequest) Key {
	h := sha256.New()

	// Length-prefix every field. Plain concatenation lets different inputs
	// collide: ("ab","c") and ("a","bc") would otherwise hash identically.
	write := func(parts ...string) {
		for _, p := range parts {
			fmt.Fprintf(h, "%d:%s|", len(p), p)
		}
	}

	write("tenant", tenantID)
	write("model", normalizeModel(req.Model))

	write("messages", strconv.Itoa(len(req.Messages)))
	for _, m := range req.Messages {
		write("role", m.Role, "name", m.Name, "content", normalizeContent(m.Content))
	}

	write("temperature", formatFloat(req.Temp()))
	write("max_tokens", strconv.Itoa(req.MaxTok()))
	if req.TopP != nil {
		write("top_p", formatFloat(*req.TopP))
	}
	if len(req.Stop) > 0 {
		// Stop sequences are order-significant to the provider, so they are
		// hashed in the order given rather than sorted.
		write("stop", strings.Join(req.Stop, "\x00"))
	}

	return Key(hex.EncodeToString(h.Sum(nil)))
}

// normalizeModel lowercases and trims the alias so casing differences do not
// fragment the cache.
func normalizeModel(m string) string {
	return strings.ToLower(strings.TrimSpace(m))
}

// normalizeContent collapses insignificant whitespace so that prompts differing
// only in indentation or trailing newlines share a cache entry.
//
// Case is deliberately preserved: it carries meaning in prompts (code
// identifiers, proper nouns, "SHOUTING"), and folding it would let materially
// different requests collide.
func normalizeContent(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// formatFloat renders a float canonically so 0.7 and 0.70 hash identically.
func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'g', 6, 64)
}

// PromptText renders the messages into the text embedded by the semantic tier.
//
// The whole conversation is included, not just the last user message: "what
// about London?" means nothing without the turns before it, and embedding it
// alone would match every other follow-up question ever asked.
func PromptText(req *provider.ChatRequest) string {
	var b strings.Builder
	for i, m := range req.Messages {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString(m.Role)
		b.WriteString(": ")
		b.WriteString(normalizeContent(m.Content))
	}
	return b.String()
}
