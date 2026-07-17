package provider

import (
	"context"
	"strings"
)

// NewGroq builds an adapter for Groq. Groq serves the OpenAI wire protocol, so
// this reuses the compatible transport rather than restating it; only the
// default base URL and identity differ.
func NewGroq(o Options) Provider {
	if o.BaseURL == "" {
		o.BaseURL = "https://api.groq.com/openai/v1"
	}
	return &groq{openAICompat{
		name:    orDefault(o.Name, "groq"),
		baseURL: strings.TrimRight(o.BaseURL, "/"),
		apiKey:  o.APIKey,
		models:  o.Models,
		timeout: o.Timeout,
		http:    newHTTPClient(),
	}}
}

// groq overrides only where Groq diverges from OpenAI.
type groq struct{ openAICompat }

// Embed reports unsupported: Groq serves no embeddings API, and letting the
// generic call through would surface a 404 as a confusing upstream error.
func (g *groq) Embed(context.Context, *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, ErrUnsupported
}
