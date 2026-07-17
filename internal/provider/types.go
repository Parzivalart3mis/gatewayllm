// Package provider defines the Provider abstraction and one adapter per LLM
// backend. Adding a backend means adding one file that implements Provider.
package provider

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Role constants for chat messages.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is a single chat message in the OpenAI wire format.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatRequest mirrors the OpenAI /v1/chat/completions request body. Fields the
// gateway does not interpret are preserved in Extra so adapters can forward them.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
	User        string    `json:"user,omitempty"`
	N           *int      `json:"n,omitempty"`

	// Resolved by the router before the request reaches a provider. Not part of
	// the client-facing wire format.
	ResolvedModel string `json:"-"`
}

// Temp returns the effective temperature, defaulting to the OpenAI default of 1.
func (r *ChatRequest) Temp() float64 {
	if r.Temperature == nil {
		return 1.0
	}
	return *r.Temperature
}

// MaxTok returns the effective max_tokens, or 0 when unset.
func (r *ChatRequest) MaxTok() int {
	if r.MaxTokens == nil {
		return 0
	}
	return *r.MaxTokens
}

// Validate checks the request against the parts of the OpenAI contract the
// gateway relies on.
func (r *ChatRequest) Validate() error {
	if r.Model == "" {
		return errors.New("model is required")
	}
	if len(r.Messages) == 0 {
		return errors.New("messages must contain at least one message")
	}
	for i, m := range r.Messages {
		if m.Role == "" {
			return fmt.Errorf("messages[%d].role is required", i)
		}
	}
	if r.Temperature != nil && (*r.Temperature < 0 || *r.Temperature > 2) {
		return errors.New("temperature must be between 0 and 2")
	}
	if r.MaxTokens != nil && *r.MaxTokens < 1 {
		return errors.New("max_tokens must be positive")
	}
	if r.N != nil && *r.N != 1 {
		return errors.New("n must be 1: the gateway does not support multiple choices")
	}
	return nil
}

// Usage reports token counts for a completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Choice is one completion alternative.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse mirrors the OpenAI /v1/chat/completions response body.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Text returns the assistant content of the first choice.
func (r *ChatResponse) Text() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.Content
}

// StreamDelta is the incremental payload inside a streaming chunk.
type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// StreamChoice is one choice within a streaming chunk.
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatChunk mirrors an OpenAI streaming chunk (`chat.completion.chunk`).
type ChatChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	// Usage is populated on the final chunk by providers that report it.
	Usage *Usage `json:"usage,omitempty"`
}

// EmbeddingRequest mirrors the OpenAI /v1/embeddings request body. Input accepts
// either a string or an array of strings, matching the OpenAI contract.
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input Input    `json:"input"`
	User  string   `json:"user,omitempty"`
	Dims  *int     `json:"dimensions,omitempty"`
	_     struct{} // keep construction keyed
}

// Input holds one or more strings decoded from a polymorphic JSON field.
type Input []string

// UnmarshalJSON accepts a bare string or an array of strings.
func (in *Input) UnmarshalJSON(b []byte) error {
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*in = Input{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return errors.New("input must be a string or an array of strings")
	}
	*in = many
	return nil
}

// MarshalJSON emits a bare string for single inputs, matching what upstream
// providers expect most often.
func (in Input) MarshalJSON() ([]byte, error) {
	if len(in) == 1 {
		return json.Marshal(in[0])
	}
	return json.Marshal([]string(in))
}

// EmbeddingData is one embedding vector in an embeddings response.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// EmbeddingResponse mirrors the OpenAI /v1/embeddings response body.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}
