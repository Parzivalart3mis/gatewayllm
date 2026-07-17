package provider

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Mock is an in-process Provider for tests and local development. It needs no
// API key and no network, so the full gateway stack can be exercised offline.
//
// Behaviour is programmable: set FailWith to make every call fail, or FailFirst
// to fail a fixed number of calls before succeeding — enough to drive retry,
// failover, and circuit-breaker paths deterministically.
type Mock struct {
	// ProviderName is the identity reported to the router and metrics.
	ProviderName string
	// Reply is the completion returned. When empty, a deterministic echo of the
	// last user message is returned instead.
	Reply string
	// Latency is slept before responding, for exercising timeouts.
	Latency time.Duration
	// FailWith, when non-nil, is returned by every call.
	FailWith error
	// FailFirst fails this many calls before succeeding. Decremented per call.
	FailFirst int64
	// AvailableModels is reported by Models().
	AvailableModels []string
	// VectorSize is the dimensionality of embeddings produced by Embed.
	VectorSize int

	calls atomic.Int64
	mu    sync.Mutex
}

// NewMock builds a Mock that succeeds with the given reply.
func NewMock(name, reply string) *Mock {
	return &Mock{ProviderName: name, Reply: reply, VectorSize: 384}
}

// Calls reports how many times the provider has been invoked. Tests assert on
// this to prove a cache hit skipped the provider entirely.
func (m *Mock) Calls() int { return int(m.calls.Load()) }

func (m *Mock) Name() string {
	if m.ProviderName == "" {
		return "mock"
	}
	return m.ProviderName
}

func (m *Mock) Models() []string { return m.AvailableModels }

// next records a call and reports the error it should fail with, if any.
func (m *Mock) next(ctx context.Context) error {
	m.calls.Add(1)
	if m.Latency > 0 {
		select {
		case <-time.After(m.Latency):
		case <-ctx.Done():
			return Wrap(KindTimeout, m.Name(), ctx.Err(), "mock timed out")
		}
	}
	if err := ctx.Err(); err != nil {
		return Wrap(KindTimeout, m.Name(), err, "context expired")
	}
	if m.FailWith != nil {
		return m.FailWith
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.FailFirst > 0 {
		m.FailFirst--
		return Errorf(KindUnavailable, m.Name(), "mock transient failure")
	}
	return nil
}

// reply returns the configured reply, or a deterministic echo.
func (m *Mock) reply(req *ChatRequest) string {
	if m.Reply != "" {
		return m.Reply
	}
	var last string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == RoleUser {
			last = req.Messages[i].Content
			break
		}
	}
	return "mock reply to: " + last
}

func (m *Mock) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	if err := m.next(ctx); err != nil {
		return nil, err
	}
	text := m.reply(req)
	return &ChatResponse{
		ID:      "chatcmpl-mock-" + NewID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: RoleAssistant, Content: text},
			FinishReason: "stop",
		}},
		Usage: mockUsage(req, text),
	}, nil
}

func (m *Mock) ChatStream(ctx context.Context, req *ChatRequest, onChunk func(*ChatChunk) error) error {
	if err := m.next(ctx); err != nil {
		return err
	}
	text := m.reply(req)
	id := "chatcmpl-mock-" + NewID()
	created := time.Now().Unix()

	words := strings.Fields(text)
	for i, w := range words {
		if err := ctx.Err(); err != nil {
			return Wrap(KindTimeout, m.Name(), err, "context expired mid-stream")
		}
		delta := StreamDelta{Content: w}
		if i > 0 {
			delta.Content = " " + w
		} else {
			delta.Role = RoleAssistant
		}
		if err := onChunk(&ChatChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
			Choices: []StreamChoice{{Index: 0, Delta: delta}},
		}); err != nil {
			return err
		}
	}
	stop := "stop"
	u := mockUsage(req, text)
	return onChunk(&ChatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &stop}},
		Usage:   &u,
	})
}

// Embed returns a deterministic pseudo-embedding derived from the input text, so
// identical text always embeds identically and different text does not. Good
// enough to exercise the semantic cache's plumbing without a real model.
func (m *Mock) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if err := m.next(ctx); err != nil {
		return nil, err
	}
	size := m.VectorSize
	if size <= 0 {
		size = 384
	}
	out := &EmbeddingResponse{Object: "list", Model: req.Model}
	for i, in := range req.Input {
		out.Data = append(out.Data, EmbeddingData{
			Object: "embedding", Index: i, Embedding: DeterministicVector(in, size),
		})
	}
	return out, nil
}

// DeterministicVector hashes text into a unit-length vector. Exported so cache
// tests can construct the exact vector a given prompt will produce.
func DeterministicVector(text string, size int) []float32 {
	v := make([]float32, size)
	var sum float64
	for i := range v {
		h := fnv.New32a()
		fmt.Fprintf(h, "%s|%d", text, i)
		// Map the hash into [-1,1) so vectors spread across the space.
		f := float32(h.Sum32()%2000)/1000 - 1
		v[i] = f
		sum += float64(f) * float64(f)
	}
	// Normalize: cosine similarity assumes unit vectors.
	norm := float32(1)
	if sum > 0 {
		norm = float32(1 / math.Sqrt(sum))
	}
	for i := range v {
		v[i] *= norm
	}
	return v
}

// mockUsage approximates token counts at ~4 characters per token, which is close
// enough for tests that assert the meter recorded something proportional.
func mockUsage(req *ChatRequest, reply string) Usage {
	var in int
	for _, m := range req.Messages {
		in += len(m.Content) / 4
	}
	out := len(reply) / 4
	return Usage{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out}
}
