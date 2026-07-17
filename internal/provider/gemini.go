package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// gemini adapts Google's Generative Language API. Unlike OpenAI and Groq, Gemini
// has its own request/response shape, role vocabulary, and streaming envelope,
// so this adapter translates in both directions. It exists to prove the Provider
// interface holds for a backend that shares nothing with the OpenAI wire format.
type gemini struct {
	name    string
	baseURL string
	apiKey  string
	models  []string
	timeout time.Duration
	http    *http.Client
}

// NewGemini builds an adapter for the Google Generative Language API.
func NewGemini(o Options) Provider {
	if o.BaseURL == "" {
		o.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	return &gemini{
		name:    orDefault(o.Name, "gemini"),
		baseURL: strings.TrimRight(o.BaseURL, "/"),
		apiKey:  o.APIKey,
		models:  o.Models,
		timeout: o.Timeout,
		http:    newHTTPClient(),
	}
}

func (p *gemini) Name() string     { return p.name }
func (p *gemini) Models() []string { return p.models }

func (p *gemini) withTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if p.timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, p.timeout)
}

// --- Gemini wire types ---

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
	CandidateCount  int      `json:"candidateCount,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent  `json:"contents"`
	SystemInstruction *geminiContent   `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenConfig `json:"generationConfig,omitempty"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiResponse struct {
	Candidates     []geminiCandidate `json:"candidates"`
	UsageMetadata  geminiUsage       `json:"usageMetadata"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

type geminiErrorEnvelope struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

// --- translation ---

// toGemini converts an OpenAI-shaped request into Gemini's format.
//
// Three mismatches need handling:
//   - Gemini names the assistant role "model" and has no "system" role; system
//     messages move to the dedicated systemInstruction field.
//   - Gemini requires strictly alternating user/model turns, so consecutive
//     same-role messages are merged.
//   - max_tokens is named maxOutputTokens.
func toGemini(req *ChatRequest) *geminiRequest {
	out := &geminiRequest{
		GenerationConfig: &geminiGenConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
			StopSequences:   req.Stop,
		},
	}

	var systemParts []string
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			systemParts = append(systemParts, m.Content)
			continue
		}
		role := "user"
		if m.Role == RoleAssistant {
			role = "model"
		}
		// Merge into the previous turn when the role repeats.
		if n := len(out.Contents); n > 0 && out.Contents[n-1].Role == role {
			prev := &out.Contents[n-1]
			prev.Parts = append(prev.Parts, geminiPart{Text: m.Content})
			continue
		}
		out.Contents = append(out.Contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: m.Content}},
		})
	}
	if len(systemParts) > 0 {
		out.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: strings.Join(systemParts, "\n\n")}},
		}
	}
	return out
}

// finishReason maps Gemini's vocabulary onto OpenAI's.
func finishReason(g string) string {
	switch g {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT":
		return "content_filter"
	case "":
		return ""
	default:
		return strings.ToLower(g)
	}
}

func (r *geminiResponse) text() string {
	if len(r.Candidates) == 0 {
		return ""
	}
	var b strings.Builder
	for _, p := range r.Candidates[0].Content.Parts {
		b.WriteString(p.Text)
	}
	return b.String()
}

func (u geminiUsage) toUsage() Usage {
	return Usage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.CandidatesTokenCount,
		TotalTokens:      u.TotalTokenCount,
	}
}

// --- Provider implementation ---

// newRequest builds a Gemini call. The API key travels in the x-goog-api-key
// header rather than a query parameter so it cannot leak into access logs.
func (p *gemini) newRequest(ctx context.Context, model, method, query string, body any) (*http.Request, error) {
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, Wrap(KindInternal, p.name, err, "marshal request")
	}
	url := fmt.Sprintf("%s/models/%s:%s", p.baseURL, model, method)
	if query != "" {
		url += "?" + query
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return nil, Wrap(KindInternal, p.name, err, "build request")
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("x-goog-api-key", p.apiKey)
	return r, nil
}

func (p *gemini) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	ctx, cancel := p.withTimeout(ctx)
	defer cancel()

	model := upstreamModel(req)
	httpReq, err := p.newRequest(ctx, model, "generateContent", "", toGemini(req))
	if err != nil {
		return nil, err
	}
	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, p.errorFromResponse(resp)
	}
	var gr geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&gr); err != nil {
		return nil, Wrap(KindUnavailable, p.name, err, "decode response")
	}
	// A prompt blocked before generation returns 200 with no candidates and a
	// block reason, which must surface as a content filter rather than an
	// empty completion.
	if gr.PromptFeedback != nil && gr.PromptFeedback.BlockReason != "" {
		return nil, Errorf(KindContentFilter, p.name, "prompt blocked: %s", gr.PromptFeedback.BlockReason)
	}
	if len(gr.Candidates) == 0 {
		return nil, Errorf(KindUnavailable, p.name, "upstream returned no candidates")
	}

	return &ChatResponse{
		ID:      "chatcmpl-" + NewID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{{
			Index:        0,
			Message:      Message{Role: RoleAssistant, Content: gr.text()},
			FinishReason: finishReason(gr.Candidates[0].FinishReason),
		}},
		Usage: gr.UsageMetadata.toUsage(),
	}, nil
}

func (p *gemini) ChatStream(ctx context.Context, req *ChatRequest, onChunk func(*ChatChunk) error) error {
	// alt=sse switches the response from a JSON array to an event stream, which
	// is what makes incremental delivery possible.
	httpReq, err := p.newRequest(ctx, upstreamModel(req), "streamGenerateContent", "alt=sse", toGemini(req))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := p.http.Do(httpReq)
	if err != nil {
		return p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return p.errorFromResponse(resp)
	}

	id := "chatcmpl-" + NewID()
	created := time.Now().Unix()
	first := true

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		payload, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		var gr geminiResponse
		if err := json.Unmarshal([]byte(strings.TrimSpace(payload)), &gr); err != nil {
			return Wrap(KindUnavailable, p.name, err, "decode stream chunk")
		}
		if gr.PromptFeedback != nil && gr.PromptFeedback.BlockReason != "" {
			return Errorf(KindContentFilter, p.name, "prompt blocked: %s", gr.PromptFeedback.BlockReason)
		}
		if len(gr.Candidates) == 0 {
			continue
		}

		delta := StreamDelta{Content: gr.text()}
		if first {
			// OpenAI clients expect the role announced on the first delta only.
			delta.Role = RoleAssistant
			first = false
		}
		var fr *string
		if r := finishReason(gr.Candidates[0].FinishReason); r != "" {
			fr = &r
		}
		chunk := &ChatChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   req.Model,
			Choices: []StreamChoice{{Index: 0, Delta: delta, FinishReason: fr}},
		}
		// Gemini reports cumulative usage on every chunk; only the last one is
		// meaningful, and it coincides with the finish reason.
		if fr != nil && gr.UsageMetadata.TotalTokenCount > 0 {
			u := gr.UsageMetadata.toUsage()
			chunk.Usage = &u
		}
		if err := onChunk(chunk); err != nil {
			return err
		}
	}
	if err := sc.Err(); err != nil {
		return p.transportError(ctx, err)
	}
	return nil
}

type geminiEmbedRequest struct {
	Model   string        `json:"model"`
	Content geminiContent `json:"content"`
}

type geminiBatchEmbedRequest struct {
	Requests []geminiEmbedRequest `json:"requests"`
}

type geminiBatchEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

func (p *gemini) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	ctx, cancel := p.withTimeout(ctx)
	defer cancel()

	model := req.Model
	// Gemini requires the fully-qualified resource name inside the body.
	qualified := model
	if !strings.HasPrefix(qualified, "models/") {
		qualified = "models/" + qualified
	}

	body := geminiBatchEmbedRequest{}
	for _, in := range req.Input {
		body.Requests = append(body.Requests, geminiEmbedRequest{
			Model:   qualified,
			Content: geminiContent{Parts: []geminiPart{{Text: in}}},
		})
	}

	httpReq, err := p.newRequest(ctx, model, "batchEmbedContents", "", body)
	if err != nil {
		return nil, err
	}
	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, p.transportError(ctx, err)
	}
	defer drainClose(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, p.errorFromResponse(resp)
	}
	var br geminiBatchEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&br); err != nil {
		return nil, Wrap(KindUnavailable, p.name, err, "decode embeddings response")
	}

	out := &EmbeddingResponse{Object: "list", Model: req.Model}
	for i, e := range br.Embeddings {
		out.Data = append(out.Data, EmbeddingData{Object: "embedding", Index: i, Embedding: e.Values})
	}
	return out, nil
}

// errorFromResponse classifies a Gemini error. Gemini uses canonical Google
// status strings, which are more precise than the HTTP status alone.
func (p *gemini) errorFromResponse(resp *http.Response) *Error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024))
	kind := ClassifyStatus(resp.StatusCode)
	msg := strings.TrimSpace(string(body))

	var ge geminiErrorEnvelope
	if err := json.Unmarshal(body, &ge); err == nil && ge.Error.Message != "" {
		msg = ge.Error.Message
		switch ge.Error.Status {
		case "RESOURCE_EXHAUSTED":
			kind = KindRateLimit
		case "UNAUTHENTICATED", "PERMISSION_DENIED":
			kind = KindAuth
		case "UNAVAILABLE", "INTERNAL":
			kind = KindUnavailable
		case "DEADLINE_EXCEEDED":
			kind = KindTimeout
		case "INVALID_ARGUMENT":
			// Gemini reports an oversized prompt as a generic INVALID_ARGUMENT;
			// only the message distinguishes it from a malformed request, and
			// the difference decides whether failover is worth attempting.
			if strings.Contains(strings.ToLower(msg), "token") {
				kind = KindContextLength
			} else {
				kind = KindInvalidRequest
			}
		}
	}
	if msg == "" {
		msg = http.StatusText(resp.StatusCode)
	}
	return &Error{
		Kind:       kind,
		Provider:   p.name,
		Status:     resp.StatusCode,
		Message:    truncate(msg, 512),
		RetryAfter: ParseRetryAfter(resp.Header),
	}
}

func (p *gemini) transportError(ctx context.Context, err error) *Error {
	switch {
	case ctx.Err() == context.Canceled:
		return Wrap(KindInternal, p.name, err, "request canceled")
	case ctx.Err() == context.DeadlineExceeded:
		return Wrap(KindTimeout, p.name, err, "request timed out")
	default:
		return Wrap(KindUnavailable, p.name, err, "transport error")
	}
}
