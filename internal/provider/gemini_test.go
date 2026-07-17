package provider

import (
	"testing"
)

// TestToGemini_SystemInstruction asserts system messages move to Gemini's
// dedicated field: Gemini has no system role, and passing one through as a
// normal turn would change the model's behaviour.
func TestToGemini_SystemInstruction(t *testing.T) {
	req := &ChatRequest{
		Model: "gemini-2.0-flash",
		Messages: []Message{
			{Role: RoleSystem, Content: "You are terse."},
			{Role: RoleUser, Content: "Hello"},
		},
	}
	got := toGemini(req)

	if got.SystemInstruction == nil {
		t.Fatal("system message must become systemInstruction")
	}
	if got.SystemInstruction.Parts[0].Text != "You are terse." {
		t.Errorf("systemInstruction = %q", got.SystemInstruction.Parts[0].Text)
	}
	if len(got.Contents) != 1 {
		t.Fatalf("contents = %d, want 1: the system message must not remain a turn", len(got.Contents))
	}
	if got.Contents[0].Role != "user" {
		t.Errorf("role = %q, want user", got.Contents[0].Role)
	}
}

// TestToGemini_MultipleSystemMessages asserts several system messages merge
// rather than the later ones being dropped.
func TestToGemini_MultipleSystemMessages(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleSystem, Content: "Rule one."},
			{Role: RoleSystem, Content: "Rule two."},
			{Role: RoleUser, Content: "Hi"},
		},
	}
	got := toGemini(req)

	want := "Rule one.\n\nRule two."
	if got.SystemInstruction.Parts[0].Text != want {
		t.Errorf("systemInstruction = %q, want %q", got.SystemInstruction.Parts[0].Text, want)
	}
}

// TestToGemini_AssistantBecomesModel asserts the role rename.
func TestToGemini_AssistantBecomesModel(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "Hi"},
			{Role: RoleAssistant, Content: "Hello"},
			{Role: RoleUser, Content: "Bye"},
		},
	}
	got := toGemini(req)

	wantRoles := []string{"user", "model", "user"}
	if len(got.Contents) != len(wantRoles) {
		t.Fatalf("contents = %d, want %d", len(got.Contents), len(wantRoles))
	}
	for i, w := range wantRoles {
		if got.Contents[i].Role != w {
			t.Errorf("contents[%d].role = %q, want %q", i, got.Contents[i].Role, w)
		}
	}
}

// TestToGemini_MergesConsecutiveRoles asserts same-role runs are merged. Gemini
// rejects non-alternating turns, so passing them through would be a hard error.
func TestToGemini_MergesConsecutiveRoles(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{
			{Role: RoleUser, Content: "First"},
			{Role: RoleUser, Content: "Second"},
			{Role: RoleAssistant, Content: "Reply"},
		},
	}
	got := toGemini(req)

	if len(got.Contents) != 2 {
		t.Fatalf("contents = %d, want 2 (two user messages merged into one turn)", len(got.Contents))
	}
	if len(got.Contents[0].Parts) != 2 {
		t.Fatalf("parts = %d, want 2: both user messages must survive the merge", len(got.Contents[0].Parts))
	}
	if got.Contents[0].Parts[0].Text != "First" || got.Contents[0].Parts[1].Text != "Second" {
		t.Error("merged parts must preserve content and order")
	}
}

// TestToGemini_GenerationConfig asserts parameter renaming.
func TestToGemini_GenerationConfig(t *testing.T) {
	temp := 0.5
	maxTok := 256
	req := &ChatRequest{
		Messages:    []Message{{Role: RoleUser, Content: "Hi"}},
		Temperature: &temp,
		MaxTokens:   &maxTok,
		Stop:        []string{"END"},
	}
	got := toGemini(req)

	if got.GenerationConfig.Temperature == nil || *got.GenerationConfig.Temperature != 0.5 {
		t.Error("temperature must be forwarded")
	}
	// max_tokens -> maxOutputTokens is the rename most likely to be missed.
	if got.GenerationConfig.MaxOutputTokens == nil || *got.GenerationConfig.MaxOutputTokens != 256 {
		t.Error("max_tokens must map to maxOutputTokens")
	}
	if len(got.GenerationConfig.StopSequences) != 1 || got.GenerationConfig.StopSequences[0] != "END" {
		t.Error("stop must map to stopSequences")
	}
}

// TestFinishReason asserts Gemini's vocabulary maps onto OpenAI's, since clients
// branch on these exact strings.
func TestFinishReason(t *testing.T) {
	cases := map[string]string{
		"STOP":               "stop",
		"MAX_TOKENS":         "length",
		"SAFETY":             "content_filter",
		"PROHIBITED_CONTENT": "content_filter",
		"":                   "",
	}
	for in, want := range cases {
		if got := finishReason(in); got != want {
			t.Errorf("finishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestGeminiUsage_ToUsage asserts token counts map onto the OpenAI shape.
func TestGeminiUsage_ToUsage(t *testing.T) {
	u := geminiUsage{PromptTokenCount: 10, CandidatesTokenCount: 20, TotalTokenCount: 30}
	got := u.toUsage()

	if got.PromptTokens != 10 || got.CompletionTokens != 20 || got.TotalTokens != 30 {
		t.Errorf("toUsage() = %+v, want {10 20 30}", got)
	}
}
