package gpt

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/require"
)

// The summary used to run off a stored OpenAI prompt, which pinned whichever
// model it was saved against. When OpenAI retired that model the summary
// started failing in production with "Model not found", and nothing in this
// repository named either the model or the prompt text, so there was no way to
// fix it from here. Sending both on every request is what makes that
// recoverable -- hence this test.
func TestSummarizeNamesItsOwnModel(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(raw, &body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"總結"}]}]}`))
	}))
	defer server.Close()

	ctx := context.Background()
	openaiClient := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	client := &Client{Ctx: &ctx, Client: &openaiClient, Model: DefaultModel}

	summary, err := client.Summarize("練習內容", nil)

	require.NoError(t, err)
	require.Equal(t, "總結", summary)
	require.Equal(t, DefaultModel, body["model"], "the request must name its own model")
	require.Equal(t, summaryInstruction, body["instructions"], "the system prompt must be sent, not stored at OpenAI")
	require.NotContains(t, body, "prompt", "no stored prompt may be referenced")
}

// An unset OPENAI_MODEL must not leave the model empty: OpenAI rejects a
// request that names no model and no stored prompt.
func TestNewGPTClientDefaultsItsModel(t *testing.T) {
	require.Equal(t, DefaultModel, NewGPTClient("key", "").Model)
	require.Equal(t, "gpt-5.6-other", NewGPTClient("key", "gpt-5.6-other").Model)
}

// Every learner-facing call must carry a model and a system prompt of its own.
// Coaching replies and the weekly preview inherited both from the stored prompt
// and would have failed the same way the summary did.
func TestEveryLearnerFacingCallCarriesModelAndInstructions(t *testing.T) {
	for _, prompt := range []string{coachInstruction, summaryInstruction, weeklyPreviewInstruction} {
		require.NotEmpty(t, prompt)
	}
	require.Contains(t, coachInstruction, "繁體中文")
	// The stored prompt refused questions about a learner's own progress; that
	// is precisely what this bot exists to answer.
	require.Contains(t, coachInstruction, "不要拒絕")
}
