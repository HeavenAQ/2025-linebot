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

// The summary runs off a stored prompt, and a stored prompt pins whichever
// model it was saved against. When OpenAI retired that model the summary
// started failing in production with "Model not found", and nothing in this
// repository named the model, so there was no way to fix it from here. Naming
// it on every request is what makes that recoverable -- hence this test.
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
	client := &Client{
		Ctx:          &ctx,
		Client:       &openaiClient,
		PromptID:     "pmpt_test",
		SummaryModel: DefaultModel,
	}

	summary, err := client.Summarize("練習內容", nil)

	require.NoError(t, err)
	require.Equal(t, "總結", summary)
	require.Equal(t, DefaultModel, body["model"], "the request must name a model, not inherit the stored prompt's")
	prompt, ok := body["prompt"].(map[string]any)
	require.True(t, ok, "the stored prompt is still what shapes the summary")
	require.Equal(t, "pmpt_test", prompt["id"])
}

// An unset OPENAI_SUMMARY_MODEL must not leave the model empty, which would
// hand the request straight back to the stored prompt's pinned model.
func TestNewGPTClientDefaultsBothModels(t *testing.T) {
	client := NewGPTClient("key", "pmpt_test", "", "")

	require.Equal(t, DefaultModel, client.SummaryModel)
	require.Equal(t, DefaultModel, client.RewriteModel)
}
