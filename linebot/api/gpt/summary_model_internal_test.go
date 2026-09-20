package gpt

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/stretchr/testify/require"
)

// A stored OpenAI prompt pins its model, and when that one was retired the
// summary failed in production with nothing here naming the model or the
// prompt. Sending both on every request is what makes that fixable.
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
	// The stored prompt this replaced answered "how am I doing" with "我無法
	// 針對動作學習進程或技術表現進行分析與評估。建議諮詢專業教練" -- the one
	// answer a coaching bot holding the learner's scores must never give.
	require.Contains(t, coachInstruction, "評估與給建議")
	require.Contains(t, coachInstruction, "絕對不要說自己無法分析或評估")
	require.Contains(t, coachInstruction, "你就是他的教練")
}

// A learner asking how they are doing is answered from their grades, so the
// grades have to reach the model with the question.
func TestCoachingCarriesTheLearnersScores(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(raw, &body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"回覆"}]}]}`))
	}))
	defer server.Close()

	ctx := context.Background()
	openaiClient := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	client := &Client{Ctx: &ctx, Client: &openaiClient, Model: DefaultModel}

	reply, err := client.Coach(nil, "目前學習進程如何", "發球", []commons.SkillScore{
		{Date: "2026-08-03", TotalGrade: 33.7, Details: []commons.GradingDetail{
			{Description: "雙手平舉", Grade: 2.4, Maximum: 20},
		}},
	})

	require.NoError(t, err)
	require.Equal(t, "回覆", reply)
	input := lastMessage(t, body)
	require.Contains(t, input, "雙手平舉: 2.4/20.0")
	require.Contains(t, input, "total 33.7")
	require.Contains(t, input, "目前學習進程如何", "the question still has to reach the model")
	require.Equal(t, DefaultModel, body["model"])
	require.Equal(t, coachInstruction, body["instructions"])
}

// With no scores on record the question must still go through unchanged, rather
// than carrying an empty header the model has to reason around.
func TestCoachingWithoutScoresSendsOnlyTheQuestion(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(raw, &body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"回覆"}]}]}`))
	}))
	defer server.Close()

	ctx := context.Background()
	openaiClient := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	client := &Client{Ctx: &ctx, Client: &openaiClient, Model: DefaultModel}

	_, err := client.Coach(nil, "怎麼練發球", "發球", nil)

	require.NoError(t, err)
	// The stroke has to be named even with no scores to carry it: a first
	// question has no earlier turn saying which stroke it is, and the coach
	// answered about the wrong one.
	input := lastMessage(t, body)
	require.Contains(t, input, "發球")
	require.Contains(t, input, "怎麼練發球")
}

// The learner's stored turns are the whole conversation now, so they have to
// travel with the question, in order and with their roles intact.
func TestCoachingSendsTheStoredHistory(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(raw, &body))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"回覆"}]}]}`))
	}))
	defer server.Close()

	ctx := context.Background()
	openaiClient := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(server.URL))
	client := &Client{Ctx: &ctx, Client: &openaiClient, Model: DefaultModel}

	history := make([]HistoryMessage, 0, historyWindow+4)
	for i := 0; i < historyWindow+2; i++ {
		history = append(history, HistoryMessage{Role: "user", Text: fmt.Sprintf("舊問題 %d", i)})
	}
	history = append(history, HistoryMessage{Role: "assistant", Text: "上一則回覆"})

	_, err := client.Coach(history, "那手腕呢", "發球", nil)
	require.NoError(t, err)

	items, ok := body["input"].([]any)
	require.True(t, ok, "history travels as input items, not one string")
	require.Len(t, items, historyWindow+1, "only the most recent turns are sent, plus the new question")

	first := items[0].(map[string]any)
	require.Equal(t, "user", first["role"])
	require.Equal(t, "舊問題 3", first["content"], "the oldest turns are dropped, not the newest")

	previous := items[historyWindow-1].(map[string]any)
	require.Equal(t, "assistant", previous["role"])
	require.Equal(t, "上一則回覆", previous["content"])

	require.Contains(t, lastMessage(t, body), "那手腕呢")

	// Storing the turns at OpenAI too would be a second copy of Firestore,
	// and its conversation ids die with the API key that made them.
	require.Equal(t, false, body["store"])
	require.NotContains(t, body, "conversation")
}

// lastMessage returns the text of the final input item, which is the question
// the learner just asked.
func lastMessage(t *testing.T, body map[string]any) string {
	t.Helper()
	items, ok := body["input"].([]any)
	require.True(t, ok, "input should be a list of messages")
	require.NotEmpty(t, items)
	last, ok := items[len(items)-1].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", last["role"])
	text, ok := last["content"].(string)
	require.True(t, ok)
	return text
}
