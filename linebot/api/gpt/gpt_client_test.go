package gpt_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/stretchr/testify/require"
)

// TestCoach verifies that a learner's question gets a reply, with the stored
// turns standing in for the conversation.
func TestCoach(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	t.Parallel()
	reply, err := gptClient.Coach(nil, "Hello, this is a test message.", "發球", nil)
	require.NoError(t, err, "Expected no error when asking the coach")
	require.NotEmpty(t, reply, "Expected the coach to reply with text")
}

// TestCoachFollowsStoredHistory checks that a follow-up question is answered
// in the context of the turns sent with it.
func TestCoachFollowsStoredHistory(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	t.Parallel()
	response, err := gptClient.Coach(
		[]gpt.HistoryMessage{
			{Role: "user", Text: "我的發球總是過高。"},
			{Role: "assistant", Text: "試著把擊球點放低一點，並放鬆手腕。"},
		},
		"那我要練什麼？", "發球", nil,
	)
	require.NoError(t, err, "Expected no error when retrieving assistant response")
	require.NotEmpty(t, response, "Expected a non-empty response from the assistant")
	t.Log("Assistant's response:", response)
}

func TestRewriteQueryUsesConversationHistory(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	rewritten, err := gptClient.RewriteQuery(
		[]gpt.HistoryMessage{
			{Role: "user", Text: "我的高遠球揮拍時手肘太低。"},
			{Role: "assistant", Text: "可以在引拍時讓手肘稍高於肩膀。"},
		},
		"那我應該怎麼改？",
	)
	require.NoError(t, err)
	require.NotEmpty(t, rewritten)
	require.Contains(t, rewritten, "高遠球")
}
