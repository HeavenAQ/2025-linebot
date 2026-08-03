package gpt

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
)

func TestBuildSummaryPromptIncludesRecentScores(t *testing.T) {
	t.Parallel()

	prompt := buildSummaryPrompt("學生說手肘太低。", []commons.SkillScore{
		{
			Date:        "2026-08-01-10-30",
			TotalGrade:  82.5,
			ScoreStatus: "進步",
			Details: []commons.GradingDetail{
				{Description: "引拍高度", Grade: 12, Maximum: 20},
			},
		},
		{Date: "2026-07-25-09-00", TotalGrade: 70},
	})

	require.Contains(t, prompt, "2026-08-01-10-30: total 82.5 (進步)")
	require.Contains(t, prompt, "引拍高度: 12.0/20.0")
	require.Contains(t, prompt, "2026-07-25-09-00: total 70.0")
	require.Contains(t, prompt, "學生說手肘太低。")
	require.Contains(t, prompt, "trust the scores")
}

func TestBuildSummaryPromptOmitsEmptySections(t *testing.T) {
	t.Parallel()

	scoresOnly := buildSummaryPrompt("  ", []commons.SkillScore{{Date: "2026-08-01-10-30", TotalGrade: 91}})
	require.NotContains(t, scoresOnly, "[Conversation]")
	require.Contains(t, scoresOnly, "[Recent scores, newest first]")

	chatOnly := buildSummaryPrompt("只有對話。", nil)
	require.NotContains(t, chatOnly, "[Recent scores")
	require.Contains(t, chatOnly, "只有對話。")
}

// A blank status must not render as an empty pair of parentheses.
func TestBuildSummaryPromptSkipsBlankScoreStatus(t *testing.T) {
	t.Parallel()

	prompt := buildSummaryPrompt("", []commons.SkillScore{{Date: "2026-08-01-10-30", TotalGrade: 88, ScoreStatus: " "}})
	require.NotContains(t, prompt, "()")
}
