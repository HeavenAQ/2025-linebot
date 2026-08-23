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

func TestBuildWeeklyPreviewPromptGroundsTheFocusInEverySkill(t *testing.T) {
	t.Parallel()

	prompt := buildWeeklyPreviewPrompt("小明", "殺球", []commons.SkillHistory{
		{
			Skill: "serve",
			Scores: []commons.SkillScore{
				{Date: "2026-08-01-10-00", TotalGrade: 88},
			},
		},
		{
			Skill: "smash",
			Scores: []commons.SkillScore{
				{Date: "2026-08-01-11-00", TotalGrade: 61, Details: []commons.GradingDetail{
					{Description: "手腕發力", Grade: 8, Maximum: 20},
				}},
			},
		},
	})

	require.Contains(t, prompt, "本週要加強的動作：殺球")
	require.Contains(t, prompt, "學生：小明")
	// Both skills are listed: the reason for focusing on one is that the other
	// is going better.
	require.Contains(t, prompt, "[serve]")
	require.Contains(t, prompt, "[smash]")
	require.Contains(t, prompt, "手腕發力: 8.0/20.0")
	require.Contains(t, prompt, "不要杜撰")
}

func TestWeeklyPreviewRefusesWithoutHistoryOrFocus(t *testing.T) {
	t.Parallel()

	client := &Client{}
	_, err := client.WeeklyPreview("小明", "殺球", nil)
	require.Error(t, err)

	_, err = client.WeeklyPreview("小明", "  ", []commons.SkillHistory{{Skill: "serve"}})
	require.Error(t, err)
}
