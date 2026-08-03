package db

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
)

func work(total float64, status string) Work {
	return Work{
		GradingOutcome: commons.GradingOutcome{
			TotalGrade:  total,
			ScoreStatus: status,
			GradingDetails: []commons.GradingDetail{
				{Description: "引拍高度", Grade: total / 5, Maximum: 20},
			},
		},
	}
}

func TestRecentSkillScoresOrdersNewestFirstAndCaps(t *testing.T) {
	t.Parallel()

	scores := recentSkillScores(map[string]Work{
		"2026-07-25-09-00": work(70, "待加強"),
		"2026-08-01-10-30": work(82.5, "進步"),
		"2026-07-30-18-05": work(78, ""),
	}, 2)

	require.Len(t, scores, 2)
	require.Equal(t, "2026-08-01-10-30", scores[0].Date)
	require.Equal(t, 82.5, scores[0].TotalGrade)
	require.Equal(t, "進步", scores[0].ScoreStatus)
	require.Equal(t, "2026-07-30-18-05", scores[1].Date)
	require.Len(t, scores[0].Details, 1)
}

func TestRecentSkillScoresHandlesEmptyAndUnlimited(t *testing.T) {
	t.Parallel()

	require.Empty(t, recentSkillScores(nil, 5))

	all := recentSkillScores(map[string]Work{
		"2026-07-25-09-00": work(70, ""),
		"2026-08-01-10-30": work(82.5, ""),
	}, 0)
	require.Len(t, all, 2)
}

// Legacy keys that predate the current layout must not drop the whole lookup.
func TestRecentSkillScoresKeepsUnparsableKeysLast(t *testing.T) {
	t.Parallel()

	scores := recentSkillScores(map[string]Work{
		"legacy-key":       work(60, ""),
		"2026-08-01-10-30": work(82.5, ""),
	}, 0)

	require.Len(t, scores, 2)
	require.Equal(t, "2026-08-01-10-30", scores[0].Date)
	require.Equal(t, "legacy-key", scores[1].Date)
}

func TestScoreFingerprintTracksNewAttempts(t *testing.T) {
	t.Parallel()

	before := ScoreFingerprint([]commons.SkillScore{{Date: "2026-07-25-09-00", TotalGrade: 70}})
	after := ScoreFingerprint([]commons.SkillScore{
		{Date: "2026-08-01-10-30", TotalGrade: 82.5},
		{Date: "2026-07-25-09-00", TotalGrade: 70},
	})
	regraded := ScoreFingerprint([]commons.SkillScore{{Date: "2026-07-25-09-00", TotalGrade: 74}})

	require.NotEqual(t, before, after)
	require.NotEqual(t, before, regraded)
	require.Empty(t, ScoreFingerprint(nil))
}
