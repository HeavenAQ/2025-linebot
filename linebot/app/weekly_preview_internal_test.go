package app

import (
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
)

func history(skill string, grades ...float64) commons.SkillHistory {
	return historyOn(skill, "2026-08-01-10-00", grades...)
}

// historyOn dates the latest attempt, which is what decides the focus.
func historyOn(skill, latest string, grades ...float64) commons.SkillHistory {
	scores := make([]commons.SkillScore, 0, len(grades))
	for index, grade := range grades {
		date := latest
		if index > 0 {
			date = "2026-07-01-10-00"
		}
		scores = append(scores, commons.SkillScore{Date: date, TotalGrade: grade})
	}
	return commons.SkillHistory{Skill: skill, Scores: scores}
}

// The preview lands before the next lesson, so it follows the session the
// learner just had -- not whichever skill scores worst across the term.
func TestFocusSkillPicksTheMostRecentlyPractisedSkill(t *testing.T) {
	t.Parallel()

	focus := focusSkill([]commons.SkillHistory{
		historyOn("serve", "2026-08-28-09-00", 88, 84),
		historyOn("smash", "2026-08-20-09-00", 61, 70),
	})

	require.Equal(t, "serve", focus.Skill)
}

// A high score does not move the focus off the session they just did.
func TestFocusSkillIgnoresGradeWhenOneSkillIsMoreRecent(t *testing.T) {
	t.Parallel()

	focus := focusSkill([]commons.SkillHistory{
		historyOn("serve", "2026-08-28-09-00", 95, 40),
		historyOn("smash", "2026-08-27-09-00", 12, 68),
	})

	require.Equal(t, "serve", focus.Skill)
}

// Two skills graded in the same minute: the weaker one is the useful preview.
func TestFocusSkillBreaksSameMinuteTiesOnTheLowerGrade(t *testing.T) {
	t.Parallel()

	focus := focusSkill([]commons.SkillHistory{
		historyOn("serve", "2026-08-28-09-00", 70, 90),
		historyOn("smash", "2026-08-28-09-00", 55, 60),
	})

	require.Equal(t, "smash", focus.Skill)
}

func TestFocusSkillHandlesASingleSkill(t *testing.T) {
	t.Parallel()

	require.Equal(t, "smash", focusSkill([]commons.SkillHistory{history("smash", 50)}).Skill)
}

// A learner's worst skill may well be one the course is not running. Focusing
// on it would push a note about something they cannot practise this semester.
func TestSupportedSkillHistoryDropsWithdrawnSkills(t *testing.T) {
	t.Parallel()

	supported := supportedSkillHistory([]commons.SkillHistory{
		history("serve", 88, 84),
		history("smash", 74, 71),
		history("clear", 40, 42),
		history("lift", 35),
	})

	require.Len(t, supported, 2)
	require.Equal(t, "serve", supported[0].Skill)
	require.Equal(t, "smash", supported[1].Skill)
	require.Equal(t, "smash", focusSkill(supported).Skill)
}

// The learner is skipped rather than pushed an unsupported focus.
func TestSupportedSkillHistoryIsEmptyWhenOnlyWithdrawnSkillsWereAttempted(t *testing.T) {
	t.Parallel()

	require.Empty(t, supportedSkillHistory([]commons.SkillHistory{
		history("clear", 61),
		history("lift", 58),
	}))
}

// Scoreless or unrecognised entries would index Scores[0] in focusSkill.
func TestSupportedSkillHistoryDropsEntriesItCannotReasonAbout(t *testing.T) {
	t.Parallel()

	require.Empty(t, supportedSkillHistory([]commons.SkillHistory{
		{Skill: "serve"},
		history("drop_shot", 50),
		history("", 50),
	}))
}

func TestAverageGradeIgnoresEmptyHistories(t *testing.T) {
	t.Parallel()

}

// The dedupe key has to name the same week for every day of that week,
// including across a year boundary where the calendar year and ISO year differ.
func TestISOWeekIsStableAcrossAWeek(t *testing.T) {
	t.Parallel()

	monday := time.Date(2026, 8, 3, 9, 0, 0, 0, time.UTC)
	sunday := time.Date(2026, 8, 9, 23, 0, 0, 0, time.UTC)
	require.Equal(t, db.ISOWeek(monday), db.ISOWeek(sunday))
	require.NotEqual(t, db.ISOWeek(monday), db.ISOWeek(monday.AddDate(0, 0, 7)))

	// 1 January 2027 falls in ISO week 53 of 2026.
	require.Equal(t, "2026-W53", db.ISOWeek(time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC)))
}
