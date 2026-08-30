package db

import (
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Week labels arrive from the browser and become document IDs, so anything that
// is not exactly what ISOWeek produces has to be refused.
func TestValidWeekAcceptsOnlyISOWeekLabels(t *testing.T) {
	t.Parallel()

	require.True(t, ValidWeek(ISOWeek(time.Date(2026, 8, 23, 0, 0, 0, 0, time.UTC))))
	require.True(t, ValidWeek("2026-W01"))
	require.True(t, ValidWeek("2026-W53"))

	for _, invalid := range []string{
		"", "2026", "2026-W", "2026-W1", "26-W01", "2026-w01",
		"2026-W01/../other", "../../etc", "2026-W01 ", "abcd-Wxy",
	} {
		require.False(t, ValidWeek(invalid), invalid)
	}
}

func TestWeeklyReflectionDocIDIsScopedToTheLearner(t *testing.T) {
	t.Parallel()

	client := &FirestoreClient{}
	require.Equal(t, "U123_2026-W34", client.weeklyReflectionDocID("U123", "2026-W34"))
	require.NotEqual(t,
		client.weeklyReflectionDocID("U123", "2026-W34"),
		client.weeklyReflectionDocID("U124", "2026-W34"),
	)
}

// A save merges by field name, so a WeeklyNoteField that does not match the
// stored tag would quietly write a stray field nothing ever reads back.
func TestWeeklyNoteFieldsMatchTheStoredTags(t *testing.T) {
	t.Parallel()

	record := reflect.TypeOf(WeeklyReflection{})
	for name, note := range map[string]WeeklyNoteField{
		"Note":    ReflectionNote,
		"Preview": PreviewNote,
	} {
		field, ok := record.FieldByName(name)
		require.True(t, ok, name)
		require.Equal(t, string(note), field.Tag.Get("firestore"), name)
		require.Equal(t, string(note), field.Tag.Get("json"), name)
	}
}

func TestWithWeeklyReflectionNotesJoinsLiffNotesByUserAndWeek(t *testing.T) {
	t.Parallel()

	const (
		userID   = "U123"
		workDate = "2026-08-19-14-30"
		week     = "2026-W34"
	)
	user := &UserData{
		ID: userID,
		Portfolio: Portfolios{
			Serve: map[string]Work{
				workDate: {
					DateTime:   workDate,
					Reflection: "legacy per-video reflection",
				},
			},
			Smash: map[string]Work{
				"2026-08-11-14-30": {
					DateTime:   "2026-08-11-14-30",
					Reflection: "different week",
				},
			},
		},
	}
	weekly := map[string]WeeklyReflection{
		week: {
			UserID:  userID,
			Week:    week,
			Note:    "LIFF reflection",
			Preview: "LIFF preview",
		},
	}

	view := WithWeeklyReflectionNotes(user, weekly)
	require.Equal(t, "LIFF reflection", view.Portfolio.Serve[workDate].Reflection)
	require.Equal(t, "LIFF preview", view.Portfolio.Serve[workDate].Preview)
	require.Equal(t, "different week", view.Portfolio.Smash["2026-08-11-14-30"].Reflection)
	require.Empty(t, view.Portfolio.Smash["2026-08-11-14-30"].Preview)

	// Rendering is read-only: joining must not mutate the Firestore-shaped user.
	require.Equal(t, "legacy per-video reflection", user.Portfolio.Serve[workDate].Reflection)
	require.Empty(t, user.Portfolio.Serve[workDate].Preview)
}

func TestWithWeeklyReflectionNotesRejectsAnotherLearnersRecord(t *testing.T) {
	t.Parallel()

	const workDate = "2026-08-19-14-30"
	user := &UserData{
		ID: "U123",
		Portfolio: Portfolios{Serve: map[string]Work{
			workDate: {DateTime: workDate, Reflection: "own legacy note"},
		}},
	}
	view := WithWeeklyReflectionNotes(user, map[string]WeeklyReflection{
		"2026-W34": {UserID: "U999", Week: "2026-W34", Note: "other learner"},
	})

	require.Equal(t, "own legacy note", view.Portfolio.Serve[workDate].Reflection)
	require.Empty(t, view.Portfolio.Serve[workDate].Preview)
}
