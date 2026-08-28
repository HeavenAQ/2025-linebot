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
