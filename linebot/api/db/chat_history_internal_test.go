package db

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRedactScoreRecordsReplacesRawPayloads(t *testing.T) {
	history := &ChatHistory{Messages: []ChatMessage{
		{Role: "user", Text: "以下為我此次動作的資料，請分析並給出改善建議：\n慣用手：right\n動作技能：serve\n動作評分細節：[{\"description\":\"雙手平舉\",\"grade\":9.29}]"},
		{Role: "assistant", Text: "1. 雙手平舉表現很好"},
		{Role: "user", Text: "你應該給我一些建議"},
	}}

	redacted := RedactScoreRecords(history)

	require.Equal(t, ScoreRecordLabel, redacted.Messages[0].Text)
	require.NotContains(t, redacted.Messages[0].Text, "grade")
	require.NotContains(t, redacted.Messages[0].Text, "9.29")
	// Everything a learner actually wrote or was told survives untouched.
	require.Equal(t, "1. 雙手平舉表現很好", redacted.Messages[1].Text)
	require.Equal(t, "你應該給我一些建議", redacted.Messages[2].Text)
}

func TestRedactScoreRecordsHandlesNil(t *testing.T) {
	require.Nil(t, RedactScoreRecords(nil))
}
