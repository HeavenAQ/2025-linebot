package line

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// A learner who never chose a stroke is not asking for a closed one; they are
// told how to choose, not that their choice is unavailable.
func TestUnsupportedSkillMessageAsksAnEmptySkillToChooseOne(t *testing.T) {
	t.Parallel()

	for _, skill := range []string{"", "   "} {
		message := unsupportedSkillMessage(skill)
		require.Contains(t, message, "動作分析")
		require.NotContains(t, message, "未開放")
	}
}

func TestUnsupportedSkillMessageNamesAClosedStroke(t *testing.T) {
	t.Parallel()

	require.Contains(t, unsupportedSkillMessage("lift"), "未開放")
	require.Contains(t, unsupportedSkillMessage("not-a-skill"), "未開放")
}
