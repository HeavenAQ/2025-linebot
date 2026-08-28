package db

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// The whole point of the toggle is that this semester's course runs serve and
// smash only; lift and clear are withdrawn.
func TestSupportedCoversThisSemestersSkills(t *testing.T) {
	t.Parallel()

	require.True(t, Serve.Supported())
	require.True(t, Smash.Supported())
	require.False(t, Clear.Supported())
	require.False(t, Lift.Supported())
}

// SkillStrToEnum returns -1 for anything it does not recognise, and both
// String and ChnString index a fixed array. Supported must answer for that
// value rather than panic, since it guards the stale-payload path.
func TestSupportedRejectsValuesOutsideTheEnum(t *testing.T) {
	t.Parallel()

	require.False(t, BadmintonSkill(-1).Supported())
	require.False(t, BadmintonSkill(9).Supported())
	require.False(t, BadmintonSkill(-1).Valid())
	require.True(t, Serve.Valid())
}

func TestIsSupportedSkillReadsRawStrings(t *testing.T) {
	t.Parallel()

	require.True(t, IsSupportedSkill("serve"))
	require.True(t, IsSupportedSkill("smash"))
	require.False(t, IsSupportedSkill("clear"))
	require.False(t, IsSupportedSkill("lift"))

	// Empty and unrecognised session or postback values must not slip through.
	require.False(t, IsSupportedSkill(""))
	require.False(t, IsSupportedSkill("Serve"))
	require.False(t, IsSupportedSkill("drop_shot"))
}

func TestSupportedSkillsKeepsEnumOrder(t *testing.T) {
	t.Parallel()

	require.Equal(t, []BadmintonSkill{Serve, Smash}, SupportedSkills())
	require.Equal(t, "【發球】、【殺球】", SupportedSkillsChnString())
}

// SkillOrder still spans every skill, so a learner's stored lift and clear
// portfolios keep being read back.
func TestSkillOrderStillCoversWithdrawnSkills(t *testing.T) {
	t.Parallel()

	require.Equal(t, [...]string{"serve", "smash", "clear", "lift"}, SkillOrder)

	portfolio := Portfolios{
		Lift:  map[string]Work{"2026-07-01-10-00": {DateTime: "2026-07-01-10-00"}},
		Clear: map[string]Work{"2026-07-02-10-00": {DateTime: "2026-07-02-10-00"}},
	}
	for _, skill := range []string{"clear", "lift"} {
		require.Len(t, portfolio.GetSkillPortfolio(skill), 1, skill)
	}
}

// String, ChnString and UserStateChnStrToEnum are three parallel lists indexed
// by the value itself. A state added to one but not the others silently renames
// every state after it -- a bug this file has already seen once -- so every
// state is round-tripped through all three.
func TestUserStateNamesStayInStep(t *testing.T) {
	t.Parallel()

	for state := WritingPreviewNote; state <= None; state++ {
		require.NotEmpty(t, state.String(), state)
		chinese := state.ChnString()
		require.NotEmpty(t, chinese, state)

		parsed, err := UserStateChnStrToEnum(chinese)
		require.NoError(t, err, chinese)
		require.Equal(t, state, parsed, chinese)
	}
}

// The rich menu sends its label as plain text, so these are the strings the
// menu areas have to carry. The two halves of a week are separate entries and
// must not collapse into one state.
func TestWeeklyMenuEntriesAreTheirOwnStates(t *testing.T) {
	t.Parallel()

	require.Equal(t, "課前預習", WritingPreviewNote.ChnString())
	require.Equal(t, "學習反思", WritingReflectionNote.ChnString())
	require.NotEqual(t, WritingPreviewNote, WritingReflectionNote)

	preview, err := UserStateChnStrToEnum("課前預習")
	require.NoError(t, err)
	require.Equal(t, WritingPreviewNote, preview)

	reflection, err := UserStateChnStrToEnum("學習反思")
	require.NoError(t, err)
	require.Equal(t, WritingReflectionNote, reflection)

	// The entry these two replaced is gone rather than kept as a fallback.
	_, err = UserStateChnStrToEnum("預習及反思")
	require.Error(t, err)
}
