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

	user := &UserData{Portfolio: Portfolios{
		Lift:  map[string]Work{"2026-07-01-10-00": work(71, "")},
		Clear: map[string]Work{"2026-07-02-10-00": work(65, "")},
	}}
	history := learningHistory(user, 5)
	require.Len(t, history, 2)
	require.Equal(t, "clear", history[0].Skill)
	require.Equal(t, "lift", history[1].Skill)
}
