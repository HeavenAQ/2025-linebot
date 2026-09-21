package app

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/stretchr/testify/require"
)

// Typing while a card is waiting for a tap used to reach a JSON decode of an
// empty payload, which answered the learner with a decode error.
func TestAwaitingButtonTapCoversTheStepsThatParseAPostback(t *testing.T) {
	t.Parallel()

	for _, session := range []db.UserSession{
		{UserState: db.ViewingPortfoilo, ActionStep: db.SelectingSkill},
		{UserState: db.WritingPreviewNote, ActionStep: db.SelectingSkill},
		{UserState: db.WritingReflectionNote, ActionStep: db.SelectingPortfolio},
		{UserState: db.ViewingExpertVideos, ActionStep: db.SelectingSkill},
		{UserState: db.AnalyzingVideo, ActionStep: db.SelectingSkill},
	} {
		require.True(t, awaitingButtonTap(&session), "%v/%v", session.UserState, session.ActionStep)
	}
}

// The steps that do read what the learner types must keep reading it.
func TestAwaitingButtonTapLeavesTheTypingStepsAlone(t *testing.T) {
	t.Parallel()

	for _, session := range []db.UserSession{
		{UserState: db.WritingReflectionNote, ActionStep: db.WritingReflection},
		{UserState: db.AnalyzingVideo, ActionStep: db.UploadingVideo},
		{UserState: db.None, ActionStep: db.Empty},
	} {
		require.False(t, awaitingButtonTap(&session), "%v/%v", session.UserState, session.ActionStep)
	}
}
