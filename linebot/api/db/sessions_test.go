package db_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/stretchr/testify/require"
)

// Test GetUserSession function
func TestGetUserSession(t *testing.T) {
	// Create a test user session
	testUserID := "test-user-id"
	expectedSession := db.UserSession{UserState: db.WritingNotes, Skill: "Writing", Handedness: "right"}
	_, err := firestoreClient.Sessions.Doc(testUserID).Set(*firestoreClient.Ctx, expectedSession)
	require.NoError(t, err)

	// Retrieve the user session
	session, err := firestoreClient.GetUserSession(testUserID)
	require.NoError(t, err)
	require.Equal(t, expectedSession.UserState, session.UserState)
	require.Equal(t, expectedSession.Skill, session.Skill)

	// Clean up
	firestoreClient.Sessions.Doc(testUserID).Delete(*firestoreClient.Ctx)
}

// Test CreateUserSession function
func TestCreateUserSession(t *testing.T) {
	// Create a new session
	testUserID := "new-user-id"
	newSession, err := firestoreClient.CreateUserSession(testUserID)
	require.NoError(t, err)
	require.NotNil(t, newSession)
	require.Equal(t, db.None, newSession.UserState)
	require.Equal(t, "", newSession.Skill)

	// Verify it was written to Firestore
	savedSession, err := firestoreClient.GetUserSession(testUserID)
	require.NoError(t, err)
	require.Equal(t, db.None, savedSession.UserState)
	require.Equal(t, "", savedSession.Skill)

	// Clean up
	firestoreClient.Sessions.Doc(testUserID).Delete(*firestoreClient.Ctx)
}

// Test UpdateSessionUserState function
func TestUpdateSessionUserState(t *testing.T) {
	// Create a test user session
	testUserID := "state-user-id"
	firestoreClient.CreateUserSession(testUserID)

	// Update the user state
	err := firestoreClient.UpdateSessionUserState(testUserID, db.ChattingWithGPT, db.SelectingSkill)
	require.NoError(t, err)

	// Verify the state was updated in Firestore
	savedSession, err := firestoreClient.GetUserSession(testUserID)
	require.NoError(t, err)
	require.Equal(t, db.ChattingWithGPT, savedSession.UserState)

	// Clean up
	firestoreClient.Sessions.Doc(testUserID).Delete(*firestoreClient.Ctx)
}

// Test UpdateSessionUserSkill function
func TestUpdateSessionUserSkill(t *testing.T) {
	// Create a test user session
	testUserID := "skill-user-id"
	firestoreClient.CreateUserSession(testUserID)

	// Update the user's skill
	newSkill := "Public Speaking"
	err := firestoreClient.UpdateSessionUserSkill(testUserID, newSkill)
	require.NoError(t, err)

	// Verify the skill was updated in Firestore
	savedSession, err := firestoreClient.GetUserSession(testUserID)
	require.NoError(t, err)
	require.Equal(t, newSkill, savedSession.Skill)

	// Clean up
	firestoreClient.Sessions.Doc(testUserID).Delete(*firestoreClient.Ctx)
}
