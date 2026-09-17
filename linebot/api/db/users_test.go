package db_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/utils"
	"github.com/stretchr/testify/require"
)

// TestCreateUserData verifies the creation of user data
func TestCreateUserData(t *testing.T) {
	requireLive(t)
	t.Parallel()

	// Define test data using RandomAlphabetString
	testUserFolders := &storage.UserFolders{
		UserID:   utils.RandomAlphabetString(10),
		UserName: utils.RandomAlphabetString(10),
		RootPath: utils.RandomAlphabetString(10),
	}

	testGPTConvs := &db.GPTConversationIDs{
		Serve: utils.RandomAlphabetString(10),
		Smash: utils.RandomAlphabetString(10),
		Clear: utils.RandomAlphabetString(10),
		Lift:  utils.RandomAlphabetString(10),
	}

	// Call the method to create user data
	userData, err := firestoreClient.CreateUserData(testUserFolders, testGPTConvs)
	require.NoError(t, err)
	require.NotNil(t, userData)

	// Verify folder IDs and user data
	require.Equal(t, testUserFolders.UserName, userData.Name)
	require.Equal(t, testUserFolders.UserID, userData.ID)

	// Verify folder IDs
	require.Equal(t, testUserFolders.RootPath, userData.FolderPaths.Root)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(userData.ID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}

// TestGetUserData verifies retrieving user data
func TestGetUserData(t *testing.T) {
	requireLive(t)
	t.Parallel()

	// Create a user with RandomAlphabetString
	testUserID := utils.RandomAlphabetString(10)
	testUser := &db.UserData{
		Name: utils.RandomAlphabetString(10),
		ID:   testUserID,
		FolderPaths: db.FolderPaths{
			Root:      utils.RandomAlphabetString(10),
			Serve:     utils.RandomAlphabetString(10),
			Smash:     utils.RandomAlphabetString(10),
			Clear:     utils.RandomAlphabetString(10),
			Lift:      utils.RandomAlphabetString(10),
			Thumbnail: utils.RandomAlphabetString(10),
		},
		Handedness: db.Right,
		Portfolio: db.Portfolios{
			Serve: map[string]db.Work{},
			Smash: map[string]db.Work{},
			Clear: map[string]db.Work{},
			Lift:  map[string]db.Work{},
		},
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(*firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Test retrieving user data
	userData, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.Equal(t, testUser.Name, userData.Name)
	require.Equal(t, db.Right, userData.Handedness)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}

// TestUpdateUserHandedness verifies that the handedness can be updated
func TestUpdateUserHandedness(t *testing.T) {
	requireLive(t)
	t.Parallel()

	// Create a test user with RandomAlphabetString
	testUserID := utils.RandomAlphabetString(10)
	testUser := &db.UserData{
		Name:       utils.RandomAlphabetString(10),
		ID:         testUserID,
		Handedness: db.Right,
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(*firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Update handedness to left-handed
	err = firestoreClient.UpdateUserHandedness(testUser, db.Left)
	require.NoError(t, err)

	// Verify that handedness was updated
	updatedUser, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.Equal(t, db.Left, updatedUser.Handedness)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}
