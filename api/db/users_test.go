package db_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/utils"
	"github.com/stretchr/testify/require"
	googleDrive "google.golang.org/api/drive/v3"
)

// TestCreateUserData verifies the creation of user data
func TestCreateUserData(t *testing.T) {
	t.Parallel()

	// Define test data using RandomAlphabetString
	testUserFolders := &storage.UserFolders{
		UserID:            utils.RandomAlphabetString(10),
		UserName:          utils.RandomAlphabetString(10),
		RootFolderID:      utils.RandomAlphabetString(10),
		ServeFolderID:     utils.RandomAlphabetString(10),
		SmashFolderID:     utils.RandomAlphabetString(10),
		ClearFolderID:     utils.RandomAlphabetString(10),
		ThumbnailFolderID: utils.RandomAlphabetString(10),
	}

	// Call the method to create user data
	userData, err := firestoreClient.CreateUserData(testUserFolders)
	require.NoError(t, err)
	require.NotNil(t, userData)

	// Verify folder IDs and user data
	require.Equal(t, testUserFolders.UserName, userData.Name)
	require.Equal(t, testUserFolders.UserID, userData.ID)

	// Verify folder IDs
	require.Equal(t, testUserFolders.RootFolderID, userData.FolderIDs.Root)
	require.Equal(t, testUserFolders.ServeFolderID, userData.FolderIDs.Serve)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(userData.ID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}

// TestGetUserData verifies retrieving user data
func TestGetUserData(t *testing.T) {
	t.Parallel()

	// Create a user with RandomAlphabetString
	testUserID := utils.RandomAlphabetString(10)
	testUser := &db.UserData{
		Name: utils.RandomAlphabetString(10),
		ID:   testUserID,
		FolderIDs: db.FolderIDs{
			Root:      utils.RandomAlphabetString(10),
			Serve:     utils.RandomAlphabetString(10),
			Smash:     utils.RandomAlphabetString(10),
			Clear:     utils.RandomAlphabetString(10),
			Thumbnail: utils.RandomAlphabetString(10),
		},
		Handedness: db.Right,
		Portfolio: db.Portfolios{
			Serve: map[string]db.Work{},
			Smash: map[string]db.Work{},
			Clear: map[string]db.Work{},
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

// TestCreateUserPortfolioVideo verifies that a video can be added to the user's portfolio
func TestCreateUserPortfolioVideo(t *testing.T) {
	t.Parallel()

	// Create a test user with RandomAlphabetString
	testUserID := utils.RandomAlphabetString(10)
	testUser := &db.UserData{
		Name: utils.RandomAlphabetString(10),
		ID:   testUserID,
		Portfolio: db.Portfolios{
			Smash: map[string]db.Work{},
			Serve: map[string]db.Work{},
			Clear: map[string]db.Work{},
		},
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(*firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Define test data for video creation
	driveFile := &googleDrive.File{
		Id:   utils.RandomAlphabetString(10),
		Name: "2024-10-14",
	}
	thumbnailFile := &googleDrive.File{
		Id: utils.RandomAlphabetString(10),
	}
	session := &db.UserSession{
		UserState: db.WritingNotes,
	}
	aiRating := float32(4.5)
	aiSuggestions := "Improve form"

	// Call the method to add video to portfolio
	err = firestoreClient.CreateUserPortfolioVideo(testUser, &testUser.Portfolio.Serve, session, driveFile, thumbnailFile, aiRating, aiSuggestions)
	require.NoError(t, err)

	// Verify that the video was added to the portfolio
	updatedUser, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.NotNil(t, updatedUser.Portfolio.Serve[driveFile.Name])
	require.Equal(t, aiRating, updatedUser.Portfolio.Serve[driveFile.Name].Rating)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}
