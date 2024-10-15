package db_test

import (
	"testing"

	"github.com/HeavenAQ/api/db"
	"github.com/HeavenAQ/api/drive"
	"github.com/stretchr/testify/require"
	googleDrive "google.golang.org/api/drive/v3"
)

func TestCreateUserData(t *testing.T) {
	// Define test data
	testUserFolders := &drive.UserFolders{
		UserID:            "test-user-id",
		UserName:          "John Doe",
		RootFolderID:      "root-folder-id",
		ServeFolderID:     "serve-folder-id",
		SmashFolderID:     "smash-folder-id",
		ClearFolderID:     "clear-folder-id",
		ThumbnailFolderID: "thumbnail-folder-id",
	}

	// Call the method to create user data
	userData, err := firestoreClient.CreateUserData(testUserFolders)
	require.NoError(t, err)
	require.NotNil(t, userData)
	require.Equal(t, "John Doe", userData.Name)
	require.Equal(t, "test-user-id", userData.ID)

	// Verify folder IDs
	require.Equal(t, "root-folder-id", userData.FolderIDs.Root)
	require.Equal(t, "serve-folder-id", userData.FolderIDs.Serve)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc("test-user-id").Delete(firestoreClient.Ctx)
	require.NoError(t, err)
}

func TestGetUserData(t *testing.T) {
	// Create a user manually
	testUserID := "test-user-id"
	testUser := &db.UserData{
		Name: "Test User",
		ID:   testUserID,
		FolderIDs: db.FolderIDs{
			Root:      "root-folder",
			Serve:     "serve-folder",
			Smash:     "smash-folder",
			Clear:     "clear-folder",
			Thumbnail: "thumbnail-folder",
		},
		Handedness: db.Right,
		Portfolio: db.Portfolio{
			Serve: map[string]db.Work{},
			Smash: map[string]db.Work{},
			Clear: map[string]db.Work{},
		},
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Test retrieving user data
	userData, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.Equal(t, "Test User", userData.Name)
	require.Equal(t, db.Right, userData.Handedness)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(firestoreClient.Ctx)
	require.NoError(t, err)
}

func TestUpdateUserHandedness(t *testing.T) {
	// Create a test user
	testUserID := "test-user-id"
	testUser := &db.UserData{
		Name:       "Test User",
		ID:         testUserID,
		Handedness: db.Right, // initially set as right-handed
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Update handedness to left-handed
	err = firestoreClient.UpdateUserHandedness(testUser, db.Left)
	require.NoError(t, err)

	// Verify that handedness was updated
	updatedUser, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.Equal(t, db.Left, updatedUser.Handedness)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(firestoreClient.Ctx)
	require.NoError(t, err)
}

func TestCreateUserPortfolioVideo(t *testing.T) {
	// Create a test user
	testUserID := "portfolio-user-id"
	testUser := &db.UserData{
		Name: "Test User",
		ID:   testUserID,
		Portfolio: db.Portfolio{
			Serve: map[string]db.Work{},
		},
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Define test data for video creation
	driveFile := &googleDrive.File{
		Id:   "video-file-id",
		Name: "2024-10-14",
	}
	thumbnailFile := &googleDrive.File{
		Id: "thumbnail-file-id",
	}
	session := &db.UserSession{
		UserState: db.WritingReflection,
	}
	aiRating := float32(4.5)
	aiSuggestions := "Improve form"

	// Call the method to add video to portfolio
	err = firestoreClient.CreateUserPortfolioVideo(testUser, &testUser.Portfolio.Serve, session, driveFile, thumbnailFile, aiRating, aiSuggestions)
	require.NoError(t, err)

	// Verify that the video was added to the portfolio
	updatedUser, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.NotNil(t, updatedUser.Portfolio.Serve["2024-10-14"])
	require.Equal(t, aiRating, updatedUser.Portfolio.Serve["2024-10-14"].Rating)

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(firestoreClient.Ctx)
	require.NoError(t, err)
}
