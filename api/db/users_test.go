package db_test

import (
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/utils"
	"github.com/stretchr/testify/require"
)

// TestCreateUserData verifies the creation of user data
func TestCreateUserData(t *testing.T) {
	t.Parallel()

	// Define test data using RandomAlphabetString
	userID := utils.RandomAlphabetString(10)
	testUserFolders := &storage.UserFolders{
		UserID:   userID,
		UserName: utils.RandomAlphabetString(10),
		RootPath: userID + "/",
	}

	testGPTthreads := &db.GPTThreadIDs{
		JumpingClear:            utils.RandomAlphabetString(10),
		FrontCourtHighPointDrop: utils.RandomAlphabetString(10),
		DefensiveClear:          utils.RandomAlphabetString(10),
		FrontCourtLowPointLift:  utils.RandomAlphabetString(10),
		JumpingSmash:            utils.RandomAlphabetString(10),
		MidCourtChasseToBack:    utils.RandomAlphabetString(10),
		ForwardCrossStep:        utils.RandomAlphabetString(10),
		MidCourtBackCrossStep:   utils.RandomAlphabetString(10),
		DefensiveSlideStep:      utils.RandomAlphabetString(10),
	}

	// Call the method to create user data
	userData, err := firestoreClient.CreateUserData(testUserFolders, testGPTthreads)
	require.NoError(t, err)
	require.NotNil(t, userData)

	// Verify folder paths and user data
	require.Equal(t, testUserFolders.UserName, userData.Name)
	require.Equal(t, testUserFolders.UserID, userData.ID)

	// Verify folder paths
	require.Equal(t, testUserFolders.RootPath, userData.FolderPaths.Root)
	require.Equal(t, testUserFolders.RootPath+"front_court_high_point_drop/", userData.FolderPaths.FrontCourtHighPointDrop)

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
		FolderPaths: db.FolderPaths{
			Root:                    testUserID + "/",
			FrontCourtHighPointDrop: testUserID + "/front_court_high_point_drop/",
			Thumbnail:               testUserID + "/thumbnails/",
		},
		Handedness: db.Right,
		Portfolio: db.Portfolios{
			JumpingClear:            map[string]db.Work{},
			FrontCourtHighPointDrop: map[string]db.Work{},
			DefensiveClear:          map[string]db.Work{},
			FrontCourtLowPointLift:  map[string]db.Work{},
			JumpingSmash:            map[string]db.Work{},
			MidCourtChasseToBack:    map[string]db.Work{},
			ForwardCrossStep:        map[string]db.Work{},
			MidCourtBackCrossStep:   map[string]db.Work{},
			DefensiveSlideStep:      map[string]db.Work{},
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
			JumpingClear:            map[string]db.Work{},
			FrontCourtHighPointDrop: map[string]db.Work{},
			DefensiveClear:          map[string]db.Work{},
			FrontCourtLowPointLift:  map[string]db.Work{},
			JumpingSmash:            map[string]db.Work{},
			MidCourtChasseToBack:    map[string]db.Work{},
			ForwardCrossStep:        map[string]db.Work{},
			MidCourtBackCrossStep:   map[string]db.Work{},
			DefensiveSlideStep:      map[string]db.Work{},
		},
	}
	_, err := firestoreClient.Data.Doc(testUserID).Set(*firestoreClient.Ctx, testUser)
	require.NoError(t, err)

	// Define test data for video creation
	today := time.Now().Format("2006-01-02-15-04")
	videoFile := &storage.UploadedFile{
		Name: today + ".mp4",
		Path: testUserID + "/front_court_high_point_drop/" + today + ".mp4",
	}
	thumbnailFile := &storage.UploadedFile{
		Name: today + "_thumbnail.jpg",
		Path: testUserID + "/thumbnails/" + today + "_thumbnail.jpg",
	}
	session := &db.UserSession{
		UserState: db.WritingNotes,
	}

	// Call the method to add video to portfolio
	err = firestoreClient.CreateUserPortfolioVideo(
		testUser,
		&testUser.Portfolio.FrontCourtHighPointDrop,
		today,
		session,
		videoFile,
		thumbnailFile,
	)
	require.NoError(t, err)

	// Verify that the video was added to the portfolio
	updatedUser, err := firestoreClient.GetUserData(testUserID)
	require.NoError(t, err)
	require.NotNil(t, updatedUser.Portfolio.FrontCourtHighPointDrop[today])

	// Clean up the created data after the test
	_, err = firestoreClient.Data.Doc(testUserID).Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}
