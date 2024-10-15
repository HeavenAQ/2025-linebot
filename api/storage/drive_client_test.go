package storage_test

import (
	"os"
	"testing"

	"github.com/HeavenAQ/api/storage"
	"github.com/stretchr/testify/require"
)

func cleanupFolders(userFolders *storage.UserFolders) {
	driveClient.DeleteFile(userFolders.ServeFolderID)
	driveClient.DeleteFile(userFolders.ThumbnailFolderID)
	driveClient.DeleteFile(userFolders.SmashFolderID)
	driveClient.DeleteFile(userFolders.ClearFolderID)
	driveClient.DeleteFile(userFolders.RootFolderID)
}

// TestCreateUserFolders tests the creation of user folders in Google Drive
func TestCreateUserFolders(t *testing.T) {
	// Create user folders
	userID := "test-user-id"
	userName := "Test User"
	userFolders, err := driveClient.CreateUserFolders(userID, userName)
	require.NoError(t, err)
	require.NotNil(t, userFolders)

	// Verify folders were created
	require.NotEmpty(t, userFolders.RootFolderID)
	require.NotEmpty(t, userFolders.ServeFolderID)
	require.NotEmpty(t, userFolders.SmashFolderID)
	require.NotEmpty(t, userFolders.ClearFolderID)
	require.NotEmpty(t, userFolders.ThumbnailFolderID)

	// Clean up
	cleanupFolders(userFolders)
}

// TestUploadVideo tests uploading a video to Google Drive
func TestUploadVideo(t *testing.T) {
	// Create user folders first
	userID := "test-user-id"
	userName := "Test User"
	userFolders, err := driveClient.CreateUserFolders(userID, userName)
	require.NoError(t, err)
	require.NotNil(t, userFolders)

	// Prepare video data (load the test file from test_files directory)
	videoPath := "./test_files/test_video.mp4"
	videoData, err := os.ReadFile(videoPath)
	require.NoError(t, err)

	// Prepare file info for video upload
	fileInfo := &storage.FileInfo{
		Drive: struct {
			ParentFolderID    string
			ThumbnailFolderID string
			Filename          string
		}{
			ParentFolderID:    userFolders.ServeFolderID,
			ThumbnailFolderID: userFolders.ThumbnailFolderID,
			Filename:          "test_video.mp4",
		},
		Local: struct {
			ThumbnailPath string
			VideoBlob     []byte
		}{
			VideoBlob: videoData,
		},
	}

	// Upload video to Google Drive
	driveFile, err := driveClient.UploadVideo(fileInfo)
	require.NoError(t, err)
	require.NotNil(t, driveFile)

	// Verify the uploaded video
	require.Equal(t, "test_video.mp4", driveFile.Name)

	// Clean up
	driveClient.DeleteFile(driveFile.Id)
	cleanupFolders(userFolders)
}

// TestUploadThumbnail tests uploading a thumbnail to Google Drive
func TestUploadThumbnail(t *testing.T) {
	// Create user folders first
	userID := "test-user-id"
	userName := "Test User"
	userFolders, err := driveClient.CreateUserFolders(userID, userName)
	require.NoError(t, err)
	require.NotNil(t, userFolders)

	// Prepare thumbnail data (load the test file from test_files directory)
	thumbnailPath := "./test_files/test_thumbnail.jpg"

	// Prepare file info for thumbnail upload
	fileInfo := &storage.FileInfo{
		Drive: struct {
			ParentFolderID    string
			ThumbnailFolderID string
			Filename          string
		}{
			ParentFolderID:    userFolders.ServeFolderID,
			ThumbnailFolderID: userFolders.ThumbnailFolderID,
			Filename:          "test_video",
		},
		Local: struct {
			ThumbnailPath string
			VideoBlob     []byte
		}{
			ThumbnailPath: thumbnailPath,
		},
	}

	// Upload thumbnail to Google Drive
	thumbnailFile, err := driveClient.UploadThumbnail(fileInfo)
	require.NoError(t, err)
	require.NotNil(t, thumbnailFile)

	// Verify the uploaded thumbnail
	require.Equal(t, "test_video_thumbnail", thumbnailFile.Name)

	// Optional: clean up uploaded files in Google Drive after test
	driveClient.DeleteFile(thumbnailFile.Id)
	cleanupFolders(userFolders)
}

func TestDeleteFile(t *testing.T) {
	// Create user folders first
	userID := "test-user-id"
	userName := "Test User"
	userFolders, err := driveClient.CreateUserFolders(userID, userName)
	require.NoError(t, err)
	require.NotNil(t, userFolders)

	// Prepare video data (load the test file from test_files directory)
	videoPath := "./test_files/test_video.mp4"
	videoData, err := os.ReadFile(videoPath)
	require.NoError(t, err)

	// Prepare file info for video upload
	fileInfo := &storage.FileInfo{
		Drive: struct {
			ParentFolderID    string
			ThumbnailFolderID string
			Filename          string
		}{
			ParentFolderID:    userFolders.ServeFolderID,
			ThumbnailFolderID: userFolders.ThumbnailFolderID,
			Filename:          "test_video.mp4",
		},
		Local: struct {
			ThumbnailPath string
			VideoBlob     []byte
		}{
			VideoBlob: videoData,
		},
	}

	// Upload video to Google Drive
	driveFile, err := driveClient.UploadVideo(fileInfo)
	require.NoError(t, err)
	require.NotNil(t, driveFile)

	// Delete the uploaded file
	err = driveClient.DeleteFile(driveFile.Id)
	require.NoError(t, err)
	cleanupFolders(userFolders)
}
