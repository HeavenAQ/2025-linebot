package poseestimation

import (
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// Replace with the actual URL of your test server and credentials.
const (
	testServerURL = "http://127.0.0.1:8000/upload"
	testUsername  = "admin"
	testPassword  = "thisisacomplicatedpassword"
)

func TestClient_ProcessVideo_Integration(t *testing.T) {
	// Read the test video file as a blob.
	videoBlob, err := os.ReadFile("../../tmp/serve.mp4") // Ensure this path points to a valid video file.
	require.NoError(t, err, "failed to read test video file")

	// Initialize the client with the test server URL, video blob, and credentials.
	client := NewClient(testUsername, testPassword, testServerURL, videoBlob)

	// Run the ProcessVideo method.
	responseData, err := client.ProcessVideo()
	require.NoError(t, err, "expected no error from ProcessVideo")
	require.NotNil(t, responseData, "expected response data")

	// Validate grading score - this will vary depending on the server logic.
	require.GreaterOrEqual(t, responseData.GradingScore, 0.0, "grading score should be >= 0")

	// Ensure processed video data is returned.
	require.NotEmpty(t, responseData.ProcessedVideo, "processed video should not be empty")

	// Save the processed video to a file.
	outputFilePath := "../../tmp/processed_video.mp4"
	err = os.WriteFile(outputFilePath, responseData.ProcessedVideo, 0644)
	require.NoError(t, err, "failed to save processed video file")

	// Log success message for confirmation.
	log.Printf("Processed video saved successfully to %s", outputFilePath)
}

func TestMain(m *testing.M) {
	// Setup before tests if necessary.
	log.Println("Starting integration tests for poseestimation package")

	// Run tests.
	exitVal := m.Run()

	// Teardown after tests if necessary.
	log.Println("Integration tests for poseestimation package completed")

	// Exit with the test's exit value.
	os.Exit(exitVal)
}
