package config

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLoadConfig(t *testing.T) {
	os.Clearenv()
	// Set environment variables for the test
	os.Setenv("LINE_CHANNEL_SECRET", "test_line_channel_secret")
	os.Setenv("LINE_CHANNEL_TOKEN", "test_line_channel_token")
	os.Setenv("GCP_PROJECT_ID", "test_gcp_project_id")
	os.Setenv("GCP_CREDENTIALS", "test_gcp_credentials")
	os.Setenv("GOOGLE_DRIVE_ROOT_FOLDER", "test_google_drive_root_folder")
	os.Setenv("SECRET_VERSION", "test_secret_version")
	os.Setenv("FIREBASE_DATA_DB", "test_firebase_data_db")
	os.Setenv("FIREBASE_SESSION_DB", "test_firebase_session_db")
	os.Setenv("OPENAI_API_KEY", "test_openai_api_key")
	os.Setenv("OPENAI_ASSISTANT_ID", "test_openai_assistant_id")
	os.Setenv("POSE_ESTIMATION_SERVER_HOST", "test_pose_estimation_server_host")
	os.Setenv("POSE_ESTIMATION_SERVER_USER", "test_pose_estimation_server_user")
	os.Setenv("POSE_ESTIMATION_SERVER_PASSWORD", "test_pose_estimation_server_password")

	// Load config
	config, err := LoadConfig()

	// Ensure no errors occurred
	require.NoError(t, err)

	// Check the configuration values
	require.Equal(t, "test_line_channel_secret", config.Line.ChannelSecret)
	require.Equal(t, "test_line_channel_token", config.Line.ChannelToken)
	require.Equal(t, "test_gcp_project_id", config.GCP.ProjectID)
	require.Equal(t, "test_gcp_credentials", config.GCP.Credentials)
	require.Equal(t, "test_google_drive_root_folder", config.GCP.Storage.GoogleDrive.RootFolder)
	require.Equal(t, "test_secret_version", config.GCP.Secrets.SecretVersion)
	require.Equal(t, "test_firebase_data_db", config.GCP.Database.DataDB)
	require.Equal(t, "test_firebase_session_db", config.GCP.Database.SessionDB)
	require.Equal(t, "test_openai_api_key", config.GPT.APIKey)
	require.Equal(t, "test_openai_assistant_id", config.GPT.AssistantID)
	require.Equal(t, "test_pose_estimation_server_host", config.PoseEstimationServer.Host)
	require.Equal(t, "test_pose_estimation_server_user", config.PoseEstimationServer.User)
	require.Equal(t, "test_pose_estimation_server_password", config.PoseEstimationServer.Password)
}
