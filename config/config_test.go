package config_test

import (
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/config"
	"github.com/stretchr/testify/require"
)

func TestLoadConfig(t *testing.T) {
	os.Clearenv()
	// Set environment variables for the test
	t.Setenv("LINE_CHANNEL_SECRET", "test_line_channel_secret")
	t.Setenv("LINE_CHANNEL_TOKEN", "test_line_channel_token")
	t.Setenv("GCP_PROJECT_ID", "test_gcp_project_id")
	t.Setenv("GCP_CREDENTIALS", "test_gcp_credentials")
	t.Setenv("GOOGLE_DRIVE_ROOT_FOLDER", "test_google_drive_root_folder")
	t.Setenv("GCP_SECRET_VERSION", "test_secret_version")
	t.Setenv("FIREBASE_DATA_DB", "test_firebase_data_db")
	t.Setenv("FIREBASE_SESSION_DB", "test_firebase_session_db")
	t.Setenv("OPENAI_API_KEY", "test_openai_api_key")
	t.Setenv("OPENAI_ASSISTANT_ID", "test_openai_assistant_id")
	t.Setenv("POSE_ESTIMATION_SERVER_HOST", "test_pose_estimation_server_host")
	t.Setenv("POSE_ESTIMATION_SERVER_USER", "test_pose_estimation_server_user")
	t.Setenv("POSE_ESTIMATION_SERVER_PASSWORD", "test_pose_estimation_server_password")
	t.Setenv("PORT", "8080")

	// Load config
	config, err := config.LoadConfig("")

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
	require.Equal(t, "8080", config.Port)
}
