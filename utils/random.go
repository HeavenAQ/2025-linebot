package utils

import (
	"os"
	"testing"
)

func SetRandomEnv(t *testing.T) {
	t.Helper()
	os.Clearenv()

	// Set environment variables for the test
	t.Setenv("LINE_CHANNEL_SECRET", "test_line_channel_secret")
	t.Setenv("LINE_CHANNEL_TOKEN", "test_line_channel_token")
	t.Setenv("GCP_PROJECT_ID", "test_gcp_project_id")
	t.Setenv("GCP_CREDENTIALS", "test_gcp_credentials")
	t.Setenv("GOOGLE_DRIVE_ROOT_FOLDER", "test_google_drive_root_folder")
	t.Setenv("SECRET_VERSION", "test_secret_version")
	t.Setenv("FIREBASE_DATA_DB", "test_firebase_data_db")
	t.Setenv("FIREBASE_SESSION_DB", "test_firebase_session_db")
	t.Setenv("OPENAI_API_KEY", "test_openai_api_key")
	t.Setenv("OPENAI_ASSISTANT_ID", "test_openai_assistant_id")
	t.Setenv("POSE_ESTIMATION_SERVER_HOST", "test_pose_estimation_server_host")
	t.Setenv("POSE_ESTIMATION_SERVER_USER", "test_pose_estimation_server_user")
	t.Setenv("POSE_ESTIMATION_SERVER_PASSWORD", "test_pose_estimation_server_password")
	t.Setenv("PORT", "8080")
}
