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
	t.Setenv("GCS_BUCKET_NAME", "test_bucket_name")
	t.Setenv("GCP_SECRET_VERSION", "test_secret_version")
	t.Setenv("FIREBASE_DATA_DB", "test_firebase_data_db")
	t.Setenv("FIREBASE_SESSION_DB", "test_firebase_session_db")
	t.Setenv("ANALYSIS_GRPC_TARGET", "analysis.example.test:443")
	t.Setenv("ANALYSIS_GRPC_API_KEY", "test_analysis_api_key")
	t.Setenv("ANALYSIS_GRPC_INSECURE", "false")
	t.Setenv("LIFF_REVIEW_URL", "https://liff.example.test/personal?tab=review")
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
	require.Equal(t, "test_bucket_name", config.GCP.Storage.BucketName)
	require.Equal(t, "test_secret_version", config.GCP.Secrets.SecretVersion)
	require.Equal(t, "test_firebase_data_db", config.GCP.Database.DataDB)
	require.Equal(t, "test_firebase_session_db", config.GCP.Database.SessionDB)
	require.Equal(t, "analysis.example.test:443", config.AnalysisServer.Target)
	require.Equal(t, "test_analysis_api_key", config.AnalysisServer.APIKey)
	require.False(t, config.AnalysisServer.Insecure)
	require.Equal(t, "8080", config.Port)
	require.Equal(t, "https://liff.example.test/personal?tab=review", config.ReviewURL())
}

// The flag that decides whether learner frames reach a third-party model must
// come from configuration, so one binary can serve a consented deployment and a
// non-consented one without a code difference between them.
func TestSkipCoachingComesFromTheEnvironment(t *testing.T) {
	t.Setenv("ANALYSIS_SKIP_COACHING", "true")
	t.Setenv("PORT", "8080")

	config, err := config.LoadConfig(".env.absent")

	require.NoError(t, err)
	require.True(t, config.AnalysisServer.SkipCoaching)
}

func TestSkipCoachingDefaultsToSendingCoaching(t *testing.T) {
	t.Setenv("PORT", "8080")

	config, err := config.LoadConfig(".env.absent")

	require.NoError(t, err)
	require.False(t, config.AnalysisServer.SkipCoaching)
}
