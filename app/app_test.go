package app

import (
	"os"
	"testing"

	"github.com/HeavenAQ/utils"
	"github.com/stretchr/testify/require"
)

func TestNewApp(t *testing.T) {
	// Clear any existing environment variables (optional)
	os.Clearenv()

	// Set required environment variables to simulate configuration loading
	// This simulates the expected env vars for your app to load the configuration
	utils.SetRandomEnv(t)

	// Call NewApp to create the app
	app := NewApp("../.env")

	// Ensure the app was created successfully
	require.NotNil(t, app, "App should not be nil")

	// Check that the config is not nil
	require.NotNil(t, app.Config, "Config should not be nil")

	// Check that the logger is initialized
	require.NotNil(t, app.Logger, "Logger should not be nil")

	// Check that the LineBot client is initialized
	require.NotNil(t, app.LineBot, "LineBot should not be nil")

	// Validate some config values (assuming they should match the environment variables set above)
	require.Equal(t, "test_line_channel_secret", app.Config.Line.ChannelSecret)
	require.Equal(t, "test_line_channel_token", app.Config.Line.ChannelToken)
	require.Equal(t, "test_gcp_project_id", app.Config.GCP.ProjectID)
	require.Equal(t, "test_google_drive_root_folder", app.Config.GCP.Storage.GoogleDrive.RootFolder)
}
