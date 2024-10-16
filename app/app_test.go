package app_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/app"
	"github.com/stretchr/testify/require"
)

func TestNewApp(t *testing.T) {
	// Set required environment variables to simulate configuration loading
	// This simulates the expected env vars for your app to load the configuration

	// Call NewApp to create the app
	app := app.NewApp("../.env")

	// Ensure the app was created successfully
	require.NotNil(t, app, "App should not be nil")

	// Check that the config is not nil
	require.NotNil(t, app.Config, "Config should not be nil")

	// Check that the logger is initialized
	require.NotNil(t, app.Logger, "Logger should not be nil")

	// Check that the LineBot client is initialized
	require.NotNil(t, app.LineBot, "LineBot should not be nil")
}
