package storage_test

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/api/secret"
	"github.com/HeavenAQ/api/storage"
	"github.com/HeavenAQ/config"
	"github.com/stretchr/testify/require"
)

// Global variables for the test
var (
	cfg         *config.Config
	driveClient *storage.GoogleDriveClient
)

func TestMain(m *testing.M) {
	// Load config (ensure your config includes GCP project, credentials, and root folder info)
	conf, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load configurations")
	}
	cfg = conf
	driveClient = setupGoogleDriveClient(&testing.T{})

	// Run tests
	os.Exit(m.Run())
}

func setupGoogleDriveClient(t *testing.T) *storage.GoogleDriveClient {
	// Fetch credentials from Secret Manager (adjust for your environment)
	secretName := secret.GetSecretString(cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)
	credentials, err := secret.AccessSecretVersion(secretName)
	require.NoError(t, err)
	require.NotNil(t, credentials)

	// Initialize Google Drive client
	rootFolderID := cfg.GCP.Storage.GoogleDrive.RootFolder
	driveClient, err := storage.NewGoogleDriveClient(credentials, rootFolderID)
	require.NoError(t, err)
	require.NotNil(t, driveClient)

	return driveClient
}
