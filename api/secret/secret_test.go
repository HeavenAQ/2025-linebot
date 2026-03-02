package secret

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestGetSecretString tests the GetSecretString function
func TestGetSecretString(t *testing.T) {
	// Expected secret string
	testProjectID := "test-project-id"
	testSecretID := "test-secret-id"
	testSecretVersion := "latest"
	expectedSecretString := "projects/test-project-id/secrets/test-secret-id/versions/latest"

	// Call the function
	result := GetSecretString(testProjectID, testSecretID, testSecretVersion)

	// Validate the result
	require.Equal(t, expectedSecretString, result)
}

func TestDownloadSecretToFileRequiresRealSecretAccess(t *testing.T) {
	projectID := os.Getenv("GCP_PROJECT_ID")
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
	}

	secretID := os.Getenv("GCP_ENV_SECRET_ID")
	secretVersion := os.Getenv("GCP_ENV_SECRET_VERSION")
	if secretVersion == "" {
		secretVersion = "latest"
	}

	if projectID == "" || secretID == "" {
		t.Skip("Skipping Secret Manager integration test: env secret is not configured")
	}

	path := filepath.Join(t.TempDir(), ".env")
	secretName := GetSecretString(projectID, secretID, secretVersion)

	err := DownloadSecretToFile(secretName, path)
	require.NoError(t, err)

	content, err := os.ReadFile(path)
	require.NoError(t, err)
	require.NotEmpty(t, content)
}
