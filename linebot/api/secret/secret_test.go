package secret

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestSecretManagerAccess tests retrieving credentials from Secret Manager
func TestSecretManagerAccess(t *testing.T) {
	// Replace with your actual secret name from Google Secret Manager
	secretName := fmt.Sprintf("projects/%v/secrets/%v/versions/%v", cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)

	// Fetch the secret
	secretData, err := AccessSecretVersion(secretName)
	require.NoError(t, err)
	require.NotNil(t, secretData)
}

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
