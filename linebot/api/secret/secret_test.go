package secret

import (
    "os"
    "testing"

    "github.com/stretchr/testify/require"
)

// TestSecretManagerAccess tests retrieving credentials from Secret Manager
func TestSecretManagerAccess(t *testing.T) {
    if os.Getenv("RUN_LIVE_SECRET") != "1" {
        t.Skip("Skipping Secret Manager live test; set RUN_LIVE_SECRET=1 to enable.")
    }
    // Fetch the secret
    err := DownloadEnvFile()
    require.NoError(t, err)

	content, err := os.ReadFile(".env")
	require.NoError(t, err)
	require.NotEmpty(t, content)
}
