package secret

import (
	"context"
	"fmt"
	"os"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

// envSecretNameVar names the Secret Manager secret holding this deployment's
// .env. It has no default: the variant and the original run in the same GCP
// project, so a fallback here would silently point the variant at the other
// deployment's bucket and databases.
const envSecretNameVar = "GCP_ENV_SECRET_NAME"

func DownloadEnvFile() error {
	ctx := context.Background()
	client, err := secretmanager.NewClient(ctx)
	if err != nil {
		return fmt.Errorf("failed to create secret manager client %w", err)
	}

	defer client.Close()

	// ensure the GCP_PROJECT_ID is set in the environment
	GCPProjectID := os.Getenv("GCP_PROJECT_ID")
	if GCPProjectID == "" {
		return fmt.Errorf("GCP project ID is not set for the current environment")
	}

	secretName := os.Getenv(envSecretNameVar)
	if secretName == "" {
		return fmt.Errorf("%s is not set for the current environment", envSecretNameVar)
	}

	// access secret
	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: fmt.Sprintf("projects/%s/secrets/%s/versions/latest", GCPProjectID, secretName),
	}
	result, err := client.AccessSecretVersion(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to access secret version: %w", err)
	}

	// save the secret as a .env file
	os.WriteFile(".env", result.Payload.Data, 0o444)
	return nil
}
