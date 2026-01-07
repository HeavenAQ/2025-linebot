package secret

import (
	"context"
	"fmt"
	"os"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

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

	// access secret
	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: fmt.Sprintf("projects/%s/secrets/2025-linebot-env/versions/latest", GCPProjectID),
	}
	result, err := client.AccessSecretVersion(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to access secret version: %w", err)
	}

	// save the secret as a .env file
	os.WriteFile(".env", result.Payload.Data, 0o444)
	return nil
}
