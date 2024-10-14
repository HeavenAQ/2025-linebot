package secret

import (
	"context"
	"fmt"

	secretmanager "cloud.google.com/go/secretmanager/apiv1"
	"cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

func AccessSecretVersion(secretName string) ([]byte, error) {
	ctx := context.Background()
	client, err := secretmanager.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create secret manager client %v", err)
	}
	defer client.Close()

	// access secret version
	req := &secretmanagerpb.AccessSecretVersionRequest{
		Name: secretName,
	}
	result, err := client.AccessSecretVersion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to access secret version: %v", err)
	}

	// return secret data
	return result.Payload.Data, nil
}

func GetSecretString(gcpProjectID, secretID, secretVersion string) string {
	res := fmt.Sprintf("projects/%s/secrets/%s/versions/%s", gcpProjectID, secretID, secretVersion)
	fmt.Println(res)
	return res
}
