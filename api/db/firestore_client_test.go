package db

import (
	"context"
	"testing"

	"cloud.google.com/go/firestore"
	"github.com/HeavenAQ/api/secret"
	"github.com/stretchr/testify/require"
	"google.golang.org/api/option"
)

// TestFirestoreRealAccess tests real Firestore access using credentials from Google Secret Manager
func TestFirestoreRealAccess(t *testing.T) {
	// Replace with your actual project ID and the secret name from Google Secret Manager

	// Fetch credentials from Secret Manager
	secretName := secret.GetSecretString(cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)
	credentials, err := secret.AccessSecretVersion(secretName)
	require.NoError(t, err)
	require.NotNil(t, credentials)

	// Initialize Firestore client
	ctx := context.Background()
	client, err := firestore.NewClient(ctx, cfg.GCP.ProjectID, option.WithCredentialsJSON(credentials))
	require.NoError(t, err)
	defer client.Close()

	// Test data to be inserted
	testData := map[string]interface{}{
		"name":  "John Doe",
		"email": "johndoe@example.com",
	}

	// Test writing a document to Firestore
	docRef, _, err := client.Collection("users").Add(ctx, testData)
	require.NoError(t, err)

	// Ensure the data was written correctly by fetching it
	docSnapshot, err := docRef.Get(ctx)
	require.NoError(t, err)
	require.NotNil(t, docSnapshot)

	// Validate that the data matches what was inserted
	fetchedData := docSnapshot.Data()
	require.Equal(t, testData["name"], fetchedData["name"])
	require.Equal(t, testData["email"], fetchedData["email"])

	// Log the document's data for debugging
	t.Logf("User Document: %v\n", fetchedData)

	// Clean up by deleting the document
	_, err = docRef.Delete(ctx)
	require.NoError(t, err)
}
