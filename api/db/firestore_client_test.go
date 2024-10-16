package db_test

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestFirestoreRealAccess tests real Firestore access using credentials from Google Secret Manager
func TestFirestoreRealAccess(t *testing.T) {
	// Test data to be inserted
	testData := map[string]interface{}{
		"name":  "John Doe",
		"email": "johndoe@example.com",
	}

	// Test writing a document to Firestore
	docRef, _, err := firestoreClient.Data.Add(*firestoreClient.Ctx, testData)
	require.NoError(t, err)

	// Ensure the data was written correctly by fetching it
	docSnapshot, err := docRef.Get(*firestoreClient.Ctx)
	require.NoError(t, err)
	require.NotNil(t, docSnapshot)

	// Validate that the data matches what was inserted
	fetchedData := docSnapshot.Data()
	require.Equal(t, testData["name"], fetchedData["name"])
	require.Equal(t, testData["email"], fetchedData["email"])

	// Log the document's data for debugging
	t.Logf("User Document: %v\n", fetchedData)

	// Clean up by deleting the document
	_, err = docRef.Delete(*firestoreClient.Ctx)
	require.NoError(t, err)
}
