package db_test

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
	"github.com/stretchr/testify/require"
)

func setupFirestoreClient(t *testing.T) *db.FirestoreClient {
	// Fetch credentials from Secret Manager
	secretName := secret.GetSecretString(cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)
	credentials, err := secret.AccessSecretVersion(secretName)
	require.NoError(t, err)
	require.NotNil(t, credentials)

	// Initialize Firestore client using NewFirestoreClient
	client, err := db.NewFirestoreClient(
		credentials,
		cfg.GCP.ProjectID,
		cfg.GCP.Database.DatabaseID,
		cfg.GCP.Database.DataDB,
		cfg.GCP.Database.SessionDB,
	)
	require.NoError(t, err)
	return client
}

var (
	cfg             *config.Config
	firestoreClient *db.FirestoreClient
)

// setup database
func TestMain(m *testing.M) {
	conf, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load configurations")
	}
	cfg = conf
	firestoreClient = setupFirestoreClient(&testing.T{})

	os.Exit(m.Run())
}
