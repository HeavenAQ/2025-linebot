package db_test

import (
	"context"
	"log"
	"os"
	"testing"

	"cloud.google.com/go/firestore"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
	"github.com/stretchr/testify/require"
	"google.golang.org/api/option"
)

func setupFirestoreClient(t *testing.T) *db.FirestoreClient {
	// Fetch credentials from Secret Manager
	secretName := secret.GetSecretString(cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)
	credentials, err := secret.AccessSecretVersion(secretName)
	require.NoError(t, err)
	require.NotNil(t, credentials)

	// Initialize Firestore client
	ctx := context.Background()
	client, err := firestore.NewClient(ctx, cfg.GCP.ProjectID, option.WithCredentialsJSON(credentials))
	require.NoError(t, err)
	return &db.FirestoreClient{
		Ctx:      &ctx,
		Client:   client,
		Data:     client.Collection(cfg.GCP.Database.DataDB),
		Sessions: client.Collection(cfg.GCP.Database.SessionDB),
	}
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
