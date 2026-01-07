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
)

func setupFirestoreClient(t *testing.T) *db.FirestoreClient {
	// Fetch credentials from Secret Manager
	err := secret.DownloadEnvFile()
	require.NoError(t, err)

	// Initialize Firestore client
	ctx := context.Background()
	client, err := firestore.NewClient(ctx, cfg.GCP.ProjectID)
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
    // Skip Firestore live tests unless explicitly enabled
    if os.Getenv("RUN_LIVE_FIRESTORE") != "1" {
        log.Println("Skipping Firestore live tests; set RUN_LIVE_FIRESTORE=1 to enable.")
        os.Exit(0)
    }
    conf, err := config.LoadConfig("../../.env")
    if err != nil {
        log.Fatal("Failed to load configurations")
    }
    cfg = conf
    firestoreClient = setupFirestoreClient(&testing.T{})

    os.Exit(m.Run())
}
