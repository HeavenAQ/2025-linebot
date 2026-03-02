package db_test

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
	"github.com/stretchr/testify/require"
)

func setupFirestoreClient(t *testing.T) *db.FirestoreClient {
	// Initialize Firestore client using NewFirestoreClient
	client, err := db.NewFirestoreClient(
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
		log.Println("Skipping Firestore integration tests: failed to load config")
		os.Exit(0)
	}
	cfg = conf
	firestoreClient = setupFirestoreClient(&testing.T{})

	os.Exit(m.Run())
}
