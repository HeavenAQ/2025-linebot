package db_test

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

var firestoreClient *db.FirestoreClient

// setup database
func TestMain(m *testing.M) {
	if os.Getenv("RUN_LIVE_FIRESTORE") != "1" {
		log.Println("Skipping Firestore live tests; set RUN_LIVE_FIRESTORE=1 to enable.")
		os.Exit(0)
	}
	cfg, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load Firestore integration-test config: ", err)
	}
	firestoreClient, err = db.NewFirestoreClient(
		cfg.GCP.ProjectID,
		cfg.GCP.Database.DataDB,
		cfg.GCP.Database.SessionDB,
	)
	if err != nil {
		log.Fatal("Failed to initialize Firestore integration client: ", err)
	}
	code := m.Run()
	if err := firestoreClient.Client.Close(); err != nil {
		log.Printf("Failed to close Firestore integration client: %v", err)
		if code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}
