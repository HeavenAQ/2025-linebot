package app

import (
	"github.com/HeavenAQ/config"

	"github.com/HeavenAQ/api/db"
	"github.com/HeavenAQ/api/line"
	"github.com/HeavenAQ/api/secret"
	"github.com/HeavenAQ/api/storage"
)

type App struct {
	Config          *config.Config
	Logger          *Logger
	LineBot         line.LineBotClient
	FirestoreClient *db.FirestoreClient
	DriveClient     *storage.GoogleDriveClient
}

func getGCPCredentials() {
}

func NewApp(configPath string) *App {
	// Set up logger
	logger := NewLogger()

	// load the configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		panic(err)
	}

	// Set up the LineBot client
	lineBot, err := line.NewBotClient(cfg.Line.ChannelSecret, cfg.Line.ChannelToken)
	if err != nil {
		panic(err)
	}

	// Set up secret manager
	secretName := secret.GetSecretString(cfg.GCP.ProjectID, cfg.GCP.Credentials, cfg.GCP.Secrets.SecretVersion)
	credentials, err := secret.AccessSecretVersion(secretName)

	// Set up firestore client
	firestoreClient, err := db.NewFirestoreClient(
		credentials,
		cfg.GCP.ProjectID,
		cfg.GCP.Database.DataDB,
		cfg.GCP.Database.SessionDB,
	)

	// Set up Google Drive client
	driveClient, err := storage.NewGoogleDriveClient(
		credentials,
		cfg.GCP.Storage.GoogleDrive.RootFolder,
	)

	return &App{
		Config:          cfg,
		Logger:          logger,
		LineBot:         lineBot,
		FirestoreClient: firestoreClient,
		DriveClient:     driveClient,
	}
}
