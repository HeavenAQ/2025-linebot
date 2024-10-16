package app

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

type App struct {
	Config          *config.Config
	Logger          *Logger
	LineBot         *line.LineBot
	FirestoreClient *db.FirestoreClient
	DriveClient     *storage.GoogleDriveClient
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
	if err != nil {
		panic(err)
	}

	// Set up firestore client
	firestoreClient, err := db.NewFirestoreClient(
		credentials,
		cfg.GCP.ProjectID,
		cfg.GCP.Database.DataDB,
		cfg.GCP.Database.SessionDB,
	)
	if err != nil {
		panic(err)
	}

	// Set up Google Drive client
	driveClient, err := storage.NewGoogleDriveClient(
		credentials,
		cfg.GCP.Storage.GoogleDrive.RootFolder,
	)
	if err != nil {
		panic(err)
	}

	return &App{
		Config:          cfg,
		Logger:          logger,
		LineBot:         lineBot,
		FirestoreClient: firestoreClient,
		DriveClient:     driveClient,
	}
}
