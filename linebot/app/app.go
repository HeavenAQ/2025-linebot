package app

import (
	"os"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

type App struct {
	Config          *config.Config
	Logger          *Logger
	LineBot         *line.Client
	FirestoreClient *db.FirestoreClient
	StorageClient   *storage.BucketClient
	GPTClient       *gpt.Client
}

func NewApp(configPath string) *App {
	// Set up logger
	logger := NewLogger()
	testMode := os.Getenv("SKIP_EXTERNAL_CLIENTS") == "1"

	// Download env file only when not in test mode and when the config file does not exist locally
	if !testMode {
		if _, statErr := os.Stat(configPath); os.IsNotExist(statErr) {
			if err := secret.DownloadEnvFile(); err != nil {
				panic(err)
			}
		}
	}

	// load the configuration
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		panic(err)
	}

	// Set up the LineBot client
	lineBot, err := line.NewBotClient(cfg.Line.ChannelSecret, cfg.Line.ChannelToken, cfg.GCP.Storage.BucketName)
	if err != nil {
		panic(err)
	}

	// When in test mode, skip external clients (Firestore, Storage, GPT)
	if testMode {
		return &App{
			Config:  cfg,
			Logger:  logger,
			LineBot: lineBot,
		}
	}

	// Set up firestore client
	firestoreClient, err := db.NewFirestoreClient(
		cfg.GCP.ProjectID,
		cfg.GCP.Database.DataDB,
		cfg.GCP.Database.SessionDB,
	)
	if err != nil {
		panic(err)
	}

	// Set up Cloud Storage client
	storageClient, err := storage.NewBucketClient(
		cfg.GCP.Storage.BucketName,
	)
	if err != nil {
		panic(err)
	}

	// Set up GPT Client
	gptClient := gpt.NewGPTClient(cfg.GPT.APIKey, cfg.GPT.PromptID)

	return &App{
		Config:          cfg,
		Logger:          logger,
		LineBot:         lineBot,
		FirestoreClient: firestoreClient,
		StorageClient:   storageClient,
		GPTClient:       gptClient,
	}
}
