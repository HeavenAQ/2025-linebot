package app

import (
	"context"
	"os"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
	"github.com/HeavenAQ/nstc-linebot-2025/api/analysisqueue"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

type App struct {
	AnalysisQueue   *analysisqueue.Client
	Config          *config.Config
	Logger          *Logger
	LineBot         *line.Client
	FirestoreClient *db.FirestoreClient
	StorageClient   *storage.BucketClient
	GPTClient       *gpt.Client
	AnalysisClient  *analysis.Client
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
	lineBot, err := line.NewBotClient(
		cfg.Line.ChannelSecret,
		cfg.Line.ChannelToken,
		cfg.GCP.Storage.BucketName,
		cfg.ReviewURL(),
	)
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
	gptClient := gpt.NewGPTClient(cfg.GPT.APIKey, cfg.GPT.Model)

	analysisClient, err := analysis.NewClient(
		cfg.AnalysisServer.Target,
		cfg.AnalysisServer.APIKey,
		cfg.AnalysisServer.Insecure,
		cfg.AnalysisServer.SkipCoaching,
		cfg.AnalysisServer.StoragePrefix,
	)
	if err != nil {
		panic(err)
	}

	var queue *analysisqueue.Client
	if cfg.AnalysisServer.TasksQueue != "" {
		queue, err = analysisqueue.New(context.Background(), cfg.AnalysisServer.TasksQueue, cfg.AnalysisServer.WorkerURL, cfg.AnalysisServer.TaskServiceAccount)
		if err != nil {
			panic(err)
		}
	}
	return &App{
		AnalysisQueue:   queue,
		Config:          cfg,
		Logger:          logger,
		LineBot:         lineBot,
		FirestoreClient: firestoreClient,
		StorageClient:   storageClient,
		GPTClient:       gptClient,
		AnalysisClient:  analysisClient,
	}
}

// Close releases the long-lived clients during graceful shutdown.
func (a *App) Close() {
	if a.AnalysisClient != nil {
		if err := a.AnalysisClient.Close(); err != nil {
			a.Logger.Warn.Printf("close analysis client: %v", err)
		}
	}
	if a.FirestoreClient != nil && a.FirestoreClient.Client != nil {
		if err := a.FirestoreClient.Client.Close(); err != nil {
			a.Logger.Warn.Printf("close firestore client: %v", err)
		}
	}
}
