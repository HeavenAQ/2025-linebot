package app

import (
	"github.com/HeavenAQ/config"

	"github.com/HeavenAQ/api/line"
)

type App struct {
	Config  *config.Config
	Logger  *Logger
	LineBot line.LineBotClient
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

	return &App{
		Config:  cfg,
		Logger:  logger,
		LineBot: lineBot,
	}
}
