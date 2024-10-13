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

func NewApp() *App {
	// load the configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		panic(err)
	}

	//
	return &App{
		Config: cfg,
		Logger: NewLogger(),
	}
}
