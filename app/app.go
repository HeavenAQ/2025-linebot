package app

import "github.com/HeavenAQ/config"

type App struct {
	Config *config.Config
	Logger *Logger
}

func NewApp() *App {
	cfg, err := config.LoadConfig()
	if err != nil {
		panic(err)
	}
	return &App{
		Config: cfg,
		Logger: NewLogger(),
	}
}
