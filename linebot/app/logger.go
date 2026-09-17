package app

import (
	"log"

	"github.com/HeavenAQ/nstc-linebot-2025/api/obs"
)

// Logger keeps the print-style API used across the app. Each line is written
// to stdout as a JSON entry with its severity and source location, which Cloud
// Run forwards to Cloud Logging as a structured log.
type Logger struct {
	Info  *log.Logger
	Warn  *log.Logger
	Error *log.Logger
}

func NewLogger() *Logger {
	// Anything still using the standard logger directly is structured too.
	log.SetFlags(log.Lshortfile)
	log.SetOutput(obs.NewStdLogger(obs.Info).Writer())
	return &Logger{
		Info:  obs.NewStdLogger(obs.Info),
		Warn:  obs.NewStdLogger(obs.Warning),
		Error: obs.NewStdLogger(obs.Error),
	}
}
