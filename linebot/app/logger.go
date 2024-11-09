package app

import "log"

type Logger struct {
	Info  *log.Logger
	Warn  *log.Logger
	Error *log.Logger
}

func NewLogger() *Logger {
	infoLogger := log.New(log.Writer(), "[INFO] ", log.LstdFlags|log.Lshortfile)
	errorLogger := log.New(log.Writer(), "[ERROR] ", log.LstdFlags|log.Lshortfile)
	warnLogger := log.New(log.Writer(), "[WARN] ", log.LstdFlags|log.Lshortfile)
	return &Logger{
		Info:  infoLogger,
		Warn:  warnLogger,
		Error: errorLogger,
	}
}
