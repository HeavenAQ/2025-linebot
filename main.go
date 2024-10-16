package main

import (
	"log"
	"net/http"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/app"
)

func main() {
	app := app.NewApp(".env")
	http.HandleFunc("/callback", app.LineWebhookHandler())

	// Default time duration
	const (
		DefaultReadTimeout  = 100 * time.Second
		DefaultWriteTimeout = 100 * time.Second
		DefaultIdleTimeout  = 120 * time.Second
	)

	// Create a new server
	server := &http.Server{
		Addr:         ":" + app.Config.Port,
		Handler:      http.DefaultServeMux,
		ReadTimeout:  DefaultReadTimeout,
		WriteTimeout: DefaultWriteTimeout,
		IdleTimeout:  DefaultIdleTimeout,
	}
	// Log the server start
	app.Logger.Info.Println("\n\tServer started on port " + app.Config.Port)

	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
