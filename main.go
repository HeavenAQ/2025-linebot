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
	http.HandleFunc("/test", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Hello, World!"))
	})

	// Default time duration
	const (
		DefaultReadTimeout  = 100 * time.Second
		DefaultWriteTimeout = 100 * time.Second
		DefaultIdleTimeout  = 120 * time.Second
	)

	// Create a new server
	server := &http.Server{
		Addr:         "0.0.0.0:" + "8080",
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
