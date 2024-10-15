package main

import (
	"log"
	"net/http"

	"github.com/HeavenAQ/app"
)

func main() {
	app := app.NewApp(".env")
	http.HandleFunc("/callback", app.LineWebhookHandler())

	// Start the server
	app.Logger.Info.Println("\n\tServer started on port " + app.Config.Port)
	if err := http.ListenAndServe(":"+app.Config.Port, nil); err != nil {
		log.Fatal(err)
	}
}
