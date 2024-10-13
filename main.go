package main

import (
	"log"
	"net/http"
	"os"

	"github.com/HeavenAQ/app"
)

func main() {
	app := app.NewApp()
	http.HandleFunc("/callback", app.LineWebhookHandler())

	app.Logger.Info.Println("\n\tServer started on port " + os.Getenv("PORT"))
	if err := http.ListenAndServe(":"+os.Getenv("PORT"), nil); err != nil {
		log.Fatal(err)
	}
}
