package app

import (
	"errors"
	"net/http"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (app *App) LineWebhookHandler() http.HandlerFunc {
	return func(writer http.ResponseWriter, req *http.Request) {
		events, err := app.LineBot.ParseRequest(req)
		if err != nil {
			app.handleParseError(err, writer)
			return
		}
		app.handleEvents(events)
	}
}

func (app *App) handleParseError(err error, writer http.ResponseWriter) {
	if errors.Is(err, linebot.ErrInvalidSignature) {
		app.Logger.Warn.Println("Invalid signature")
		writer.WriteHeader(http.StatusBadRequest)
		return
	}
	app.Logger.Error.Println("Error parsing request:", err)
	writer.WriteHeader(http.StatusInternalServerError)
}
