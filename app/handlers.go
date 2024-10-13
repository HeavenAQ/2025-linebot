package app

import (
	"net/http"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (app *App) LineWebhookHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Verify the incoming request
		events, err := app.LineBot.ParseRequest(r)
		if err != nil {
			if err == linebot.ErrInvalidSignature {
				app.Logger.Warn.Println("invalid signature")
				w.WriteHeader(http.StatusBadRequest)
			} else {
				app.Logger.Error.Println(err)
				w.WriteHeader(http.StatusInternalServerError)
			}
		}

		// Process the events from the request
		for _, event := range events {
			switch event.Type {
			case linebot.EventTypeMessage:
				// Handle the message event
			case linebot.EventTypeFollow:
				// Handle the follow event
			default:
				// Log a warning and reply with a default message
				app.Logger.Warn.Printf("unsupported event type: %s", event.Type)
				app.LineBot.ReplyWithTypeError(event.ReplyToken)
			}
		}
	}
}
