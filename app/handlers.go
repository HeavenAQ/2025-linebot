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
			// If the error is due to an invalid signature and it's not in testing mode
			if err == linebot.ErrInvalidSignature {
				app.Logger.Warn.Println("invalid signature")
				w.WriteHeader(http.StatusBadRequest)
				return // Stop further processing
			}

			// Log other errors
			app.Logger.Error.Println("error parsing request:", err)
			w.WriteHeader(http.StatusInternalServerError)
			return // Stop further processing
		}

		// Process the events from the request
		for _, event := range events {
			switch event.Type {
			case linebot.EventTypeMessage:
				// Example: handle a text message
				message := event.Message.(*linebot.TextMessage)
				if _, err := app.LineBot.ReplyMessage(event.ReplyToken, linebot.NewTextMessage("Received: "+message.Text)); err != nil {
					app.Logger.Error.Println("error sending reply:", err)
				}
				app.Logger.Info.Println("Received message:", message.Text)
				app.LineBot.ReplyMessage(event.ReplyToken, linebot.NewTextMessage("Received: "+message.Text))

			case linebot.EventTypeFollow:
				// Handle the follow event, e.g., welcome the user
				if _, err := app.LineBot.ReplyMessage(event.ReplyToken, linebot.NewTextMessage("Thanks for following!")); err != nil {
					app.Logger.Error.Println("error sending follow reply:", err)
				}

			default:
				// Log a warning and reply with a default message
				app.Logger.Warn.Printf("unsupported event type: %s", event.Type)
				app.LineBot.ReplyWithTypeError(event.ReplyToken)
			}
		}
	}
}
