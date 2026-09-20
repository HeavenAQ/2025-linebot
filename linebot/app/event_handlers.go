package app

import (
	"context"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

const loadingSeconds = 60

func (app *App) handleEvents(events []*linebot.Event) {
	for _, event := range events {
		user, allowed := app.registeredEventUser(event)
		if !allowed {
			continue
		}
		session := app.createUserSessionIfNotExist(event.Source.UserID)
		if session == nil {
			app.registrationReply(event, "暫時無法開啟功能，請稍後再試。")
			continue
		}

		switch event.Type {
		case linebot.EventTypeFollow:
			app.handleFollowEvent(event)
		case linebot.EventTypeMessage:
			app.showLoading(event.Source.UserID)
			app.handleMessageEvent(event, user, session)
		case linebot.EventTypePostback:
			app.showLoading(event.Source.UserID)
			app.handlePostbackEvent(event, user, session)
		default:
			app.handleUnsupportedEvent(event)
		}
	}
}

func (app *App) showLoading(userID string) {
	if userID == "" {
		return
	}
	if err := app.LineBot.ShowLoading(context.Background(), userID, loadingSeconds); err != nil {
		app.Logger.Warn.Printf("loading indicator: %v", err)
	}
}

func (app *App) handleFollowEvent(event *linebot.Event) {
	app.Logger.Info.Printf("Follow event received. New user ID: %s", event.Source.UserID)
	res, err := app.LineBot.SendWelcomeReply(event)
	app.handleMessageResponseError(res, err, event.ReplyToken)
}

func (app *App) handleUnsupportedEvent(event *linebot.Event) {
	app.Logger.Warn.Printf("Unsupported event type: %s", event.Type)
	_, err := app.LineBot.SendDefaultErrorReply(event.ReplyToken)
	if err != nil {
		app.Logger.Warn.Println("Error sending type error message:", err)
	}
}
