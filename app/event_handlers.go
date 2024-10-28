package app

import "github.com/line/line-bot-sdk-go/v7/linebot"

func (app *App) handleEvents(events []*linebot.Event) {
	for _, event := range events {
		user := app.createUserIfNotExist(event.Source.UserID)
		session := app.createUserSessionIfNotExist(event.Source.UserID)

		switch event.Type {
		case linebot.EventTypeFollow:
			app.handleFollowEvent(event)
		case linebot.EventTypeMessage:
			app.handleMessageEvent(event, user, session)
		case linebot.EventTypePostback:
			app.handlePostbackEvent(event, user, session)
		default:
			app.handleUnsupportedEvent(event)
		}
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
