package app

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (app *App) handleMessageEvent(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	message, ok := event.Message.(*linebot.TextMessage)
	if !ok {
		app.handleNonTextMessage(event, session, user)
		return
	}
	app.handleTextMessage(event, message, user, session)
}

func (app *App) handleNonTextMessage(event *linebot.Event, session *db.UserSession, user *db.UserData) {
	if _, ok := event.Message.(*linebot.VideoMessage); ok {
		app.Logger.Info.Println("Video message received")
		app.handleUploadingVideo(event, session, user, event.ReplyToken)
	} else {
		app.Logger.Warn.Println("The message type is not supported")
	}
}

func (app *App) handleTextMessage(event *linebot.Event, message *linebot.TextMessage, user *db.UserData, session *db.UserSession) {
	incomingState, err := db.UserStateChnStrToEnum(message.Text)
	if err != nil {
		app.Logger.Info.Println("Incoming message is not a rich menu message; handling as a reflection note")
		app.handleUserState(event, user, session, event.ReplyToken)
		return
	}
	app.handleRichMenuMessage(incomingState, user, session.UserState, event.ReplyToken)
}

func (app *App) handleUnsupportedMessage(replyToken string) {
	app.Logger.Warn.Println("Unsupported message type")
	_, err := app.LineBot.SendDefaultReply(replyToken)
	handleLineMessageResponseError(err)
}

func (app *App) handleRichMenuMessage(
	incomingState db.UserState,
	user *db.UserData,
	userState db.UserState,
	replyToken string,
) {
	switch incomingState {
	case db.ReadingInstruction:
		app.processReadingInstruction(user, replyToken)
	case db.ViewingPortfoilo:
		app.processViewingPortfolio(user, userState, replyToken)
	case db.ViewingExpertVideos:
		app.processViewingExpertVideos(user, userState, replyToken)
	case db.AnalyzingVideo:
		app.processAnalyzingVideo(user, userState, replyToken)
	case db.WritingNotes:
		app.processWritingNotes(user, userState, replyToken)
	case db.ChattingWithGPT:
		app.processChattingWithGPT(user, userState, replyToken)
	default:
		app.handleUnsupportedMessage(replyToken)
	}
}
