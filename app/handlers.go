package app

import (
	"errors"
	"net/http"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (app *App) LineWebhookHandler() http.HandlerFunc {
	return func(writer http.ResponseWriter, req *http.Request) {
		// Verify the incoming request
		events, err := app.LineBot.ParseRequest(req)
		if err != nil {
			// If the error is due to an invalid signature and it's not in testing mode
			if errors.Is(err, linebot.ErrInvalidSignature) {
				app.Logger.Warn.Println("invalid signature")
				writer.WriteHeader(http.StatusBadRequest)
				return
			}

			// Log other errors
			app.Logger.Error.Println("error parsing request:", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		// process the incomging events
		app.processEvents(events)
	}
}

func (app *App) processEvents(events []*linebot.Event) {
	for _, event := range events {
		user := app.createUserIfNotExist(event.Source.UserID)
		session := app.createUserSessionIfNotExist(event.Source.UserID)

		switch event.Type {
		case linebot.EventTypeFollow:
			app.Logger.Info.Printf("Follow event received. New user ID: %s", event.Source.UserID)
			app.LineBot.SendWelcomeReply(event)
		case linebot.EventTypeMessage:
			// Example: handle a text message
			app.Logger.Info.Println("Text message received")
			message, ok := event.Message.(*linebot.TextMessage)
			if !ok {
				app.Logger.Warn.Println("Non-text message received")
				return
			}

			// If the user is in the None state, handle the message based on the riche menu
			if session.UserState == db.None {
				app.handleRichMenuMessage(
					message,
					user,
					session.UserState,
					event.ReplyToken,
				)
				return
			}

			// Handle the message based on the user's state
			app.handleUserState(event, user, session, event.ReplyToken)
		case linebot.EventTypePostback:
			app.Logger.Info.Println("Postback event received")
			app.handleUserState(event, user, session, event.ReplyToken)

		default:
			// Log a warning and reply with a default message
			app.Logger.Warn.Printf("unsupported event type: %s", event.Type)
			app.LineBot.ReplyWithTypeError(event.ReplyToken)
		}
	}
}

func (app *App) handleRichMenuMessage(
	event *linebot.TextMessage,
	user *db.UserData,
	userState db.UserState,
	replyToken string,
) {
	// Handle the rich menu message
	var res *linebot.BasicResponse
	var err error

	switch event.Text {
	case "使用說明":
		app.resetUserSession(user.ID)
		res, err = app.LineBot.SendInstruction(replyToken)
	case "學習歷程":
		app.resetUserSession(user.ID)
		app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingPortfoilo, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要查看的動作")
	case "專家影片":
		app.resetUserSession(user.ID)
		app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingExpertVideos, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要觀看的動作")
	case "分析影片":
		app.resetUserSession(user.ID)
		app.FirestoreClient.UpdateSessionUserState(user.ID, db.AnalyzingVideo, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要分析的動作")
	case "預習及反思":
		app.resetUserSession(user.ID)
		app.FirestoreClient.UpdateSessionUserState(user.ID, db.WritingNotes, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要記錄的動作")
	case "GPT對談":
		app.resetUserSession(user.ID)
		app.FirestoreClient.UpdateSessionUserState(user.ID, db.ChattingWithGPT, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要對談的動作")
	default:
		app.resetUserSession(user.ID)
		res, err = app.LineBot.ReplyMessage(replyToken, linebot.NewTextMessage("抱歉，您所輸入的訊息格式目前並未支援，請重試一次！"))
	}

	if err != nil {
		app.Logger.Warn.Println("Error sending instruction: ", err)
		return
	}

	app.Logger.Info.Println("Instruction sent. Response from line: ", res)
}

func (app *App) handleUserState(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.UserState {
	case db.WritingNotes:
		if session.ActionStep == db.SelectingSkill {
		}
	case db.ChattingWithGPT:
	case db.ChattingWithTeacher:
	case db.ViewingDashboard:
	case db.ViewingExpertVideos:
	case db.ViewingPortfoilo:
	case db.AnalyzingVideo:
	case db.None:
	default:
	}
}
