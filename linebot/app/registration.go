package app

import (
	"fmt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (app *App) registrationReply(event *linebot.Event, message string) {
	if event.ReplyToken == "" {
		return
	}
	_, err := app.LineBot.SendReply(event.ReplyToken, message)
	handleLineMessageResponseError(err)
}

// This gate runs before session changes, video downloads, GPT calls, rich-menu
// handlers, or postbacks. A successful registration consumes the message.
func (app *App) registeredEventUser(event *linebot.Event) (*db.UserData, bool) {
	return gateExperimentEvent(event, app.FirestoreClient.GetUserData, app.createUser,
		app.FirestoreClient.RegisterExperiment, app.registrationReply)
}

func gateExperimentEvent(event *linebot.Event,
	lookup func(string) (*db.UserData, error),
	create func(string) *db.UserData,
	save func(string, db.ExperimentRegistration) (*db.UserData, error),
	reply func(*linebot.Event, string),
) (*db.UserData, bool) {
	if event == nil || event.Source == nil || event.Source.UserID == "" {
		return nil, false
	}
	if event.Type != linebot.EventTypeFollow && event.Type != linebot.EventTypeMessage &&
		event.Type != linebot.EventTypePostback {
		return nil, false
	}
	if event.Source.Type != linebot.EventSourceTypeUser {
		reply(event, "請在與機器人的一對一 LINE 聊天室完成登記及使用功能。")
		return nil, false
	}
	user, err := lookup(event.Source.UserID)
	if err != nil && status.Code(err) != codes.NotFound {
		reply(event, "暫時無法確認登記資料，請稍後再試。")
		return nil, false
	}
	if user.HasExperimentRegistration() {
		// Repeated LINE deliveries of the registration message are acknowledgments,
		// not reflection notes or GPT prompts containing the student's identity.
		if message, ok := event.Message.(*linebot.TextMessage); ok {
			parsed, parseErr := db.ParseExperimentRegistration(message.Text)
			if parseErr == nil && parsed.RealName == user.RealName &&
				parsed.ExperimentNumber == user.ExperimentNumber {
				reply(event, "已完成登記，可以使用下方選單。")
				return nil, false
			}
		}
		return user, true
	}
	message, textMessage := event.Message.(*linebot.TextMessage)
	if event.Type != linebot.EventTypeMessage || !textMessage {
		reply(event, db.RegistrationInstructions)
		return nil, false
	}
	registration, err := db.ParseExperimentRegistration(message.Text)
	if err != nil {
		reply(event, db.RegistrationFormatError)
		return nil, false
	}
	if user == nil {
		user = create(event.Source.UserID)
		if user == nil {
			reply(event, "登記尚未完成，請稍後重新傳送編號與姓名。")
			return nil, false
		}
	}
	user, err = save(event.Source.UserID, registration)
	if err != nil || !user.HasExperimentRegistration() {
		reply(event, "登記尚未完成或資料與已登記內容不同，請稍後重試或聯絡老師。")
		return nil, false
	}
	reply(event, fmt.Sprintf("登記完成！\n實驗編號：%s\n姓名：%s\n現在可以使用下方選單；若資料有誤，請聯絡老師更正。", user.ExperimentNumber, user.RealName))
	return nil, false
}
