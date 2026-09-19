package app

import (
	"strings"

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

// This gate runs before session changes, video downloads, rich-menu handlers,
// or postbacks.
func (app *App) registeredEventUser(event *linebot.Event) (*db.UserData, bool) {
	return gateExperimentEvent(event, app.FirestoreClient.GetUserData,
		app.registrationReply, app.registrationLinkReply)
}

// Registration happens on the dashboard, so the bot's part is a link to it.
// Without a configured URL the card would carry an empty one, which LINE
// rejects outright -- leaving the learner with no reply at all -- so say it in
// words instead.
func (app *App) registrationLinkReply(event *linebot.Event) {
	if event.ReplyToken == "" {
		return
	}
	url, text := registrationInvite(app.Config.RegistrationURL())
	if url == "" {
		app.Logger.Error.Println("LIFF_REGISTRATION_URL is not set; replying without the registration button")
		app.registrationReply(event, text)
		return
	}
	_, err := app.LineBot.SendRegistrationLink(event.ReplyToken, url)
	handleLineMessageResponseError(err)
}

// registrationInvite picks how to invite a learner to register: the card with
// its button, or the instructions in words when no URL is configured.
func registrationInvite(configured string) (url, text string) {
	if strings.TrimSpace(configured) == "" {
		return "", db.RegistrationInstructions
	}
	return strings.TrimSpace(configured), ""
}

func gateExperimentEvent(event *linebot.Event,
	lookup func(string) (*db.UserData, error),
	reply func(*linebot.Event, string),
	replyRegistrationLink func(*linebot.Event),
) (*db.UserData, bool) {
	if event == nil || event.Source == nil || event.Source.UserID == "" {
		return nil, false
	}
	if event.Type != linebot.EventTypeFollow && event.Type != linebot.EventTypeMessage &&
		event.Type != linebot.EventTypePostback {
		return nil, false
	}
	if event.Source.Type != linebot.EventSourceTypeUser {
		reply(event, "請在與機器人的一對一 LINE 聊天室使用功能。")
		return nil, false
	}
	user, err := lookup(event.Source.UserID)
	if err != nil && status.Code(err) != codes.NotFound {
		reply(event, "暫時無法確認登記資料，請稍後再試。")
		return nil, false
	}
	if user.HasExperimentRegistration() {
		// A learner who registered in the chat before the form existed may
		// still send their number and name out of habit. That is an
		// acknowledgment, not a reflection note.
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
	// The details are typed into the web form, so the chat never carries a
	// learner's real name.
	replyRegistrationLink(event)
	return nil, false
}
