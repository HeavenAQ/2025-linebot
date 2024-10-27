package app

import (
	"errors"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func (app *App) handlePostbackEvent(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	// Ignore the event if the postback data is for switching the menu
	if event.Postback.Data == "switch-to-main" ||
		event.Postback.Data == "switch-to-secondary" {
		app.Logger.Info.Printf("Menu switch event received. User ID: %v", event.Source.UserID)
		return
	}

	app.Logger.Info.Printf("Postback event received. User ID: %v", event.Source.UserID)
	app.handleUserState(event, user, session, event.ReplyToken)
}

func (app *App) handleUserState(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	var rawData string
	if event.Type == linebot.EventTypePostback {
		rawData = event.Postback.Data
	}

	switch session.UserState {
	case db.WritingNotes:
		// Process notes
	case db.ChattingWithGPT:
		// Process chat
	case db.ViewingExpertVideos:
		app.handleViewingExpertVideos(event, rawData, user, session, replyToken)
	case db.ViewingPortfoilo:
		app.handleViewingPortfolio(event, rawData, user, session, replyToken)
	case db.AnalyzingVideo:
		app.handleAnalyzingVideoActions(event, rawData, user, session, replyToken)
	default:
	}
}

func (app *App) handleViewingExpertVideos(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		session.ActionStep = db.SelectingHandedness
		app.handleSelectingSkill(
			event,
			session,
			rawData,
			replyToken,
			app.LineBot.PromptHandednessSelection,
		)
	case db.SelectingHandedness:
		app.handleSendingExpertVideos(event, session, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

func (app *App) handleViewingPortfolio(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleSelectingSkill(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	err = app.LineBot.SendPortfolio(event, user, db.SkillStrToEnum(data.Skill), session.UserState, "以下為您的學習歷程：", false)
	if err != nil {
		app.handleSendPortfolioError(err, replyToken)
		return
	}

	err = app.FirestoreClient.ResetSession(user.ID)
	if err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

func (app *App) handleAnalyzingVideoActions(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		session.ActionStep = db.UploadingVideo
		app.handleSelectingSkill(
			event,
			session,
			rawData,
			replyToken,
			app.LineBot.PromptUploadVideo,
		)
	case db.UploadingVideo:
		app.handleUploadingVideo(event, session, user, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// Helper functions

func (app *App) handleSelectingSkill(
	event *linebot.Event,
	session *db.UserSession,
	rawData,
	replyToken string,
	nextStepFunc func(*linebot.Event) error,
) {
	data, err := app.LineBot.HandleSelectingSkill(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	err = nextStepFunc(event)
	if err != nil {
		app.handleVideoUploadPromptError(err, replyToken)
		return
	}

	session.Skill = data.Skill
	err = app.FirestoreClient.UpdateUserSession(
		event.Source.UserID,
		*session,
	)
	if err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

func (app *App) handleUploadingVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	videoContent, err := app.getVideoContent(event, user.ID)
	if err != nil {
		app.handleGetVideoError(err, replyToken)
		return
	}

	thumbnailPath, err := app.createVideoThumbnail(event, user, videoContent)
	if err != nil {
		app.handleThumbnailCreationError(err, replyToken)
		return
	}

	app.uploadVideoContent(event, user, session, videoContent, thumbnailPath, replyToken)
}

func (app *App) getVideoContent(event *linebot.Event, userID string) ([]byte, error) {
	videoMsg, ok := event.Message.(*linebot.VideoMessage)
	if !ok {
		app.Logger.Warn.Println("Non-video message received")
		app.FirestoreClient.ResetSession(userID)
		return nil, errors.New("non-video message")
	}
	return app.LineBot.GetVideoContent(videoMsg.ID)
}

func (app *App) uploadVideoContent(event *linebot.Event, user *db.UserData, session *db.UserSession, videoContent []byte, thumbnailPath, replyToken string) {
	today := time.Now().Format("2006-01-02-15-04")
	video, thumbnail, err := app.uploadVideoToDrive(user, session, videoContent, thumbnailPath, today)
	if err != nil {
		app.handleUploadToDriveError(err, replyToken)
		return
	}

	err = app.updateUserPortfolioVideo(user, session, today, video, thumbnail)
	if err != nil {
		app.handleUpdateUserPortfolioError(err, replyToken)
		return
	}

	err = app.sendVideoUploadedReply(event, session, user)
	if err != nil {
		app.handleSendingReplyMessageError(err, replyToken)
		return
	}
	app.FirestoreClient.ResetSession(user.ID)
}

func (app *App) handleMessageResponse(res *linebot.BasicResponse, err error, replyToken string) {
	if err != nil {
		app.Logger.Warn.Println("Error sending message:", err)
		_, sendErr := app.LineBot.SendDefaultErrorReply(replyToken)
		if sendErr != nil {
			app.Logger.Warn.Println("Error sending default error reply:", sendErr)
		}
		return
	}
	app.Logger.Info.Println("Message sent successfully. Response from LINE:", res)
}

func (app *App) handleInvalidActionStep(userID string, replyToken string) {
	app.FirestoreClient.ResetSession(userID)
	_, err := app.LineBot.SendDefaultErrorReply(replyToken)
	if err != nil {
		app.Logger.Warn.Println("Error sending default error reply:", err)
		return
	}
}

func (app *App) handleSendingExpertVideos(event *linebot.Event, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleSelectingHandedness(event.Postback.Data)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	handedness, err := db.HandednessStrToEnum(data.Handedness)
	skill := db.SkillStrToEnum(session.Skill)
	err = app.LineBot.SendExpertVideos(handedness, skill, replyToken)
	if err != nil {
		app.handleSendExpertVideosError(err, replyToken)
		return
	}
}
