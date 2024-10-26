package app

import (
	"errors"
	"net/http"
	"time"

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

		// If not none, handle actions based user's session state
		if session.UserState != db.None {
			app.Logger.Info.Println("Try to handle user's state")
			app.handleUserState(event, user, session, event.ReplyToken)
		}

		switch event.Type {
		case linebot.EventTypeFollow:
			app.Logger.Info.Printf("Follow event received. New user ID: %s", event.Source.UserID)
			res, err := app.LineBot.SendWelcomeReply(event)
			if err != nil {
				app.Logger.Warn.Println("Error sending welcome message: ", err)
				return
			}

			app.Logger.Info.Println("Welcome message sent. Response from line: ", res)
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

		case linebot.EventTypePostback:
			app.Logger.Info.Printf("Postback event received. User ID: %v", event.Source.UserID)
			app.handleUserState(event, user, session, event.ReplyToken)

		default:
			// Log a warning and reply with a default message
			app.Logger.Warn.Printf("unsupported event type: %s", event.Type)
			_, err := app.LineBot.SendTypeErrorReply(event.ReplyToken)
			if err != nil {
				app.Logger.Warn.Println("Error sending type error message: ", err)
				return
			}
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

	var errSessionUpdate error

	switch event.Text {
	case "使用說明":
		app.FirestoreClient.ResetSession(user.ID)
		res, err = app.LineBot.SendInstruction(replyToken)
	case "學習歷程":
		app.FirestoreClient.ResetSession(user.ID)
		errSessionUpdate = app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingPortfoilo, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要查看的動作")
	case "專家影片":
		app.FirestoreClient.ResetSession(user.ID)
		errSessionUpdate = app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingExpertVideos, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要觀看的動作")
	case "動作分析":
		app.FirestoreClient.ResetSession(user.ID)

		errSessionUpdate = app.FirestoreClient.UpdateSessionUserState(user.ID, db.AnalyzingVideo, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要分析的動作")
	case "預習及反思":
		app.FirestoreClient.ResetSession(user.ID)
		errSessionUpdate = app.FirestoreClient.UpdateSessionUserState(user.ID, db.WritingNotes, db.SelectingSkill)

		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要記錄的動作")
	case "GPT對談":
		app.FirestoreClient.ResetSession(user.ID)
		errSessionUpdate = app.FirestoreClient.UpdateSessionUserState(user.ID, db.ChattingWithGPT, db.SelectingSkill)
		res, err = app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要對談的動作")
	default:
		app.FirestoreClient.ResetSession(user.ID)
		res, err = app.LineBot.ReplyMessage(replyToken, linebot.NewTextMessage("抱歉，您所輸入的訊息格式目前並未支援，請重試一次！"))
	}

	if err != nil || errSessionUpdate != nil {
		_, err = app.LineBot.SendDefaultErrorReply(replyToken)
		if err != nil {
			app.Logger.Warn.Println("Error sending type error message: ", err)
			return
		}

		app.Logger.Warn.Println("Error sending instruction: ", err)
		return
	}

	app.Logger.Info.Println("Instruction sent. Response from line: ", res)
}

func (app *App) handleUserState(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	// Try to get the raw data
	var rawData string
	if event.Type == linebot.EventTypePostback {
		rawData = event.Postback.Data
	}

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
		switch session.ActionStep {
		case db.SelectingSkill:
			app.handleSelectingSkill(event, user, rawData, replyToken)
		case db.UploadingVideo:
			app.handleUploadingVideo(event, session, user, replyToken)
		default:
			app.handleInvalidActionStep(user.ID, replyToken)
		}
	case db.None:
	default:
	}
}

func (app *App) handleViewingPortfolio(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	// Get the user's portfolio
	data, err := app.LineBot.HandleSelectingSkill(rawData)
	app.handlePostbackDataTypeError(err, replyToken)

	// Send the user's portfolio
	err = app.LineBot.SendPortfolio(
		event,
		user,
		db.SkillStrToEnum(data.Skill),
		session.UserState,
		"以下為您的學習歷程：",
		false,
	)
	app.handleSendingReplyMessageError(err, replyToken)
}

func (app *App) handleSelectingSkill(event *linebot.Event, user *db.UserData, rawData string, replyToken string) {
	// Unmarshal raw data for SelectingSkill action
	data, err := app.LineBot.HandleSelectingSkill(rawData)
	app.handlePostbackDataTypeError(err, replyToken)

	// Prompt user to upload video
	err = app.LineBot.PromptUploadVideo(event)
	app.handleVideoUploadPromptError(err, replyToken)

	// Update the skill session and move on to the next step
	err = app.FirestoreClient.UpdateUserSession(
		user.ID,
		db.UserSession{
			Skill:      data.Skill,
			UserState:  db.AnalyzingVideo,
			ActionStep: db.UploadingVideo,
		})
	app.handleUpdateSessionError(err, replyToken)
}

func (app *App) handleUploadingVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	// Type assertion to get the video message
	videoMsg, ok := event.Message.(*linebot.VideoMessage)
	if !ok {
		app.Logger.Warn.Println("Non-video message received")
		app.FirestoreClient.ResetSession(user.ID)
		_, err := app.LineBot.SendDefaultErrorReply(replyToken)
		app.handleGetVideoError(err, replyToken)
		return
	}

	// Get the video content
	app.Logger.Info.Println("Getting video content")
	videoContent, err := app.LineBot.GetVideoContent(videoMsg.ID)
	app.handleGetVideoError(err, replyToken)

	// Create video thumbnail
	app.Logger.Info.Println("Creating video thumbnail")
	thumbnailPath, err := app.createVideoThumbnail(event, user, videoContent)
	app.handleThumbnailCreationError(err, replyToken)

	// Upload the video to Google Drive
	today := time.Now().Format("2006-01-02-15-04")
	app.Logger.Info.Println("Uploading video to Google Drive")
	video, thumbnail, err := app.uploadVideoToDrive(user, session, videoContent, thumbnailPath, today)
	app.handleUploadToDriveError(err, replyToken)

	// Update user portfolio
	app.Logger.Info.Println("Updating user portfolio")
	err = app.updateUserPortfolioVideo(user, session, today, video, thumbnail)
	app.handleUpdateUserPortfolioError(err, replyToken)

	// Send a success message
	err = app.sendVideoUploadedReply(event, session, user)
	app.handleSendingReplyMessageError(err, replyToken)

	// Reset the user session
	err = app.FirestoreClient.ResetSession(user.ID)
	app.handleUpdateSessionError(err, replyToken)
}

// Function to handle invalid action steps
func (app *App) handleInvalidActionStep(userID string, replyToken string) {
	app.FirestoreClient.ResetSession(userID)
	_, err := app.LineBot.SendDefaultErrorReply(replyToken)
	if err != nil {
		app.Logger.Warn.Println("Error sending default error reply:", err)
	}
}
