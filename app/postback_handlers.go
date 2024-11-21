package app

import (
	"errors"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

// handlePostbackEvent processes LINE postback events. If the event is for menu switching, it is ignored.
func (app *App) handlePostbackEvent(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	if isMenuSwitchEvent(event.Postback.Data) {
		app.Logger.Info.Printf("Menu switch event ignored. User ID: %v", event.Source.UserID)
		return
	}

	app.Logger.Info.Printf("Postback event received. User ID: %v", event.Source.UserID)
	app.handleUserState(event, user, session, event.ReplyToken)
}

// isMenuSwitchEvent checks if a postback data string corresponds to a menu-switching action.
func isMenuSwitchEvent(data string) bool {
	return data == "switch-to-main" || data == "switch-to-secondary"
}

// handleUserState manages the user's session state and delegates the processing to appropriate handlers based on the session state.
func (app *App) handleUserState(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	rawData := getPostbackData(event)
	app.Logger.Info.Println("rawData: ", rawData)

	// Handle GPT stop chatting action as a special case
	if data, ok := app.isStopChattingWithGPTAction(rawData); ok {
		if data.Stop {
			app.FirestoreClient.ResetSession(user.ID)
			app.LineBot.SendReply(replyToken, "已結束對話")
			return
		}
	}

	// Handle note updating action as a special case
	if data, ok := app.isUpdateNoteAction(rawData); ok {
		app.forceStateToWritingNotes(user, session, data, replyToken)
		return
	}

	// Handle video watching action as a special case
	if data, ok := app.isWatchVideoAction(rawData); ok {
		app.LineBot.SendVideoMessage(replyToken, data)
		return
	}

	// Route the event handling based on the current user session state
	switch session.UserState {
	case db.WritingNotes:
		app.handleWritingNotes(event, rawData, user, session, replyToken)
	case db.ChattingWithGPT:
		app.handleChattingWithGPT(event, user, replyToken)
	case db.ViewingExpertVideos:
		app.handleViewingExpertVideos(event, rawData, user, session, replyToken)
	case db.ViewingPortfoilo:
		app.handleViewingPortfolio(event, rawData, user, session, replyToken)
	case db.UploadingVideo:
		app.handleAnalyzingVideoActions(event, rawData, user, session, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// getPostbackData extracts postback data from the event if available.
func getPostbackData(event *linebot.Event) string {
	if event.Type == linebot.EventTypePostback {
		return event.Postback.Data
	}
	return ""
}

// isUpdateNoteAction checks if the postback event indicates an update note action.
// If true, it updates the session accordingly and sends a response to the user.
func (app *App) isUpdateNoteAction(rawData string) (*line.WritingNotePostback, bool) {
	data, err := app.LineBot.HandleWritingNotePostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

// isWatchVideoAction checks if the postback event indicates a watch video action.
// If true, it sends a video message to the user.
func (app *App) isWatchVideoAction(rawData string) (*line.VideoPostback, bool) {
	data, err := app.LineBot.HandleVideoPostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

// isStopChattingWithGPTAction checks if the postback event indicates a GPT chatting action.
// If true, it hands the control over to the GPT chat handler.
func (app *App) isStopChattingWithGPTAction(rawData string) (*line.StopGPTPostback, bool) {
	data, err := app.LineBot.HandleStopGPTPostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

func (app *App) forceStateToWritingNotes(user *db.UserData, session *db.UserSession, data *line.WritingNotePostback, replyToken string) {
	// Determine the action step and update the session state
	actionStep, err := db.ActionStepStrToEnum(data.ActionStep)
	if err != nil {
		app.Logger.Warn.Println("Invalid action step for updating note")
		app.FirestoreClient.ResetSession(user.ID)
	}

	session.ActionStep, session.UpdatingDate, session.UserState = actionStep, data.WorkDate, db.WritingNotes
	session.Skill = data.Skill
	app.FirestoreClient.UpdateUserSession(user.ID, *session)

	// Generate and send a response message to the user based on the action step
	msg := generateUpdateNoteMessage(data.WorkDate, data.Skill)
	app.LineBot.SendReply(replyToken, msg)
}

// generateUpdateNoteMessage creates a message for updating a note, specifying the date and skill.
func generateUpdateNoteMessage(workDate, skill string) string {
	skillStr := db.SkillStrToEnum(skill).ChnString()
	return "請輸入【" + workDate + "】的【" + skillStr + "】的學習反思"
}

// Handlers for different user states

// handleViewingExpertVideos manages the flow for viewing expert videos, advancing action steps as necessary.
func (app *App) handleViewingExpertVideos(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		session.ActionStep = db.SelectingHandedness
		app.handleSelectingSkill(event, session, rawData, replyToken, app.LineBot.PromptHandednessSelection)
	case db.SelectingHandedness:
		app.handleSendingExpertVideos(event, session, replyToken)
		app.resetSessionWithErrorHandling(user.ID, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// handleViewingPortfolio handles viewing portfolio actions and resets the session after completion.
func (app *App) handleViewingPortfolio(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	if err := app.LineBot.SendPortfolio(event, user, db.SkillStrToEnum(data.Skill), session.UserState, "以下為您的學習歷程：", false); err != nil {
		app.handleSendPortfolioError(err, replyToken)
		return
	}

	app.resetSessionWithErrorHandling(user.ID, replyToken)
}

// handleAnalyzingVideoActions handles actions for analyzing videos and updates session state as needed.
func (app *App) handleAnalyzingVideoActions(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		session.ActionStep = db.SelectingVideoUploadMethod
		app.handleSelectingSkill(event, session, rawData, replyToken, app.LineBot.PromptUploadVideo)
	case db.SelectingVideoUploadMethod:
		app.handleUploadingVideo(event, session, user, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// Helper functions

// handleSelectingSkill facilitates the skill selection process and moves to the next step.
func (app *App) handleSelectingSkill(
	event *linebot.Event,
	session *db.UserSession,
	rawData,
	replyToken string,
	nextStepFunc func(*linebot.Event) error,
) {
	data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
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
	err = app.FirestoreClient.UpdateUserSession(event.Source.UserID, *session)
	if err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleUploadingVideo processes video uploads, creates thumbnails, and updates the user portfolio.
func (app *App) handleUploadingVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	// Get video content
	videoContent, err := app.getVideoContent(event, user.ID)
	if err != nil {
		app.handleGetVideoError(err, replyToken)
		return
	}

	// Cteate thumbnail
	thumbnailPath, err := app.createVideoThumbnail(event, user, videoContent)
	if err != nil {
		app.handleThumbnailCreationError(err, replyToken)
		return
	}

	// Upload video to Google Drive and update user portfolio
	app.uploadVideoContent(event, user, session, videoContent, thumbnailPath, replyToken)
}

// getVideoContent retrieves the video content if the event includes a video message.
func (app *App) getVideoContent(event *linebot.Event, userID string) ([]byte, error) {
	videoMsg, ok := event.Message.(*linebot.VideoMessage)
	if !ok {
		app.Logger.Warn.Println("Non-video message received")
		app.FirestoreClient.ResetSession(userID)
		return nil, errors.New("non-video message")
	}
	return app.LineBot.GetVideoContent(videoMsg.ID)
}

// uploadVideoContent uploads video content and thumbnail to Google Drive, then updates the user portfolio.
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

// resetSessionWithErrorHandling resets the user session and handles any errors encountered.
func (app *App) resetSessionWithErrorHandling(userID, replyToken string) {
	if err := app.FirestoreClient.ResetSession(userID); err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleInvalidActionStep manages cases where the user session has an unexpected action step.
func (app *App) handleInvalidActionStep(userID string, replyToken string) {
	app.FirestoreClient.ResetSession(userID)
	_, err := app.LineBot.SendDefaultErrorReply(replyToken)
	if err != nil {
		app.Logger.Warn.Println("Error sending default error reply:", err)
	}
}

func (app *App) handleWritingNotes(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		// Transition to SelectingPortfolio step after choosing a skill
		session.ActionStep = db.SelectingPortfolio
		data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
		if err != nil {
			app.handlePostbackDataTypeError(err, replyToken)
			return
		}

		// Update session with selected skill and prompt user to select a portfolio
		session.Skill = data.Skill
		err = app.FirestoreClient.UpdateUserSession(user.ID, *session)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return
		}

		// Prompt user to select a portfolio based on chosen skill
		err = app.LineBot.SendPortfolio(event, user, db.SkillStrToEnum(data.Skill), session.UserState, "請選擇您要更新的學習歷程：", true)
		if err != nil {
			app.handleSendPortfolioError(err, replyToken)
			return
		}

	case db.SelectingPortfolio:
		// Handle the selection of a portfolio for note updating
		app.handleSelectingPortfolio(rawData, user, session, replyToken)

	default:
		// Handle unexpected action steps in WritingNotes state
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// handleChattingWithGPT handles the GPT chatting action.
func (app *App) handleChattingWithGPT(event *linebot.Event, user *db.UserData, replyToken string) {
	// get message from event
	var msg string
	message, ok := event.Message.(*linebot.TextMessage)
	if ok {
		msg = message.Text
	}
	// Add message to the GPT thread
	err := app.GPTClient.AddMessageToThread(user.GPTThreadIDs.Strategy, msg)
	if err != nil {
		app.handleAddMessageToGPTThreadError(err, replyToken)
		return
	}

	// Run the GPT thread
	runID, err := app.GPTClient.RunThread(user.GPTThreadIDs.Strategy)
	if err != nil {
		app.handleGPTRunThreadError(err, replyToken)
		return
	}

	// Retrieve the assistant's response
	response, err := app.GPTClient.GetAssistantResponse(user.GPTThreadIDs.Strategy, runID)
	if err != nil {
		app.handleGetGPTResponseError(err, replyToken)
		return
	}

	// Send a message to the user to start chatting with GPT
	_, err = app.LineBot.SendGPTChattingModeReply(replyToken, response)
	if err != nil {
		handleLineMessageResponseError(err)
		return
	}
}

// handleSelectingPortfolio processes the portfolio selection during note writing
func (app *App) handleSelectingPortfolio(rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleWritingNotePostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	// Update the action step and session with the selected portfolio details
	actionStep, err := db.ActionStepStrToEnum(data.ActionStep)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}
	session.ActionStep = actionStep
	session.UpdatingDate = data.WorkDate

	// Save the updated session
	err = app.FirestoreClient.UpdateUserSession(user.ID, *session)
	if err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleUpdatingNote handles the actual note update for either preview or reflection notes
func (app *App) handleUpdatingNote(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	// Verify the correct action step for updating the note
	if session.ActionStep != db.WritingReflection {
		app.Logger.Warn.Println("Invalid action step for updating note")
		app.handleInvalidActionStep(user.ID, event.ReplyToken)
		return
	}

	// Retrieve the text message content for the note update
	note, ok := event.Message.(*linebot.TextMessage)
	if !ok {
		app.Logger.Warn.Println("Non-text message received when updating note")
		app.FirestoreClient.ResetSession(user.ID)
		return
	}

	// Update the note in the user portfolio based on the action step (preview or reflection)
	portfolio := user.Portfolio.GetSkillPortfolio(session.Skill)
	app.FirestoreClient.UpdateUserPortfolioReflection(
		user,
		&portfolio,
		session.UpdatingDate,
		note.Text,
	)

	// Send a confirmation message to the user showing the updated portfolio
	app.LineBot.SendPortfolio(event, user, db.SkillStrToEnum(session.Skill), session.UserState, "以下為您的學習歷程：", false)
}

// handleSelectingHandedness processes the handedness selection for viewing expert videos
func (app *App) handleSendingExpertVideos(event *linebot.Event, session *db.UserSession, replyToken string) {
	// Parse handedness from the event's postback data
	data, err := app.LineBot.HandleSelectingHandednessPostbackData(event.Postback.Data)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	// Convert handedness and skill to their respective enums
	handedness, err := db.HandednessStrToEnum(data.Handedness)
	if err != nil {
		app.Logger.Warn.Println("Invalid handedness received:", data.Handedness)
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	skill := db.SkillStrToEnum(session.Skill)

	// Send expert videos based on handedness and skill selection
	err = app.LineBot.SendExpertVideos(handedness, skill, replyToken)
	if err != nil {
		app.handleSendExpertVideosError(err, replyToken)
		return
	}

	app.Logger.Info.Printf("Expert videos sent for User ID: %v, Skill: %v, Handedness: %v", event.Source.UserID, skill, handedness)
}
