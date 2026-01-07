package app

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

// ============================================================================
// 1. High-Level Handlers
// ============================================================================

// handlePostbackEvent processes LINE postback events.
// - If it’s a menu-switch event, it’s ignored.
// - Otherwise, it delegates to handleUserState.
func (app *App) handlePostbackEvent(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	if isMenuSwitchEvent(event.Postback.Data) {
		app.Logger.Info.Printf("Menu switch event ignored. User ID: %v", event.Source.UserID)
		return
	}

	app.Logger.Info.Printf("Postback event received. User ID: %v", event.Source.UserID)
	app.handleUserState(event, user, session, event.ReplyToken)
}

// handleUserState manages the user's session state, routing to the appropriate
// handler function based on the session’s current state.
func (app *App) handleUserState(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	rawData := getPostbackData(event)
	app.Logger.Info.Println("rawData: ", rawData)

	// 1. GPT stop-chatting action
	if data, ok := app.isStopChattingWithGPTAction(rawData); ok {
		if data.Stop {
			app.FirestoreClient.ResetSession(user.ID)
			app.LineBot.SendReply(replyToken, "已結束對話")
			return
		}
	}

	// 2. Note updating action
	if data, ok := app.isUpdateNoteAction(rawData); ok {
		app.forceStateToWritingNotes(user, session, data, replyToken)
		return
	}

	// 3. Video watching action
	if data, ok := app.isWatchVideoAction(rawData); ok {
		app.LineBot.SendVideoMessage(replyToken, data)
		return
	}

	// 4. Ask AI for help
	if data, ok := app.isAnalyzingPortfolioWithGPT(rawData); ok {
		app.handleAnalyzePortfolioWithGPT(event, user, data, session, replyToken)
		return
	}

	// 5. Route by user state
	switch session.UserState {
	case db.WritingNotes:
		app.handleWritingNotes(event, rawData, user, session, replyToken)
	case db.ChattingWithGPT:
		app.handleChattingWithGPT(event, rawData, user, session, replyToken)
	case db.ViewingExpertVideos:
		app.handleViewingExpertVideos(event, rawData, user, session, replyToken)
	case db.ViewingPortfoilo:
		app.handleViewingPortfolio(event, rawData, user, session, replyToken)
	case db.AnalyzingVideo:
		app.handleAnalyzingVideoActions(event, rawData, user, session, replyToken)
	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// ============================================================================
// 2. State Machine Sub-Handlers
// ============================================================================

// handleWritingNotes handles logic for the “WritingNotes” state.
func (app *App) handleWritingNotes(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		// Move to “SelectingPortfolio” after skill selection
		session.ActionStep = db.SelectingPortfolio
		data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
		if err != nil {
			app.handlePostbackDataTypeError(err, replyToken)
			return
		}
		session.Skill = data.Skill

		if err := app.FirestoreClient.UpdateUserSession(user.ID, *session); err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return
		}

		// Prompt user to select which portfolio entry to update
		if err := app.LineBot.SendPortfolio(
			event,
			user,
			db.SkillStrToEnum(data.Skill),
			session.Handedness,
			session.UserState,
			"請選擇您要更新的學習歷程：",
			true,
		); err != nil {
			app.handleSendPortfolioError(err, replyToken)
			return
		}

	case db.SelectingPortfolio:
		app.handleSelectingPortfolio(rawData, user, session, replyToken)

	case db.WritingPreviewNote, db.WritingReflection:
		app.handleUpdatingNote(event, user, session)
		app.FirestoreClient.ResetSession(user.ID)

	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// handleChattingWithGPT handles logic for the “ChattingWithGPT” state.
func (app *App) handleChattingWithGPT(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		// Move to “Chatting”
		session.ActionStep = db.Chatting
		lineData, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
		if err != nil {
			app.handlePostbackDataTypeError(err, replyToken)
			return
		}
		session.Skill = lineData.Skill
		app.FirestoreClient.UpdateUserSession(user.ID, *session)

		// Inform user we are entering GPT chatting mode
		app.LineBot.SendGPTChattingModeReply(replyToken, "已進入和GPT對話模式")

	case db.Chatting:
		// Get user text message
		var msg string
		message, ok := event.Message.(*linebot.TextMessage)
		if ok {
			msg = message.Text
		}

        // Add the message to GPT conversation and get reply
        conversationID := app.getUserGPTConversation(user, session.Skill)
        response, err := app.GPTClient.AddMessageToConversation(conversationID, msg)
        if err != nil {
            app.handleAddMessageToGPTConversationError(err, replyToken)
            return
        }

		if _, err := app.LineBot.SendGPTChattingModeReply(replyToken, response); err != nil {
			handleLineMessageResponseError(err)
			return
		}

	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// handleViewingExpertVideos handles logic for the “ViewingExpertVideos” state.
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

// handleViewingPortfolio handles logic for the “ViewingPortfoilo” state.
func (app *App) handleViewingPortfolio(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	if err := app.LineBot.SendPortfolio(
		event,
		user,
		db.SkillStrToEnum(data.Skill),
		session.Handedness,
		session.UserState,
		"以下為您的學習歷程：",
		false,
	); err != nil {
		app.handleSendPortfolioError(err, replyToken)
		return
	}

	app.resetSessionWithErrorHandling(user.ID, replyToken)
}

// handleAnalyzingVideoActions handles logic for the “AnalyzingVideo” state.
func (app *App) handleAnalyzingVideoActions(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		session.ActionStep = db.SelectingHandedness
		app.handleSelectingSkill(event, session, rawData, replyToken, app.LineBot.PromptHandednessSelection)

	case db.SelectingHandedness:
		session.ActionStep = db.UploadingVideo
		data, err := app.LineBot.HandleSelectingHandednessPostbackData(rawData)
		if err != nil {
			app.handlePostbackDataTypeError(err, replyToken)
			return
		}
		app.FirestoreClient.UpdateSessionHandedness(user.ID, data.Handedness)
		app.LineBot.PromptUploadVideo(event)

	case db.UploadingVideo:
		app.handleUploadingVideo(event, session, user, replyToken)

	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// ============================================================================
// 3. Individual Action Handlers
// ============================================================================

// forceStateToWritingNotes forces the session to WritingNotes, prompting the user.
func (app *App) forceStateToWritingNotes(user *db.UserData, session *db.UserSession, data *line.WritingNotePostback, replyToken string) {
	actionStep, err := db.ActionStepStrToEnum(data.ActionStep)
	if err != nil {
		app.Logger.Warn.Println("Invalid action step for updating note")
		app.FirestoreClient.ResetSession(user.ID)
		return
	}

	session.ActionStep = actionStep
	session.UpdatingDate = data.WorkDate
	session.UserState = db.WritingNotes
	session.Skill = data.Skill

	if err := app.FirestoreClient.UpdateUserSession(user.ID, *session); err != nil {
		app.handleUpdateSessionError(err, replyToken)
		return
	}

	msg := generateUpdateNoteMessage(data.WorkDate, data.Skill, actionStep)
	app.LineBot.SendReply(replyToken, msg)
}

// handleSelectingPortfolio is invoked when selecting which portfolio entry to update.
func (app *App) handleSelectingPortfolio(rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleWritingNotePostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	actionStep, err := db.ActionStepStrToEnum(data.ActionStep)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	session.ActionStep = actionStep
	session.UpdatingDate = data.WorkDate

	if err := app.FirestoreClient.UpdateUserSession(user.ID, *session); err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleUpdatingNote updates the preview or reflection note in the user’s portfolio.
func (app *App) handleUpdatingNote(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	if session.ActionStep != db.WritingPreviewNote && session.ActionStep != db.WritingReflection {
		app.Logger.Warn.Println("Invalid action step for updating note")
		app.handleInvalidActionStep(user.ID, event.ReplyToken)
		return
	}

	note, ok := event.Message.(*linebot.TextMessage)
	if !ok {
		app.Logger.Warn.Println("Non-text message received when updating note")
		app.FirestoreClient.ResetSession(user.ID)
		return
	}

	portfolio := user.Portfolio.GetSkillPortfolio(session.Skill)

	if session.ActionStep == db.WritingPreviewNote {
		app.FirestoreClient.UpdateUserPortfolioPreviewNote(
			user,
			&portfolio,
			session.UpdatingDate,
			note.Text,
		)
	} else {
		app.FirestoreClient.UpdateUserPortfolioReflection(
			user,
			&portfolio,
			session.UpdatingDate,
			note.Text,
		)
	}

	app.LineBot.SendPortfolio(
		event,
		user,
		db.SkillStrToEnum(session.Skill),
		session.Handedness,
		session.UserState,
		"以下為您的學習歷程：",
		false,
	)
}

// handleAnalyzePortfolioWithGPT processes the user's request to ask GPT for help.
func (app *App) handleAnalyzePortfolioWithGPT(
	event *linebot.Event,
	user *db.UserData,
	data *line.AnalyzingWithGPTPostback,
	session *db.UserSession,
	replyToken string,
) {
	portfolio := app.getUserPortfolio(user, data.Skill)
	gradingDetails := (*portfolio)[data.WorkDate].GradingOutcome.GradingDetails

	preprocessedUsedAnglesData, err := json.Marshal(gradingDetails)
	if err != nil {
		app.Logger.Error.Println("Failed to marshal used angles data:", err)
		app.LineBot.SendReply(replyToken, "無法取得動作資料，請再試一次")
		return
	}

    conversationID := app.getUserGPTConversation(user, data.Skill)
    aiResponse := app.analyzeWithGPT(data, preprocessedUsedAnglesData, conversationID)
	if err := app.FirestoreClient.UpdateUserPortfolioAINote(user, portfolio, data.WorkDate, aiResponse); err != nil {
		app.Logger.Error.Println("Failed to update AI note:", err)
		app.LineBot.SendReply(replyToken, "無法更新AI筆記，請再試一次")
		return
	}

	err = app.LineBot.SendPortfolio(
		event,
		user,
		db.SkillStrToEnum(data.Skill),
		data.Handedness,
		session.UserState,
		"以下為您的學習歷程：",
		false,
	)
	if err != nil {
		app.handleSendPortfolioError(err, replyToken)
		return
	}
}

// handleUploadingVideo processes video uploads, calls AI analysis, and updates the portfolio.
func (app *App) handleUploadingVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	// Get video content
	videoContent, err := app.getVideoContent(event, user.ID)
	if err != nil {
		app.handleGetVideoError(err, replyToken)
		return
	}

	// Send video to AI server for analysis
	resp, err := app.analyzeVideo(videoContent, session.Skill, session.Handedness)
	if err != nil {
		app.handleVideoAnalysisError(err, replyToken)
		return
	}
	app.Logger.Info.Println("AI total grade: ", resp.Grade.TotalGrade)

	// Stitches the video with expert video
	stitchedVideoPath := app.stitchVideoWithExpertVideo(user, resp.ProcessedVideo, session.Skill, session.Handedness)

	// Create thumbnail
	thumbnailPath, err := app.createVideoThumbnail(event, user, stitchedVideoPath)
	if err != nil {
		app.handleThumbnailCreationError(err, replyToken)
		return
	}

	// 5. Upload AI-processed video and update user portfolio
	app.uploadVideoContent(event, user, session, resp.Grade, stitchedVideoPath, thumbnailPath, replyToken)
}

// ============================================================================
// 4. Helper Functions
// ============================================================================

// --------------------------------------------------------------------
// 4.1 Basic Postback Data Extraction
// --------------------------------------------------------------------

func getPostbackData(event *linebot.Event) string {
	if event.Type == linebot.EventTypePostback {
		return event.Postback.Data
	}
	return ""
}

func isMenuSwitchEvent(data string) bool {
	return data == "switch-to-main" || data == "switch-to-secondary"
}

// --------------------------------------------------------------------
// 4.2 Specialized Postback Actions
// --------------------------------------------------------------------

func (app *App) isStopChattingWithGPTAction(rawData string) (*line.StopGPTPostback, bool) {
	data, err := app.LineBot.HandleStopGPTPostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

func (app *App) isUpdateNoteAction(rawData string) (*line.WritingNotePostback, bool) {
	data, err := app.LineBot.HandleWritingNotePostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

func (app *App) isWatchVideoAction(rawData string) (*line.VideoPostback, bool) {
	data, err := app.LineBot.HandleVideoPostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

func (app *App) isAnalyzingPortfolioWithGPT(rawData string) (*line.AnalyzingWithGPTPostback, bool) {
	data, err := app.LineBot.HandleAskingAIForHelpPostbackData(rawData)
	if err != nil {
		return nil, false
	}
	return data, true
}

// --------------------------------------------------------------------
// 4.3 GPT / AI Interaction
// --------------------------------------------------------------------

func (app *App) analyzeWithGPT(
    data *line.AnalyzingWithGPTPostback,
    preprocessedUsedAnglesData []byte,
    conversationID string,
) string {
	msg := fmt.Sprintf(
		"以下為我此次動作的資料，請分析並給出改善建議：\n慣用手：%v\n動作技能：%v\n動作評分細節：%v",
		data.Handedness,
		data.Skill,
		string(preprocessedUsedAnglesData),
	)
    response, err := app.GPTClient.AddMessageToConversation(conversationID, msg)
    if err != nil {
        app.Logger.Error.Println("Failed to get GPT response:", err)
        return "無法取得建議，請再試一次"
    }
    return response
}

// --------------------------------------------------------------------
// 4.4 Video & Portfolio Updates
// --------------------------------------------------------------------

// getVideoContent retrieves the video bytes from a linebot.VideoMessage.
func (app *App) getVideoContent(event *linebot.Event, userID string) ([]byte, error) {
	videoMsg, ok := event.Message.(*linebot.VideoMessage)
	if !ok {
		app.Logger.Warn.Println("Non-video message received")
		app.FirestoreClient.ResetSession(userID)
		return nil, errors.New("non-video message")
	}
	return app.LineBot.GetVideoContent(videoMsg.ID)
}

// uploadVideoContent handles the final step of uploading the processed video
// to Google Drive (or your storage), then updating the user’s portfolio.
func (app *App) uploadVideoContent(
	event *linebot.Event,
	user *db.UserData,
	session *db.UserSession,
	grade commons.GradingOutcome,
	stitchedVideoPath string,
	thumbnailPath string,
	replyToken string,
) {
	timestamp := time.Now().Format("2006-01-02-15-04")

	// Decode AI-processed video
	videoData, err := os.ReadFile(stitchedVideoPath)
    if err != nil {
        app.handleUploadToDriveError(fmt.Errorf("failed to find stitched video: %w", err), replyToken)
        return
    }

	// Upload video + thumbnail to cloud storage (placeholder function)
	videoLink, thumbLink, err := app.uploadVideoToBucket(
		user,
		session,
		videoData,
		thumbnailPath,
		timestamp,
	)
	if err != nil {
		app.handleUploadToDriveError(err, replyToken)
		return
	}

	// Update user portfolio with AI grading
	if err := app.updateUserPortfolioVideo(user, session, timestamp, grade, videoLink, thumbLink); err != nil {
		app.handleUpdateUserPortfolioError(err, replyToken)
		return
	}

	// Notify user
	if err := app.sendVideoUploadedReply(event, session, user); err != nil {
		app.handleSendingReplyMessageError(err, replyToken)
		return
	}

	app.FirestoreClient.ResetSession(user.ID)
}

// generateUpdateNoteMessage forms a response prompt for note updating.
func generateUpdateNoteMessage(workDate, skill string, actionStep db.ActionStep) string {
	skillStr := db.SkillStrToEnum(skill).ChnString()
	msg := "請輸入【" + workDate + "】的【" + skillStr + "】的"
	if actionStep == db.WritingPreviewNote {
		msg += "課前檢視要點"
	} else {
		msg += "學習反思"
	}
	return msg
}

// --------------------------------------------------------------------
// 4.5 Helper for Selecting Skill
// --------------------------------------------------------------------

// handleSelectingSkill helps transition the user from “SelectingSkill” to the
// next action, e.g., choosing handedness or uploading a video.
func (app *App) handleSelectingSkill(
	event *linebot.Event,
	session *db.UserSession,
	rawData string,
	replyToken string,
	nextStepFunc func(*linebot.Event) error,
) {
	data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	if err := nextStepFunc(event); err != nil {
		app.handleVideoUploadPromptError(err, replyToken)
		return
	}

	session.Skill = data.Skill
	if err := app.FirestoreClient.UpdateUserSession(event.Source.UserID, *session); err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleSendingExpertVideos is a helper that sets up the correct expert videos
// after the user selects their handedness.
func (app *App) handleSendingExpertVideos(event *linebot.Event, session *db.UserSession, replyToken string) {
	data, err := app.LineBot.HandleSelectingHandednessPostbackData(event.Postback.Data)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	handedness, err := db.HandednessStrToEnum(data.Handedness)
	if err != nil {
		app.Logger.Warn.Println("Invalid handedness received:", data.Handedness)
		app.handlePostbackDataTypeError(err, replyToken)
		return
	}

	skill := db.SkillStrToEnum(session.Skill)
	if err := app.LineBot.SendExpertVideos(handedness, skill, replyToken); err != nil {
		app.handleSendExpertVideosError(err, replyToken)
		return
	}

	app.Logger.Info.Printf("Expert videos sent for User ID: %v, Skill: %v, Handedness: %v", event.Source.UserID, skill, handedness)
}

// ============================================================================
// 5. Error Handling & Session Management
// ============================================================================

// handleInvalidActionStep resets the session and sends a default error reply.
func (app *App) handleInvalidActionStep(userID, replyToken string) {
	app.FirestoreClient.ResetSession(userID)
	if _, err := app.LineBot.SendDefaultErrorReply(replyToken); err != nil {
		app.Logger.Warn.Println("Error sending default error reply:", err)
	}
}

// resetSessionWithErrorHandling is a small helper to reset the session.
func (app *App) resetSessionWithErrorHandling(userID, replyToken string) {
	if err := app.FirestoreClient.ResetSession(userID); err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}
