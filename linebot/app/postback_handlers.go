package app

import (
	"errors"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
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

	// 1. Note updating action
	if data, ok := app.isUpdateNoteAction(rawData); ok {
		app.redirectNotePostbackToReview(user, session, data, replyToken)
		return
	}

	// 2. Video watching action
	if data, ok := app.isWatchVideoAction(rawData); ok {
		app.handleWatchPortfolioVideo(user, data, replyToken)
		return
	}

	// 3. Route by user state
	switch session.UserState {
	case db.WritingReflectionNote:
		app.handleWritingNotes(event, rawData, user, session, replyToken)
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

// handleWritingNotes is the old over-chat reflection conversation. Nothing puts
// a session into WritingReflectionNote any more -- 學習反思 hands over a link and
// resets -- so this is unreachable, and kept only because the postback data it
// reads still exists.
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
		if !db.IsSupportedSkill(data.Skill) {
			app.rejectUnsupportedSkill(user.ID, data.Skill, replyToken)
			return
		}
		session.Skill = data.Skill

		if err := app.FirestoreClient.UpdateUserSession(user.ID, *session); err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return
		}

		// Prompt user to select which portfolio entry to update
		if err := app.sendPortfolio(
			event,
			user,
			db.SkillStrToEnum(data.Skill),
			session.UserState,
			"請選擇您要更新的學習歷程：",
			true,
		); err != nil {
			app.handleSendPortfolioError(err, replyToken)
			return
		}

	case db.SelectingPortfolio:
		app.handleSelectingPortfolio(rawData, user, session, replyToken)

	case db.WritingReflection:
		app.handleUpdatingNote(event, user, session)
		app.FirestoreClient.ResetSession(user.ID)

	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// handleViewingExpertVideos handles logic for the “ViewingExpertVideos” state.
func (app *App) handleViewingExpertVideos(event *linebot.Event, rawData string, user *db.UserData, session *db.UserSession, replyToken string) {
	switch session.ActionStep {
	case db.SelectingSkill:
		// Handedness comes from the learner's profile, so picking the stroke
		// is the whole flow.
		noPrompt := func(*linebot.Event) error { return nil }
		if !app.handleSelectingSkill(event, session, rawData, replyToken, noPrompt) {
			return
		}
		app.handleSendingExpertVideos(event, user, session, replyToken)
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

	// The skill picker no longer offers unsupported skills, so this is a stale
	// payload. Past lift/clear work is still readable through the learning
	// dashboard and the portfolio cards already sitting in the chat.
	if !db.IsSupportedSkill(data.Skill) {
		app.rejectUnsupportedSkill(user.ID, data.Skill, replyToken)
		return
	}

	if err := app.sendPortfolio(
		event,
		user,
		db.SkillStrToEnum(data.Skill),
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
		session.ActionStep = db.UploadingVideo
		app.handleSelectingSkill(event, session, rawData, replyToken, app.LineBot.PromptUploadVideo)

	case db.UploadingVideo:
		app.handleUploadingVideo(event, session, user, replyToken)

	default:
		app.handleInvalidActionStep(user.ID, replyToken)
	}
}

// ============================================================================
// 3. Individual Action Handlers
// ============================================================================

// redirectNotePostbackToReview answers a portfolio card's old note button. It
// used to put the session into note-writing mode; rather than start a
// conversation that no longer exists it sends the learner to the web app, where
// the reflection now lives.
func (app *App) redirectNotePostbackToReview(user *db.UserData, _ *db.UserSession, data *line.WritingNotePostback, replyToken string) {
	app.Logger.Info.Printf(
		"[notes] deprecated note postback user_id=%s skill=%s work_date=%s",
		user.ID, data.Skill, data.WorkDate,
	)
	app.FirestoreClient.ResetSession(user.ID)
	if _, err := app.LineBot.SendWeeklyReflectionLink(replyToken, app.Config.ReviewURL()); err != nil {
		app.handleSendingReplyMessageError(err, replyToken)
	}
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
	session.UpdatedDate = data.WorkDate

	if err := app.FirestoreClient.UpdateUserSession(user.ID, *session); err != nil {
		app.handleUpdateSessionError(err, replyToken)
	}
}

// handleUpdatingNote updates the reflection note in the user’s portfolio.
func (app *App) handleUpdatingNote(event *linebot.Event, user *db.UserData, session *db.UserSession) {
	if session.ActionStep != db.WritingReflection {
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

	app.FirestoreClient.UpdateUserPortfolioReflection(
		user,
		&portfolio,
		session.Skill,
		session.UpdatedDate,
		note.Text,
	)

	app.sendPortfolio(
		event,
		user,
		db.SkillStrToEnum(session.Skill),
		session.UserState,
		"以下為您的學習歷程：",
		false,
	)
}

func (app *App) handleWatchPortfolioVideo(
	user *db.UserData,
	data *line.VideoPostback,
	replyToken string,
) {
	portfolio := user.Portfolio.GetSkillPortfolio(data.Skill)
	work, ok := portfolio[data.WorkDate]
	if !ok {
		app.LineBot.SendReply(replyToken, "找不到這次影片紀錄，請重新開啟學習歷程")
		return
	}

	if work.StudentVideo.ObjectPath == "" {
		if work.AnalysisStatus == "pending" {
			app.LineBot.SendReply(replyToken, "影片仍在分析中，完成後可從學習歷程查看結果，無需重新上傳。")
			return
		}
		app.LineBot.SendReply(replyToken, "這次紀錄沒有可播放的影片，請重新上傳")
		return
	}
	// Always signed fresh. The URL a record was created with expires within the
	// hour, so there is nothing worth falling back to.
	video, err := app.StorageClient.SignPlaybackURLIn(
		storage.BucketFromGCSURI(work.StudentVideo.GCSURI),
		work.StudentVideo.ObjectPath, app.Config.GCP.ServiceAccountEmail,
	)
	if err != nil {
		app.Logger.Error.Printf("failed to refresh portfolio video URL: %v", err)
		app.LineBot.SendReply(replyToken, "影片連結更新失敗，請稍後再試")
		return
	}

	thumbnail, err := app.StorageClient.SignThumbnailURL(work.Thumbnail, app.Config.GCP.ServiceAccountEmail)
	if err != nil {
		app.Logger.Warn.Println("portfolio video sent as a link because its thumbnail is unavailable")
		_, replyErr := app.LineBot.SendReply(replyToken, "這筆紀錄的預覽圖無法使用，請點擊連結播放影片：\n"+video.SignedURL)
		handleLineMessageResponseError(replyErr)
		return
	}
	if _, err := app.LineBot.SendVideoMessage(
		replyToken, video.SignedURL, thumbnail.SignedURL,
	); err != nil {
		app.Logger.Error.Printf("failed to send portfolio video through LINE: %v", err)
	}
}

// handleUploadingVideo accepts a learner's video and records a durable
// analysis job, replying at once with a pending portfolio card. Analysis never
// runs inside the webhook request: it can take minutes, which would hold the
// request (and a Cloud Run instance) open and outlive LINE's reply token.
//
// Without a configured queue, or with ANALYSIS_ASYNC_ACCEPT=false (the
// operational switch that pauses new uploads while queued jobs drain), the
// learner is told to try again later before the video is even downloaded.
func (app *App) handleUploadingVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	// A video message routes straight here on the session's stored skill, which
	// may have been chosen before the skill was withdrawn.
	if !db.IsSupportedSkill(session.Skill) {
		app.rejectUnsupportedSkill(user.ID, session.Skill, replyToken)
		return
	}
	if app.AnalysisQueue == nil || !app.Config.AnalysisServer.AsyncAccept {
		app.Logger.Warn.Printf("video upload refused: analysis queue unavailable or paused user_id=%s", user.ID)
		_, err := app.LineBot.SendReply(replyToken, "動作分析暫時暫停中，影片未被接收。請稍後再上傳一次。")
		handleLineMessageResponseError(err)
		return
	}

	videoContent, err := app.getVideoContent(event, user.ID)
	if err != nil {
		app.handleGetVideoError(err, replyToken)
		return
	}
	app.enqueueVideoAnalysis(event, session, user, videoContent, replyToken)
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

// --------------------------------------------------------------------
// 4.3 Video & Portfolio Updates
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

// --------------------------------------------------------------------
// 4.5 Helper for Selecting Skill
// --------------------------------------------------------------------

// rejectUnsupportedSkill turns away a request for a skill the course is not
// running this semester. The selection UI never offers one, so reaching here
// means a stale postback, an old rich menu, or a client that skipped the menu.
// The session is reset so the learner lands back on the main menu instead of
// being stranded mid-flow, and nothing downstream (analysis, portfolio writes)
// is called.
func (app *App) rejectUnsupportedSkill(userID, skill, replyToken string) {
	app.Logger.Warn.Printf(
		"Unsupported skill requested. User ID: %v, Skill: %v", userID, skill,
	)
	if err := app.FirestoreClient.ResetSession(userID); err != nil {
		app.Logger.Warn.Println("Error resetting session after unsupported skill:", err)
	}
	_, err := app.LineBot.SendUnsupportedSkillReply(replyToken, skill)
	handleLineMessageResponseError(err)
}

// handleSelectingSkill moves the learner from “SelectingSkill” to the next
// action, e.g. uploading a video. It reports whether the stroke was accepted
// and saved, so a caller that acts on the choice does not act on a rejected
// one.
func (app *App) handleSelectingSkill(
	event *linebot.Event,
	session *db.UserSession,
	rawData string,
	replyToken string,
	nextStepFunc func(*linebot.Event) error,
) bool {
	data, err := app.LineBot.HandleSelectingSkillPostbackData(rawData)
	if err != nil {
		app.handlePostbackDataTypeError(err, replyToken)
		return false
	}

	if !db.IsSupportedSkill(data.Skill) {
		app.rejectUnsupportedSkill(event.Source.UserID, data.Skill, replyToken)
		return false
	}

	if err := nextStepFunc(event); err != nil {
		app.handleVideoUploadPromptError(err, replyToken)
		return false
	}

	session.Skill = data.Skill
	if err := app.FirestoreClient.UpdateUserSession(event.Source.UserID, *session); err != nil {
		app.handleUpdateSessionError(err, replyToken)
		return false
	}
	return true
}

// handleSendingExpertVideos sends the demonstrations that match the learner's
// stroke and their own handedness.
func (app *App) handleSendingExpertVideos(event *linebot.Event, user *db.UserData, session *db.UserSession, replyToken string) {
	// The skill was already checked when it was selected; re-check in case the
	// session was saved before the skill was withdrawn.
	if !db.IsSupportedSkill(session.Skill) {
		app.rejectUnsupportedSkill(event.Source.UserID, session.Skill, replyToken)
		return
	}

	handedness := user.Handedness
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
