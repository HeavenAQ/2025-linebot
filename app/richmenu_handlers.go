package app

import "github.com/HeavenAQ/nstc-linebot-2025/api/db"

func (app *App) processReadingInstruction(user *db.UserData, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	res, err := app.LineBot.SendInstruction(replyToken)
	app.handleMessageResponse(res, err, replyToken)
}

func (app *App) processViewingPortfolio(user *db.UserData, userState db.UserState, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingPortfoilo, db.SelectingSkill)
	res, err := app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要查看的動作")
	app.handleMessageResponse(res, err, replyToken)
}

func (app *App) processViewingExpertVideos(user *db.UserData, userState db.UserState, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingExpertVideos, db.SelectingSkill)
	res, err := app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要觀看的動作")
	app.handleMessageResponse(res, err, replyToken)
}

func (app *App) processAnalyzingVideo(user *db.UserData, userState db.UserState, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.AnalyzingVideo, db.SelectingSkill)
	res, err := app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要分析的動作")
	app.handleMessageResponse(res, err, replyToken)
}

func (app *App) processWritingNotes(user *db.UserData, userState db.UserState, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.WritingNotes, db.SelectingSkill)
	res, err := app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要記錄的動作")
	app.handleMessageResponse(res, err, replyToken)
}

func (app *App) processChattingWithGPT(user *db.UserData, userState db.UserState, replyToken string) {
	app.FirestoreClient.ResetSession(user.ID)
	err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ChattingWithGPT, db.SelectingSkill)
	res, err := app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要對談的動作")
	app.handleMessageResponse(res, err, replyToken)
}
