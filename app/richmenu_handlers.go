package app

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

func processWrapper(
	app *App,
	user *db.UserData,
	replyToken string,
	processFunc func(replyToken string) (res *linebot.BasicResponse, err error),
) func() {
	return func() {
		app.FirestoreClient.ResetSession(user.ID)
		res, err := processFunc(replyToken)
		app.handleMessageResponseError(res, err, replyToken)
	}
}

func (app *App) processReadingInstruction(user *db.UserData, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		return app.LineBot.SendInstruction(replyToken)
	})()
}

func (app *App) processViewingPortfolio(user *db.UserData, userState db.UserState, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingPortfoilo, db.SelectingSkill)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return nil, err
		}
		return app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要查看的動作")
	})()
}

func (app *App) processViewingExpertVideos(user *db.UserData, userState db.UserState, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ViewingExpertVideos, db.SelectingSkill)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return nil, err
		}
		return app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要觀看的動作")
	})()
}

func (app *App) processUploadingVideo(user *db.UserData, userState db.UserState, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.UploadingVideo, db.SelectingSkill)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return nil, err
		}
		return app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要上傳的動作")
	})()
}

func (app *App) processWritingNotes(user *db.UserData, userState db.UserState, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.WritingNotes, db.SelectingSkill)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return nil, err
		}
		return app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要紀錄的動作")
	})()
}

func (app *App) processChattingWithGPT(user *db.UserData, userState db.UserState, replyToken string) {
	processWrapper(app, user, replyToken, func(replyToken string) (*linebot.BasicResponse, error) {
		err := app.FirestoreClient.UpdateSessionUserState(user.ID, db.ChattingWithGPT, db.SelectingSkill)
		if err != nil {
			app.handleUpdateSessionError(err, replyToken)
			return nil, err
		}
		return app.LineBot.PromptSkillSelection(replyToken, userState, "請選擇要與 GPT 討論的動作")
	})()
}
