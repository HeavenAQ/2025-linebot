package app

func (app *App) lineErrorReplyHandler(err error) {
	if err != nil {
		app.Logger.Warn.Println("Error sending type error message: ", err)
		return
	}
}

func (app *App) handleLineError(errMsg string) func(err error, replyToken string) {
	return func(err error, replyToken string) {
		if err != nil {
			app.Logger.Error.Println(errMsg, err)
			_, err = app.LineBot.SendDefaultErrorReply(replyToken)
			app.lineErrorReplyHandler(err)
		}
	}
}

func (app *App) handleUpdateSessionError(err error, replyToken string) {
	app.handleLineError("Error updating session user state")(err, replyToken)
}

func (app *App) handleVideoUploadPromptError(err error, replyToken string) {
	app.handleLineError("Error prompting video upload")(err, replyToken)
}
