package app

import "fmt"

func handleLineMessageResponseError(err error) {
	if err != nil {
		fmt.Println("Error sending type error message: ", err)
		return
	}
}

func (app *App) handleLineError(errMsg string, successMsg string) func(err error, replyToken string) {
	return func(err error, replyToken string) {
		if err != nil {
			app.Logger.Error.Println(errMsg, err)
			_, err = app.LineBot.SendDefaultErrorReply(replyToken)
			handleLineMessageResponseError(err)
			return
		}
		app.Logger.Info.Println(successMsg)
	}
}

func (app *App) handleUpdateSessionError(err error, replyToken string) {
	app.handleLineError(
		"Error updating session user state",
		"Session user state has been updated",
	)(err, replyToken)
}

func (app *App) handleVideoUploadPromptError(err error, replyToken string) {
	app.handleLineError(
		"Error prompting video upload",
		"Video upload has been prompted",
	)(err, replyToken)
}

func (app *App) handleGetVideoError(err error, replyToken string) {
	app.handleLineError(
		"Error getting the video",
		"Video content has been received",
	)(err, replyToken)
}

func (app *App) handleThumbnailCreationError(err error, replyToken string) {
	app.handleLineError(
		"Error creating a thumbnail for the video",
		"A Thumbnail has been created!",
	)(err, replyToken)
}

func (app *App) handleUploadToDriveError(err error, replyToken string) {
	app.handleLineError(
		"Error uploading the video to Google Drive",
		"Video has been uploaded to Google Drive successfully",
	)(err, replyToken)
}

func (app *App) handleSendingReplyMessageError(err error, replyToken string) {
	app.handleLineError(
		"Failed to send reply messages through LINE",
		"Reply messages has been sent",
	)(err, replyToken)
}

func (app *App) handleUpdateUserPortfolioError(err error, replyToken string) {
	app.handleLineError(
		"Failed to update user portfolio",
		"The user portfolio has been updated successfully",
	)(err, replyToken)
}

func (app *App) handlePostbackDataTypeError(err error, replyToken string) {
	app.handleLineError(
		"Error handling postback data type casting",
		"Postback data type has been handled",
	)(err, replyToken)
}
