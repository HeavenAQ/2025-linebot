package app

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

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

func (app *App) handleSendPortfolioError(err error, replyToken string) {
	if err, ok := err.(*line.NoPortfolioError); ok {
		app.Logger.Info.Println(err)
		err := app.LineBot.SendNoPortfolioReply(replyToken, err.Skill)
		handleLineMessageResponseError(err)
		return
	}

	app.handleLineError(
		"Error sending portfolio",
		"Portfolio has been sent",
	)(err, replyToken)
}

func (app *App) handleSendExpertVideosError(err error, replyToken string) {
	app.handleLineError(
		"Error sending expert videos",
		"Expert videos has been sent",
	)(err, replyToken)
}

func (app *App) handleMessageResponseError(res *linebot.BasicResponse, err error, replyToken string) {
	app.handleLineError(
		"Error sending message",
		fmt.Sprintf("Message sent successfully. Response from LINE: %v", res),
	)(err, replyToken)
}

func (app *App) handleVideoAnalysisError(err error, replyToken string) {
	app.handleLineError(
		"Error analyzing the video",
		"Video has been analyzed",
	)(err, replyToken)
}

func (app *App) handleAddMessageToGPTThreadError(err error, replyToken string) {
	app.handleLineError(
		"Error adding message to GPT thread",
		"Message has been added to GPT thread.",
	)(err, replyToken)
}

func (app *App) handleGPTRunThreadError(err error, replyToken string) {
	app.handleLineError(
		"Error running GPT thread",
		"GPT thread has been run. Response from LINE: %v",
	)(err, replyToken)
}

func (app *App) handleGetGPTResponseError(err error, replyToken string) {
	app.handleLineError(
		"Error getting GPT response",
		"GPT response has been received. Response from LINE: %v",
	)(err, replyToken)
}
