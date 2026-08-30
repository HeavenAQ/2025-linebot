package app

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	linebotsdk "github.com/line/line-bot-sdk-go/v7/linebot"
)

// sendPortfolio joins the week-level notes written by LIFF onto the video
// attempts rendered by the LINE carousel. Both records live in the same named
// Firestore database, but in different collections by design.
func (app *App) sendPortfolio(
	event *linebotsdk.Event,
	user *db.UserData,
	skill db.BadmintonSkill,
	userState db.UserState,
	textMsg string,
	showBtns bool,
) error {
	weekly, err := app.FirestoreClient.ListWeeklyReflections(user.ID)
	if err != nil {
		return fmt.Errorf("load weekly notes for portfolio: %w", err)
	}
	return app.LineBot.SendPortfolio(
		event,
		db.WithWeeklyReflectionNotes(user, weekly),
		skill,
		userState,
		textMsg,
		showBtns,
	)
}
