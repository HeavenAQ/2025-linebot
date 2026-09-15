package app

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
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
	display := db.WithWeeklyReflectionNotes(user, weekly)
	// Sign a presentation copy, never persist expiring links in Firestore.
	portfolio := display.Portfolio.GetSkillPortfolio(skill.String())
	failures := signPortfolioThumbnails(portfolio, app.LineBot.PortfolioWorksForDisplay(portfolio),
		func(value string) (commons.MediaRef, error) {
			return app.StorageClient.SignThumbnailURL(value, app.Config.GCP.ServiceAccountEmail)
		})
	if len(failures) > 0 {
		app.Logger.Warn.Printf("portfolio reply omitting %d unavailable thumbnails", len(failures))
	}
	return app.LineBot.SendPortfolio(
		event,
		display,
		skill,
		userState,
		textMsg,
		showBtns,
	)
}

// Mutates only the presentation copy. One bad image must not prevent the
// current analysis or other portfolio records from reaching the learner.
func signPortfolioThumbnails(portfolio map[string]db.Work, visible []db.Work,
	sign func(string) (commons.MediaRef, error),
) []error {
	var failures []error
	for _, work := range visible {
		if work.Thumbnail == "" {
			continue
		}
		signed, err := sign(work.Thumbnail)
		if err != nil {
			failures = append(failures, err)
			work.Thumbnail = ""
		} else {
			work.Thumbnail = signed.SignedURL
		}
		portfolio[work.DateTime] = work
	}
	return failures
}
