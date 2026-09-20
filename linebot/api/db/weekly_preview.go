package db

import (
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// WeeklyPreview is a learner's 課前預習 note for one week. The scheduled push
// and the 產生課前預習 button share it, so both show the same advice.
//
// SourceKey fingerprints the history the note was written from: a new analysis
// changes it, and the note is rewritten on the next request. Pushed is kept
// apart from the note so asking for one by hand does not consume that week's
// push, and a retried run still does not message anyone twice.
type WeeklyPreview struct {
	UserID    string    `json:"user_id" firestore:"user_id"`
	Week      string    `json:"week" firestore:"week"`
	Skill     string    `json:"skill" firestore:"skill"`
	Message   string    `json:"message" firestore:"message"`
	SourceKey string    `json:"source_key" firestore:"source_key"`
	Pushed    bool      `json:"pushed" firestore:"pushed"`
	CreatedAt time.Time `json:"created_at" firestore:"created_at"`
}

// Fresh reports whether the note still describes the learner's current record.
func (p *WeeklyPreview) Fresh(sourceKey string) bool {
	return p != nil && strings.TrimSpace(p.Message) != "" && p.SourceKey == sourceKey
}

// Delivered reports whether this week's push has gone out. Records written
// before notes could be asked for by hand carry neither field, and their
// existence meant the push had been sent.
func (p *WeeklyPreview) Delivered() bool {
	return p != nil && (p.Pushed || p.SourceKey == "")
}

// ISOWeek labels a week the way the dedupe key needs it, e.g. "2026-W32".
func ISOWeek(at time.Time) string {
	year, week := at.ISOWeek()
	return fmt.Sprintf("%d-W%02d", year, week)
}

func (client *FirestoreClient) weeklyPreviewDocID(userID, week string) string {
	return fmt.Sprintf("%s_%s", userID, week)
}

// GetWeeklyPreview returns this week's push for a learner, or nil if they have
// not been sent one yet.
func (client *FirestoreClient) GetWeeklyPreview(userID, week string) (*WeeklyPreview, error) {
	snap, err := client.WeeklyPreviews.Doc(client.weeklyPreviewDocID(userID, week)).Get(*client.Ctx)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return nil, nil
		}
		return nil, err
	}
	var preview WeeklyPreview
	if err := snap.DataTo(&preview); err != nil {
		return nil, err
	}
	return &preview, nil
}

// SetWeeklyPreview stores a note, whether it has been pushed or only asked for.
func (client *FirestoreClient) SetWeeklyPreview(preview WeeklyPreview) error {
	preview.CreatedAt = time.Now().UTC()
	_, err := client.WeeklyPreviews.Doc(
		client.weeklyPreviewDocID(preview.UserID, preview.Week),
	).Set(*client.Ctx, preview)
	return err
}
