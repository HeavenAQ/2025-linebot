package db

import (
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// WeeklyPreview records the 課前預習 push a learner has already been sent this
// week, so a retried or rescheduled run does not message them twice.
type WeeklyPreview struct {
	UserID    string    `json:"user_id" firestore:"user_id"`
	Week      string    `json:"week" firestore:"week"`
	Skill     string    `json:"skill" firestore:"skill"`
	Message   string    `json:"message" firestore:"message"`
	CreatedAt time.Time `json:"created_at" firestore:"created_at"`
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

// SetWeeklyPreview records a push that has been delivered.
func (client *FirestoreClient) SetWeeklyPreview(userID, week, skill, message string) error {
	_, err := client.WeeklyPreviews.Doc(client.weeklyPreviewDocID(userID, week)).Set(*client.Ctx, WeeklyPreview{
		UserID:    userID,
		Week:      week,
		Skill:     skill,
		Message:   message,
		CreatedAt: time.Now().UTC(),
	})
	return err
}
