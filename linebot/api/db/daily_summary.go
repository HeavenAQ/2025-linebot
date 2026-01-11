package db

import (
    "fmt"
    "time"

    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// DailySummary stores a cached per-user summary for a specific date.
type DailySummary struct {
    Summary   string    `json:"summary" firestore:"summary"`
    LastCount int       `json:"last_count" firestore:"last_count"`
    Date      string    `json:"date" firestore:"date"`
    Skill     string    `json:"skill" firestore:"skill"`
    UpdatedAt time.Time `json:"updated_at" firestore:"updated_at"`
}

func (client *FirestoreClient) dailySummaryDocID(userID, date, skill string) string {
    return fmt.Sprintf("%s_%s_%s", userID, date, skill)
}

// GetDailySummary returns the cached summary for a user on a given date, or nil if none exists.
func (client *FirestoreClient) GetDailySummary(userID, date, skill string) (*DailySummary, error) {
    ctx := *client.Ctx
    docID := client.dailySummaryDocID(userID, date, skill)
    docRef := client.DailySummaries.Doc(docID)
    snap, err := docRef.Get(ctx)
    if err != nil {
        if status.Code(err) == codes.NotFound {
            return nil, nil
		}
		return nil, err
	}
	var ds DailySummary
	if err := snap.DataTo(&ds); err != nil {
		return nil, err
	}
	return &ds, nil
}

// SetDailySummary upserts the cached summary and the last message count used to compute it.
func (client *FirestoreClient) SetDailySummary(userID, date, skill, summary string, lastCount int) error {
    ctx := *client.Ctx
    docID := client.dailySummaryDocID(userID, date, skill)
    docRef := client.DailySummaries.Doc(docID)

    payload := DailySummary{
        Summary:   summary,
        LastCount: lastCount,
        Date:      date,
        Skill:     skill,
        UpdatedAt: time.Now().UTC(),
    }

    _, err := docRef.Set(ctx, payload)
    return err
}
