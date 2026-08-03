package db

import (
    "fmt"
    "time"

    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// DailySummary stores a cached per-user summary for a specific date.
// ScoreKey fingerprints the grades the summary was written from, so a new
// graded attempt invalidates the cache the same way a new message does.
type DailySummary struct {
    Summary   string    `json:"summary" firestore:"summary"`
    LastCount int       `json:"last_count" firestore:"last_count"`
    ScoreKey  string    `json:"score_key" firestore:"score_key"`
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

// SetDailySummary upserts the cached summary along with the message count and
// score fingerprint it was computed from.
func (client *FirestoreClient) SetDailySummary(userID, date, skill, summary string, lastCount int, scoreKey string) error {
    ctx := *client.Ctx
    docID := client.dailySummaryDocID(userID, date, skill)
    docRef := client.DailySummaries.Doc(docID)

    payload := DailySummary{
        Summary:   summary,
        LastCount: lastCount,
        ScoreKey:  scoreKey,
        Date:      date,
        Skill:     skill,
        UpdatedAt: time.Now().UTC(),
    }

    _, err := docRef.Set(ctx, payload)
    return err
}
