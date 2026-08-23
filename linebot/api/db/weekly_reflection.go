package db

import (
	"fmt"
	"regexp"
	"time"

	"google.golang.org/api/iterator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// MaxReflectionLength caps a stored note. Reflections are written by hand in a
// textarea; anything longer is a paste or a client bug, not a student.
const MaxReflectionLength = 4000

// weekPattern is the ISO week label produced by ISOWeek, e.g. "2026-W34".
var weekPattern = regexp.MustCompile(`^\d{4}-W\d{2}$`)

// ValidWeek reports whether a label is one ISOWeek would produce. Week labels
// arrive from the browser and are used to build document IDs, so they are
// checked rather than trusted.
func ValidWeek(week string) bool {
	return weekPattern.MatchString(week)
}

// WeeklyReflection is a learner's own write-up of one week's practice, written
// in the LIFF review tab. It is deliberately per week rather than per analysis:
// students review the week as a whole, and one considered note beats several
// half-filled ones.
type WeeklyReflection struct {
	UserID    string    `json:"user_id" firestore:"user_id"`
	Week      string    `json:"week" firestore:"week"`
	Note      string    `json:"note" firestore:"note"`
	UpdatedAt time.Time `json:"updated_at" firestore:"updated_at"`
}

func (client *FirestoreClient) weeklyReflectionDocID(userID, week string) string {
	return fmt.Sprintf("%s_%s", userID, week)
}

// GetWeeklyReflection returns one week's note, or nil when nothing is written.
func (client *FirestoreClient) GetWeeklyReflection(userID, week string) (*WeeklyReflection, error) {
	snap, err := client.WeeklyReflections.Doc(client.weeklyReflectionDocID(userID, week)).Get(*client.Ctx)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return nil, nil
		}
		return nil, err
	}
	var reflection WeeklyReflection
	if err := snap.DataTo(&reflection); err != nil {
		return nil, err
	}
	return &reflection, nil
}

// ListWeeklyReflections returns every note a learner has written, keyed by week,
// so the review tab can render all of them without a request per week.
func (client *FirestoreClient) ListWeeklyReflections(userID string) (map[string]WeeklyReflection, error) {
	iter := client.WeeklyReflections.Where("user_id", "==", userID).Documents(*client.Ctx)
	defer iter.Stop()

	reflections := map[string]WeeklyReflection{}
	for {
		doc, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("list weekly reflections: %w", err)
		}
		var reflection WeeklyReflection
		if err := doc.DataTo(&reflection); err != nil {
			return nil, fmt.Errorf("decode weekly reflection %s: %w", doc.Ref.ID, err)
		}
		reflections[reflection.Week] = reflection
	}
	return reflections, nil
}

// SetWeeklyReflection writes one week's note, replacing whatever was there.
func (client *FirestoreClient) SetWeeklyReflection(userID, week, note string) (*WeeklyReflection, error) {
	reflection := WeeklyReflection{
		UserID:    userID,
		Week:      week,
		Note:      note,
		UpdatedAt: time.Now().UTC(),
	}
	_, err := client.WeeklyReflections.Doc(client.weeklyReflectionDocID(userID, week)).Set(*client.Ctx, reflection)
	if err != nil {
		return nil, fmt.Errorf("write weekly reflection: %w", err)
	}
	return &reflection, nil
}
