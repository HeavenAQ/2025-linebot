package db

import (
	"fmt"
	"regexp"
	"time"

	"cloud.google.com/go/firestore"
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
	UserID string `json:"user_id" firestore:"user_id"`
	Week   string `json:"week" firestore:"week"`
	Note   string `json:"note" firestore:"note"`
	// Preview is the same week seen from the other end: the learner's own
	// 課前檢視要點, what they mean to watch for next lesson. It shares the
	// document because it belongs to the same week, and a record written before
	// this field existed simply reads back with it empty.
	Preview   string    `json:"preview" firestore:"preview"`
	UpdatedAt time.Time `json:"updated_at" firestore:"updated_at"`
}

// WeeklyNoteField names one of the two notes a week's record holds. The value
// is the Firestore field itself, so it is also what a save is scoped to.
type WeeklyNoteField string

const (
	ReflectionNote WeeklyNoteField = "note"
	PreviewNote    WeeklyNoteField = "preview"
)

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

// WithWeeklyReflectionNotes returns a presentation copy of user whose
// portfolio attempts carry the notes written in the LIFF weekly review. The
// persisted portfolio predates weekly reviews and still has a per-video
// Reflection field, so LINE cards must join the two data sets explicitly.
//
// A copy is returned because displaying a portfolio must not copy week-level
// notes back into every persisted video document the next time UserData is
// updated for an unrelated reason.
func WithWeeklyReflectionNotes(
	user *UserData,
	reflections map[string]WeeklyReflection,
) *UserData {
	if user == nil {
		return nil
	}

	view := *user
	view.Portfolio = Portfolios{
		Serve: withWeeklyNotesForSkill(user, user.Portfolio.Serve, reflections),
		Smash: withWeeklyNotesForSkill(user, user.Portfolio.Smash, reflections),
		Clear: withWeeklyNotesForSkill(user, user.Portfolio.Clear, reflections),
		Lift:  withWeeklyNotesForSkill(user, user.Portfolio.Lift, reflections),
	}
	return &view
}

func withWeeklyNotesForSkill(
	user *UserData,
	works map[string]Work,
	reflections map[string]WeeklyReflection,
) map[string]Work {
	if works == nil {
		return nil
	}

	view := make(map[string]Work, len(works))
	for key, work := range works {
		view[key] = work
		at, err := time.Parse("2006-01-02-15-04", work.DateTime)
		if err != nil {
			continue
		}
		weekly, ok := reflections[ISOWeek(at)]
		if !ok || weekly.UserID != user.ID {
			continue
		}
		work.Reflection = weekly.Note
		work.Preview = weekly.Preview
		view[key] = work
	}
	return view
}

// SetWeeklyReflectionNotes writes the notes it is given and leaves the rest of
// the week's record as it was. The two notes are written from separate editors,
// often days apart, so this merges the named fields instead of replacing the
// document: saving a reflection must never blank a preview, or the other way
// round.
func (client *FirestoreClient) SetWeeklyReflectionNotes(
	userID, week string,
	notes map[WeeklyNoteField]string,
) (*WeeklyReflection, error) {
	// user_id and week are written every time, so a document created by
	// whichever note came first is still a complete record for
	// ListWeeklyReflections to read back.
	fields := map[string]interface{}{
		"user_id":    userID,
		"week":       week,
		"updated_at": time.Now().UTC(),
	}
	paths := []firestore.FieldPath{{"user_id"}, {"week"}, {"updated_at"}}
	for field, text := range notes {
		fields[string(field)] = text
		paths = append(paths, firestore.FieldPath{string(field)})
	}

	doc := client.WeeklyReflections.Doc(client.weeklyReflectionDocID(userID, week))
	if _, err := doc.Set(*client.Ctx, fields, firestore.Merge(paths...)); err != nil {
		return nil, fmt.Errorf("write weekly reflection: %w", err)
	}

	// Read back rather than assemble the answer here: the caller hands it to
	// the browser as the whole week, so the note this save did not touch has to
	// come with it.
	saved, err := client.GetWeeklyReflection(userID, week)
	if err != nil {
		return nil, fmt.Errorf("read back weekly reflection: %w", err)
	}
	if saved == nil {
		return nil, fmt.Errorf("weekly reflection %s is missing right after being written", week)
	}
	return saved, nil
}
