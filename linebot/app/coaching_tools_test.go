package app

import (
	"strings"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// The learner a tool reads is bound by the caller. If any tool took an
// identifier, a question -- or text quoted inside one -- could name someone
// else's record.
func TestNoCoachingToolTakesAnIdentifier(t *testing.T) {
	app := &App{}
	for _, tool := range app.CoachingTools("U1") {
		properties, ok := tool.Parameters["properties"].(map[string]any)
		if !ok {
			t.Fatalf("%s has no properties", tool.Name)
		}
		for name := range properties {
			lowered := strings.ToLower(name)
			for _, forbidden := range []string{"user", "id", "learner", "student"} {
				if strings.Contains(lowered, forbidden) {
					t.Errorf("%s accepts %q, which could point at another learner", tool.Name, name)
				}
			}
		}
	}
}

// Strict tools need a closed schema, or OpenAI rejects the definition and the
// coach silently loses every lookup.
func TestCoachingToolSchemasAreClosed(t *testing.T) {
	app := &App{}
	tools := app.CoachingTools("U1")
	if len(tools) == 0 {
		t.Fatal("no tools offered")
	}
	for _, tool := range tools {
		if tool.Parameters["additionalProperties"] != false {
			t.Errorf("%s allows extra properties", tool.Name)
		}
		properties, _ := tool.Parameters["properties"].(map[string]any)
		required, _ := tool.Parameters["required"].([]any)
		if len(required) != len(properties) {
			t.Errorf("%s lists %d of %d properties as required", tool.Name, len(required), len(properties))
		}
		if tool.Description == "" || tool.Handler == nil {
			t.Errorf("%s is missing a description or handler", tool.Name)
		}
	}
}

func TestParseArgsRejectsUnknownSkillsAndBoundsTheLimit(t *testing.T) {
	if _, err := parseArgs(`{"skill":"golf"}`); err == nil {
		t.Error("an unsupported skill should be refused before it reaches Firestore")
	}
	if _, err := parseArgs(`not json`); err == nil {
		t.Error("malformed arguments should be refused")
	}
	for _, tc := range []struct{ in, want int }{{0, 5}, {-3, 5}, {100, 5}, {7, 7}} {
		args, err := parseArgs(`{"skill":"smash","limit":` + itoa(tc.in) + `}`)
		if err != nil {
			t.Fatal(err)
		}
		if args.Limit != tc.want {
			t.Errorf("limit %d became %d, want %d", tc.in, args.Limit, tc.want)
		}
	}
}

// A video sent with no question still has to read as something in the history.
func TestChatQuestionFallsBackToADescription(t *testing.T) {
	if got := chatQuestion(db.AnalysisJob{Question: "  "}); got == "" || strings.TrimSpace(got) == "" {
		t.Errorf("empty question produced %q", got)
	}
	if got := chatQuestion(db.AnalysisJob{Question: "這球哪裡要改"}); got != "這球哪裡要改" {
		t.Errorf("question was rewritten to %q", got)
	}
}

func itoa(value int) string {
	if value < 0 {
		return "-" + itoa(-value)
	}
	if value < 10 {
		return string(rune('0' + value))
	}
	return itoa(value/10) + string(rune('0'+value%10))
}

// The coaching pass runs against an attempt that is already recorded and
// already answered, so it must add only its own results.
func TestMergeCoachingTouchesOnlyCoachingFields(t *testing.T) {
	outcome := &commons.AnalysisOutcome{
		CoachingCues:    []commons.CoachingCue{{Title: "手腕發力", Feedback: "擊球瞬間再快一點"}},
		OverallFeedback: "整體不錯",
		FeedbackVideo:   commons.MediaRef{ObjectPath: "analyses/v1/u/r/student_corrected.mp4"},
		// A second pass also produces these; none of them may be written.
		Grade:        commons.GradingOutcome{TotalGrade: 1},
		StudentVideo: commons.MediaRef{ObjectPath: "should-not-be-written.mp4"},
	}
	paths := db.CoachingUpdatePaths(db.AnalysisJob{Skill: "smash", WorkDate: "2026-09-19-10-00"}, outcome)
	want := map[string]bool{"coaching_cues": true, "ai_note": true, "feedback_video": true}
	for _, path := range paths {
		field := path[len(path)-1]
		if !want[field] {
			t.Errorf("coaching pass would overwrite %q", field)
		}
		delete(want, field)
	}
	if len(want) > 0 {
		t.Errorf("coaching pass never writes %v", want)
	}
}
