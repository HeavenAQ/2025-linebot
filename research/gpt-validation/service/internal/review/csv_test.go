package review

import (
	"bytes"
	"encoding/csv"
	"reflect"
	"strings"
	"testing"
	"time"
)

func readCSV(t *testing.T, raw []byte) [][]string {
	t.Helper()
	if !bytes.HasPrefix(raw, []byte("\xef\xbb\xbf")) {
		t.Fatalf("missing UTF-8 BOM: %q", raw[:min(len(raw), 16)])
	}
	rows, err := csv.NewReader(bytes.NewReader(raw[3:])).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

func exportFixture() ExportData {
	now := time.Date(2026, 9, 17, 0, 0, 0, 0, time.UTC)
	a := testItem("serve-a", 3)
	a.GPTFlaggedCriteria = []string{"c1", "c2", "c3"}
	b := testItem("smash-b", 0)
	b.Skill = "smash"
	b.TotalGrade = nil
	failed := testItem("serve-failed", 0)
	failed.Status = "failed"
	failed.Eligible = false

	step2 := func(s float64) *float64 { return &s }
	return ExportData{
		BatchID:   testBatch,
		ExpertIDs: []string{"expert_b", "expert_a"},
		Items:     []Item{b, failed, a},
		Ratings: []Rating{
			{ItemID: "serve-a", ExpertID: "expert_a", Skill: "serve", BatchID: testBatch,
				NeedsImprovement: map[string]bool{"c1": true, "c2": false, "c3": true, "c4": false, "c5": false, "c6": false},
				Step1SubmittedAt: &now, Step1Seconds: 31.5,
				CueRatings:   map[string]string{"1": "correct", "2": "partial", "3": "incorrect"},
				OverallScore: ptrInt(2), Comment: "=HYPERLINK(\"x\")", Step2SubmittedAt: &now, Step2Seconds: step2(20), Completed: true},
			{ItemID: "serve-a", ExpertID: "expert_b", Skill: "serve", BatchID: testBatch,
				NeedsImprovement: map[string]bool{"c1": false, "c2": false, "c3": false, "c4": true, "c5": false, "c6": false},
				Step1SubmittedAt: &now, Step1Seconds: 12},
		},
	}
}

func TestItemsCSV(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteItemsCSV(&buf, exportFixture()); err != nil {
		t.Fatal(err)
	}
	rows := readCSV(t, buf.Bytes())
	wantHeader := []string{"item_id", "display_code", "skill", "source_file", "workbook_id", "status", "eligible", "coaching_source", "coaching_model", "handedness", "total_grade", "n_cues", "gpt_flagged_criteria"}
	if !reflect.DeepEqual(rows[0], wantHeader) {
		t.Fatalf("header %v", rows[0])
	}
	if len(rows) != 4 {
		t.Fatalf("want 3 item rows, got %d", len(rows)-1)
	}
	want := []string{"serve-a", "SV-serve-a", "serve", "secret_student_file.mp4", "EG01", "ready", "1", "openai", "gpt-test", "right", "71.5", "3", "c1;c2;c3"}
	if !reflect.DeepEqual(rows[1], want) {
		t.Errorf("row %v\nwant %v", rows[1], want)
	}
	if rows[2][0] != "serve-failed" || rows[2][6] != "0" {
		t.Errorf("ineligible row %v", rows[2])
	}
	if rows[3][10] != "" || rows[3][11] != "0" || rows[3][12] != "" {
		t.Errorf("no-grade/no-cue row %v", rows[3])
	}
}

func TestCriteriaCSV(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteCriteriaCSV(&buf, exportFixture()); err != nil {
		t.Fatal(err)
	}
	rows := readCSV(t, buf.Bytes())
	if !reflect.DeepEqual(rows[0], []string{"item_id", "skill", "expert_id", "criterion_id", "criterion_name", "expert_needs_improvement", "gpt_flagged"}) {
		t.Fatalf("header %v", rows[0])
	}
	// Both experts submitted step 1 on serve-a only: 2 × 6 rows.
	if len(rows) != 13 {
		t.Fatalf("rows %d: %v", len(rows)-1, rows)
	}
	if !reflect.DeepEqual(rows[1], []string{"serve-a", "serve", "expert_a", "c1", "檢核點1", "1", "1"}) {
		t.Errorf("row 1 %v", rows[1])
	}
	if !reflect.DeepEqual(rows[10], []string{"serve-a", "serve", "expert_b", "c4", "檢核點4", "1", "0"}) {
		t.Errorf("row 10 %v", rows[10])
	}
}

func TestCuesCSV(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteCuesCSV(&buf, exportFixture()); err != nil {
		t.Fatal(err)
	}
	rows := readCSV(t, buf.Bytes())
	if !reflect.DeepEqual(rows[0], []string{"item_id", "skill", "expert_id", "cue_index", "criterion_id", "cue_rating", "cue_score"}) {
		t.Fatalf("header %v", rows[0])
	}
	want := [][]string{
		{"serve-a", "serve", "expert_a", "1", "c1", "correct", "2"},
		{"serve-a", "serve", "expert_a", "2", "c2", "partial", "1"},
		{"serve-a", "serve", "expert_a", "3", "c3", "incorrect", "0"},
	}
	if !reflect.DeepEqual(rows[1:], want) {
		t.Errorf("rows %v", rows[1:])
	}
}

func TestOverallCSV(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteOverallCSV(&buf, exportFixture()); err != nil {
		t.Fatal(err)
	}
	rows := readCSV(t, buf.Bytes())
	if !reflect.DeepEqual(rows[0], []string{"item_id", "skill", "expert_id", "overall_score", "comment", "step1_seconds", "step2_seconds", "completed"}) {
		t.Fatalf("header %v", rows[0])
	}
	want := [][]string{
		{"serve-a", "serve", "expert_a", "2", "'=HYPERLINK(\"x\")", "31.5", "20", "1"},
		{"serve-a", "serve", "expert_b", "", "", "12", "", "0"},
		{"smash-b", "smash", "expert_a", "", "", "", "", "0"},
		{"smash-b", "smash", "expert_b", "", "", "", "", "0"},
	}
	if !reflect.DeepEqual(rows[1:], want) {
		t.Errorf("rows\n%v\nwant\n%v", rows[1:], want)
	}
}

func TestExportEndpoint(t *testing.T) {
	env := newTestEnv(t, testItem("serve-a", 1))
	for name := range exporters {
		rec := env.do("GET", "/api/admin/export/"+name, "code-admin-9999", nil)
		if rec.Code != 200 {
			t.Errorf("%s: %d", name, rec.Code)
			continue
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/csv") {
			t.Errorf("%s content type %q", name, ct)
		}
		if !strings.Contains(rec.Header().Get("Content-Disposition"), name) {
			t.Errorf("%s disposition %q", name, rec.Header().Get("Content-Disposition"))
		}
		readCSV(t, rec.Body.Bytes())
	}
	if rec := env.do("GET", "/api/admin/export/secrets.csv", "code-admin-9999", nil); rec.Code != 404 {
		t.Errorf("unknown export: %d", rec.Code)
	}
}

func TestCueScore(t *testing.T) {
	for rating, want := range map[string]int{"correct": 2, "partial": 1, "incorrect": 0} {
		if got, ok := CueScore(rating); !ok || got != want {
			t.Errorf("CueScore(%s) = %d %v", rating, got, ok)
		}
	}
	if _, ok := CueScore("Correct"); ok {
		t.Error("case-insensitive cue rating accepted")
	}
}
