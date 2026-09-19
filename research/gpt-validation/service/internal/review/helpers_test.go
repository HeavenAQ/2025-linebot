package review

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

const testBatch = "batch-test"

var testCodes = map[string]string{
	"expert_a": "code-alpha-1234",
	"expert_b": "code-bravo-5678",
	"admin":    "code-admin-9999",
}

type fakeSigner struct{}

func (fakeSigner) SignedURL(p string) (string, error) {
	if !AllowedObjectPath(p) {
		return "", errTestNotSignable
	}
	return "https://storage.googleapis.com/signed/" + p, nil
}

var errTestNotSignable = &ValidationError{Msg: "not signable"}

func testItem(id string, cues int) Item {
	it := Item{
		ItemID:             id,
		BatchID:            testBatch,
		Skill:              "serve",
		SourceFile:         "secret_student_file.mp4",
		WorkbookID:         "EG01",
		DisplayCode:        "SV-" + id,
		Status:             "ready",
		Eligible:           true,
		CoachingSource:     "openai",
		CoachingModel:      "gpt-test",
		Handedness:         "right",
		TotalGrade:         ptrFloat(71.5),
		GPTOverallFeedback: "OVERALL-GPT-TEXT",
		Media: map[string]string{
			"detected_overlay": "gpt-validation/" + testBatch + "/renders/" + id + "/detected_overlay.mp4",
			"feedback_video":   "gpt-validation/" + testBatch + "/renders/" + id + "/feedback.mp4",
			"skeleton_overlay": "gpt-validation/" + testBatch + "/renders/" + id + "/skeleton_overlay.mp4",
		},
	}
	for i := 1; i <= 6; i++ {
		it.Criteria = append(it.Criteria, Criterion{ID: "c" + string(rune('0'+i)), NameZh: "檢核點" + string(rune('0'+i))})
	}
	for i := 1; i <= cues; i++ {
		cid := it.Criteria[i-1].ID
		it.GPTCues = append(it.GPTCues, Cue{Index: i, CriterionID: cid, Title: "CUE-TITLE", Feedback: "CUE-FEEDBACK"})
		it.GPTFlaggedCriteria = append(it.GPTFlaggedCriteria, cid)
	}
	return it
}

func ptrFloat(f float64) *float64 { return &f }
func ptrInt(i int) *int           { return &i }

type testEnv struct {
	t       *testing.T
	store   *MemoryStore
	server  *Server
	handler http.Handler
}

func newTestEnv(t *testing.T, items ...Item) *testEnv {
	t.Helper()
	store := NewMemoryStore(items...)
	static := fstest.MapFS{
		"index.html": {Data: []byte("<!doctype html><title>t</title>")},
		"app.js":     {Data: []byte("'use strict';")},
	}
	srv := NewServer(testBatch, store, fakeSigner{}, NewAuthenticator(testCodes), static, nil)
	srv.now = func() time.Time { return time.Date(2026, 9, 17, 10, 0, 0, 0, time.UTC) }
	return &testEnv{t: t, store: store, server: srv, handler: srv.Handler()}
}

func (e *testEnv) do(method, path, code string, body any) *httptest.ResponseRecorder {
	e.t.Helper()
	var reader *bytes.Reader
	switch b := body.(type) {
	case nil:
		reader = bytes.NewReader(nil)
	case string:
		reader = bytes.NewReader([]byte(b))
	default:
		raw, err := json.Marshal(b)
		if err != nil {
			e.t.Fatal(err)
		}
		reader = bytes.NewReader(raw)
	}
	req := httptest.NewRequest(method, path, reader)
	req.RemoteAddr = "203.0.113.7:5555"
	if code != "" {
		req.Header.Set("X-Review-Code", code)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	e.handler.ServeHTTP(rec, req)
	return rec
}

func decodeMap(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("decode %q: %v", rec.Body.String(), err)
	}
	return m
}

func allNeeds(value bool) map[string]bool {
	m := map[string]bool{}
	for i := 1; i <= 6; i++ {
		m["c"+string(rune('0'+i))] = value
	}
	return m
}

func assertNoForbidden(t *testing.T, body string) {
	t.Helper()
	for _, forbidden := range []string{"source_file", "secret_student_file", "workbook_id", "EG01", "total_grade", "71.5", "coaching_source", "coaching_model", "gpt-test", "skeleton_overlay", "expert_b", "handedness", "gpt_flagged_criteria"} {
		if strings.Contains(body, forbidden) {
			t.Errorf("response leaks %q: %s", forbidden, body)
		}
	}
}

func contains(s, sub string) bool { return strings.Contains(s, sub) }

func decodeInto(t *testing.T, raw []byte, dst any) {
	t.Helper()
	if err := json.Unmarshal(raw, dst); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
}
