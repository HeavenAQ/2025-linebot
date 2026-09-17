package review

import "testing"

func TestAllowedObjectPath(t *testing.T) {
	allowed := []string{
		"gpt-validation/batch/renders/serve-abc/detected_overlay.mp4",
		"gpt-validation/x.mp4",
	}
	denied := []string{
		"",
		"analyses/user/video.mp4",
		"/gpt-validation/batch/x.mp4",
		"gpt-validation/../analyses/x.mp4",
		"gpt-validation/batch/..",
		"gpt-validation..x/y.mp4",
		"GPT-VALIDATION/batch/x.mp4",
		"experts/gpt-validation/x.mp4",
		"gpt-validation/a\nb.mp4",
	}
	for _, p := range allowed {
		if !AllowedObjectPath(p) {
			t.Errorf("denied %q", p)
		}
	}
	for _, p := range denied {
		if AllowedObjectPath(p) {
			t.Errorf("allowed %q", p)
		}
	}
}

func TestNormalizeObjectPath(t *testing.T) {
	cases := []struct {
		in   string
		want string
		ok   bool
	}{
		{"gpt-validation/b/renders/i/feedback.mp4", "gpt-validation/b/renders/i/feedback.mp4", true},
		{"gs://nstc-2025-storage/gpt-validation/b/renders/i/feedback.mp4", "gpt-validation/b/renders/i/feedback.mp4", true},
		{"gs://other-bucket/gpt-validation/b/x.mp4", "", false},
		{"gs://nstc-2025-storage/analyses/x.mp4", "analyses/x.mp4", false},
		{"gs://nstc-2025-storage", "", false},
	}
	for _, c := range cases {
		got, ok := normalizeObjectPath("nstc-2025-storage", c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("normalize(%q) = %q, %v", c.in, got, ok)
		}
	}
}

func TestSignerRefusesOutsidePrefix(t *testing.T) {
	// A GCSSigner with a nil client must refuse before touching the client.
	g := NewGCSSigner(nil, "nstc-2025-storage", "sa@example.iam.gserviceaccount.com")
	for _, p := range []string{"analyses/u/v.mp4", "gpt-validation/../x", "gs://other/gpt-validation/x.mp4"} {
		if _, err := g.SignedURL(p); err == nil {
			t.Errorf("signed %q", p)
		}
	}
	if _, err := (NoopSigner{}).SignedURL("analyses/x.mp4"); err == nil {
		t.Error("noop signer accepted disallowed path")
	}
	// Unsignable media never breaks the item view; the URL is just empty.
	env := newTestEnv(t)
	item := testItem("serve-a", 0)
	item.Media["detected_overlay"] = "analyses/private.mp4"
	env.store = NewMemoryStore(item)
	env.server.store = env.store
	rec := env.do("GET", "/api/items/serve-a", "code-alpha-1234", nil)
	if m := decodeMap(t, rec); rec.Code != 200 || m["detected_overlay_url"] != "" {
		t.Errorf("status %d body %v", rec.Code, m)
	}
}

func TestItemFromDataDecoding(t *testing.T) {
	d := map[string]any{
		"item_id": "serve-1", "batch_id": "b", "skill": "serve", "source_file": "f.mp4", "workbook_id": "EG01",
		"display_code": "SV-001", "status": "ready", "eligible": true, "coaching_source": "openai",
		"total_grade": int64(80), "attempts": int64(2),
		"criteria":             []any{map[string]any{"id": "c1", "name_zh": "握拍", "grade": 1.5, "maximum": int64(2)}},
		"gpt_cues":             []any{map[string]any{"index": int64(1), "criterion_id": "c1", "title": "t", "feedback": "f"}},
		"gpt_flagged_criteria": []any{"c1"},
		"media":                map[string]any{"detected_overlay": "gpt-validation/b/renders/serve-1/detected_overlay.mp4"},
	}
	it := ItemFromData("serve-1", d)
	if it.WorkbookID != "EG01" || *it.TotalGrade != 80 || it.Attempts != 2 || len(it.Criteria) != 1 || it.GPTCues[0].Index != 1 || !it.ServableTo("b") {
		t.Errorf("decoded %+v", it)
	}
	r := RatingFromData(map[string]any{"overall_score": int64(3), "cue_ratings": map[string]any{"1": "correct"}, "needs_improvement": map[string]any{"c1": true}})
	if *r.OverallScore != 3 || r.CueRatings["1"] != "correct" || !r.NeedsImprovement["c1"] || r.Step1Done() {
		t.Errorf("rating %+v", r)
	}
}
