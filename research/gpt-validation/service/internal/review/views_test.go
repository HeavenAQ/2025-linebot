package review

import (
	"encoding/json"
	"testing"
	"time"
)

func jsonKeys(t *testing.T, v any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestItemViewBeforeStep1HidesGPT(t *testing.T) {
	item := testItem("serve-a", 2)
	sign := func(p string) string { return "https://storage.googleapis.com/x/" + p }
	for _, rating := range []*Rating{nil, {ItemID: "serve-a", ExpertID: "expert_a"}} {
		m := jsonKeys(t, BuildItemView(item, rating, sign))
		for _, key := range []string{"gpt_overall_feedback", "gpt_cues", "feedback_video_url", "rating"} {
			if _, ok := m[key]; ok {
				t.Errorf("pre-step-1 view has %s", key)
			}
		}
		for _, key := range []string{"item_id", "display_code", "skill", "criteria", "detected_overlay_url", "step1_done", "completed"} {
			if _, ok := m[key]; !ok {
				t.Errorf("pre-step-1 view missing %s", key)
			}
		}
		raw, _ := json.Marshal(m)
		assertNoForbidden(t, string(raw))
		for _, leak := range []string{"OVERALL-GPT-TEXT", "CUE-FEEDBACK", "CUE-TITLE", "feedback.mp4"} {
			if contains(string(raw), leak) {
				t.Errorf("pre-step-1 view leaks %s", leak)
			}
		}
	}
}

func TestItemViewAfterStep1(t *testing.T) {
	item := testItem("serve-a", 2)
	now := time.Now()
	rating := &Rating{ItemID: "serve-a", ExpertID: "expert_a", NeedsImprovement: allNeeds(true), Step1SubmittedAt: &now}
	m := jsonKeys(t, BuildItemView(item, rating, func(p string) string { return "https://storage.googleapis.com/x/" + p }))
	if m["gpt_overall_feedback"] != "OVERALL-GPT-TEXT" {
		t.Errorf("feedback %v", m["gpt_overall_feedback"])
	}
	cues, _ := m["gpt_cues"].([]any)
	if len(cues) != 2 {
		t.Fatalf("cues %v", m["gpt_cues"])
	}
	cue := cues[0].(map[string]any)
	if len(cue) != 4 || cue["index"] != float64(1) || cue["criterion_id"] != "c1" || cue["title"] != "CUE-TITLE" || cue["feedback"] != "CUE-FEEDBACK" {
		t.Errorf("cue %v", cue)
	}
	if m["feedback_video_url"] != "https://storage.googleapis.com/x/gpt-validation/batch-test/renders/serve-a/feedback.mp4" {
		t.Errorf("feedback url %v", m["feedback_video_url"])
	}
	r, _ := m["rating"].(map[string]any)
	if r == nil || r["needs_improvement"].(map[string]any)["c1"] != true {
		t.Errorf("rating %v", m["rating"])
	}
	for _, key := range []string{"expert_id", "item_id", "batch_id", "step1_seconds", "step2_seconds"} {
		if _, ok := r[key]; ok {
			t.Errorf("rating view includes %s", key)
		}
	}
	raw, _ := json.Marshal(m)
	assertNoForbidden(t, string(raw))

	noCues := testItem("serve-z", 0)
	m = jsonKeys(t, BuildItemView(noCues, rating, func(string) string { return "" }))
	if cues, ok := m["gpt_cues"].([]any); !ok || len(cues) != 0 {
		t.Errorf("zero-cue item should have empty gpt_cues array, got %#v", m["gpt_cues"])
	}
}

func TestGetItemEndpoint(t *testing.T) {
	ineligible := testItem("serve-fallback", 1)
	ineligible.Eligible = false
	ineligible.CoachingSource = "deterministic_fallback"
	other := testItem("serve-other", 1)
	other.BatchID = "another"
	env := newTestEnv(t, testItem("serve-a", 1), ineligible, other)
	const a = "code-alpha-1234"

	for _, path := range []string{"/api/items/serve-fallback", "/api/items/serve-other", "/api/items/unknown", "/api/items/serve-a__expert_b", "/api/items/..%2Fx"} {
		if rec := env.do("GET", path, a, nil); rec.Code != 404 {
			t.Errorf("%s: %d", path, rec.Code)
		}
	}
	if rec := env.do("PUT", "/api/items/serve-fallback/step1", a, map[string]any{"needs_improvement": allNeeds(true), "seconds": 1}); rec.Code != 404 {
		t.Errorf("step1 on ineligible: %d", rec.Code)
	}

	// Expert B rates first; expert A must not see any of it.
	env.do("PUT", "/api/items/serve-a/step1", "code-bravo-5678", map[string]any{"needs_improvement": allNeeds(true), "seconds": 1})
	env.do("PUT", "/api/items/serve-a/step2", "code-bravo-5678", map[string]any{"cue_ratings": map[string]string{"1": "incorrect"}, "overall_score": 0, "comment": "B-PRIVATE-COMMENT", "seconds": 2})

	rec := env.do("GET", "/api/items/serve-a", a, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	assertNoForbidden(t, body)
	if contains(body, "B-PRIVATE-COMMENT") || contains(body, "OVERALL-GPT-TEXT") || contains(body, "feedback_video_url") {
		t.Errorf("expert A pre-step-1 view leaks: %s", body)
	}
	m := decodeMap(t, rec)
	if m["detected_overlay_url"] != "https://storage.googleapis.com/signed/gpt-validation/batch-test/renders/serve-a/detected_overlay.mp4" {
		t.Errorf("overlay url %v", m["detected_overlay_url"])
	}

	env.do("PUT", "/api/items/serve-a/step1", a, map[string]any{"needs_improvement": allNeeds(false), "seconds": 1})
	rec = env.do("GET", "/api/items/serve-a", a, nil)
	body = rec.Body.String()
	assertNoForbidden(t, body)
	if contains(body, "B-PRIVATE-COMMENT") || contains(body, "incorrect") {
		t.Errorf("expert A sees expert B's rating: %s", body)
	}
	if !contains(body, "OVERALL-GPT-TEXT") {
		t.Errorf("GPT not revealed after step 1: %s", body)
	}
}
