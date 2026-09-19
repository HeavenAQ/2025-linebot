package review

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func step2Req(cues map[string]string, score any) map[string]any {
	return map[string]any{"cue_ratings": cues, "overall_score": score, "comment": "ok", "seconds": 12.5}
}

func TestParseStep1(t *testing.T) {
	item := testItem("serve-a", 2)
	good := `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":42.3}`
	needs, secs, err := ParseStep1(item, strings.NewReader(good))
	if err != nil {
		t.Fatal(err)
	}
	if !needs["c1"] || needs["c2"] || len(needs) != 6 || secs != 42.3 {
		t.Errorf("parsed %v %v", needs, secs)
	}

	bad := map[string]string{
		"missing key":   `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false},"seconds":1}`,
		"extra key":     `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false,"c7":true},"seconds":1}`,
		"wrong key":     `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"x6":false},"seconds":1}`,
		"string value":  `{"needs_improvement":{"c1":"true","c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":1}`,
		"number value":  `{"needs_improvement":{"c1":1,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":1}`,
		"null value":    `{"needs_improvement":{"c1":null,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":1}`,
		"no map":        `{"seconds":1}`,
		"negative secs": `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":-1}`,
		"missing secs":  `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false}}`,
		"unknown field": `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":1,"expert_id":"expert_b"}`,
		"trailing data": `{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":1}{}`,
		"not json":      `hello`,
	}
	for name, body := range bad {
		if _, _, err := ParseStep1(item, strings.NewReader(body)); err == nil {
			t.Errorf("%s: accepted", name)
		} else if !errors.As(err, new(*ValidationError)) {
			t.Errorf("%s: not a validation error: %v", name, err)
		}
	}

	_, secs, err = ParseStep1(item, strings.NewReader(`{"needs_improvement":{"c1":true,"c2":false,"c3":false,"c4":true,"c5":false,"c6":false},"seconds":999999}`))
	if err != nil || secs != MaxClientSeconds {
		t.Errorf("huge seconds not capped: %v %v", secs, err)
	}
}

func TestParseStep2(t *testing.T) {
	item := testItem("serve-a", 2)
	s, err := ParseStep2(item, strings.NewReader(`{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":3,"comment":"很好","seconds":5}`))
	if err != nil {
		t.Fatal(err)
	}
	if s.OverallScore != 3 || s.CueRatings["2"] != "partial" || s.Comment != "很好" || s.Seconds != 5 {
		t.Errorf("parsed %+v", s)
	}

	longComment := strings.Repeat("字", MaxCommentRunes+1)
	bad := map[string]string{
		"missing cue":   `{"cue_ratings":{"1":"correct"},"overall_score":1,"comment":"","seconds":1}`,
		"extra cue":     `{"cue_ratings":{"1":"correct","2":"partial","3":"correct"},"overall_score":1,"comment":"","seconds":1}`,
		"zero index":    `{"cue_ratings":{"0":"correct","1":"partial"},"overall_score":1,"comment":"","seconds":1}`,
		"bad value":     `{"cue_ratings":{"1":"correct","2":"good"},"overall_score":1,"comment":"","seconds":1}`,
		"non-string":    `{"cue_ratings":{"1":"correct","2":2},"overall_score":1,"comment":"","seconds":1}`,
		"score 4":       `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":4,"comment":"","seconds":1}`,
		"score -1":      `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":-1,"comment":"","seconds":1}`,
		"score float":   `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":2.5,"comment":"","seconds":1}`,
		"score 2.0":     `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":2.0,"comment":"","seconds":1}`,
		"score string":  `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":"2","comment":"","seconds":1}`,
		"score missing": `{"cue_ratings":{"1":"correct","2":"partial"},"comment":"","seconds":1}`,
		"score null":    `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":null,"comment":"","seconds":1}`,
		"long comment":  `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":1,"comment":"` + longComment + `","seconds":1}`,
		"negative secs": `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":1,"comment":"","seconds":-0.1}`,
		"missing cues":  `{"overall_score":1,"comment":"","seconds":1}`,
		"unknown field": `{"cue_ratings":{"1":"correct","2":"partial"},"overall_score":1,"comment":"","seconds":1,"completed":false}`,
	}
	for name, body := range bad {
		if _, err := ParseStep2(item, strings.NewReader(body)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}

	okComment := strings.Repeat("字", MaxCommentRunes)
	if _, err := ParseStep2(item, strings.NewReader(`{"cue_ratings":{"1":"incorrect","2":"partial"},"overall_score":0,"comment":"`+okComment+`","seconds":0}`)); err != nil {
		t.Errorf("2000-char comment rejected: %v", err)
	}

	noCues := testItem("serve-z", 0)
	for _, body := range []string{
		`{"cue_ratings":{},"overall_score":2,"comment":"","seconds":1}`,
		`{"overall_score":2,"seconds":1}`,
	} {
		s, err := ParseStep2(noCues, strings.NewReader(body))
		if err != nil || len(s.CueRatings) != 0 || s.OverallScore != 2 {
			t.Errorf("zero-cue body %s: %+v %v", body, s, err)
		}
	}
	if _, err := ParseStep2(noCues, strings.NewReader(`{"cue_ratings":{"1":"correct"},"overall_score":2,"seconds":1}`)); err == nil {
		t.Error("zero-cue item accepted a cue rating")
	}
}

func TestMemoryStoreLocks(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore(testItem("serve-a", 1))
	now := time.Now().UTC()

	if err := store.SubmitStep2(ctx, "serve-a", "expert_a", Step2{OverallScore: 1, Seconds: 3}, now); !errors.Is(err, ErrStep1Required) {
		t.Fatalf("step 2 before step 1: %v", err)
	}
	r := Rating{ItemID: "serve-a", ExpertID: "expert_a", BatchID: testBatch, Skill: "serve", NeedsImprovement: allNeeds(false), Step1SubmittedAt: &now, Step1Seconds: 9}
	if err := store.SubmitStep1(ctx, r); err != nil {
		t.Fatal(err)
	}
	r2 := r
	r2.NeedsImprovement = allNeeds(true)
	if err := store.SubmitStep1(ctx, r2); !errors.Is(err, ErrStep1Locked) {
		t.Fatalf("second step 1: %v", err)
	}
	got, _ := store.GetRating(ctx, "serve-a", "expert_a")
	if got.NeedsImprovement["c1"] {
		t.Error("step 1 was overwritten")
	}

	// Concurrent step 1 submissions: exactly one wins.
	store2 := NewMemoryStore(testItem("serve-a", 1))
	var wg sync.WaitGroup
	var mu sync.Mutex
	wins := 0
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := store2.SubmitStep1(ctx, r); err == nil {
				mu.Lock()
				wins++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if wins != 1 {
		t.Errorf("%d concurrent step 1 submissions succeeded", wins)
	}

	if err := store.SubmitStep2(ctx, "serve-a", "expert_a", Step2{CueRatings: map[string]string{"1": "correct"}, OverallScore: 2, Seconds: 10}, now); err != nil {
		t.Fatal(err)
	}
	if err := store.SubmitStep2(ctx, "serve-a", "expert_a", Step2{CueRatings: map[string]string{"1": "partial"}, OverallScore: 1, Comment: "revised", Seconds: 5}, now); err != nil {
		t.Fatal(err)
	}
	got, _ = store.GetRating(ctx, "serve-a", "expert_a")
	if !got.Completed || *got.OverallScore != 1 || got.CueRatings["1"] != "partial" || got.Comment != "revised" || *got.Step2Seconds != 15 || got.Step1Seconds != 9 {
		t.Errorf("after re-save: %+v", got)
	}
	if got.NeedsImprovement["c1"] || got.Step1SubmittedAt == nil {
		t.Error("step 2 changed step 1")
	}
}

func TestStepHandlers(t *testing.T) {
	env := newTestEnv(t, testItem("serve-a", 2), testItem("serve-none", 0))
	const a = "code-alpha-1234"
	const b = "code-bravo-5678"

	// Step 2 before step 1 -> 409.
	rec := env.do("PUT", "/api/items/serve-a/step2", a, step2Req(map[string]string{"1": "correct", "2": "correct"}, 3))
	if rec.Code != 409 || decodeMap(t, rec)["error"] != "step1_required" {
		t.Fatalf("step2 before step1: %d %s", rec.Code, rec.Body)
	}

	// Invalid step 1 -> 400, nothing stored.
	rec = env.do("PUT", "/api/items/serve-a/step1", a, map[string]any{"needs_improvement": map[string]bool{"c1": true}, "seconds": 3})
	if rec.Code != 400 {
		t.Fatalf("invalid step1: %d", rec.Code)
	}

	rec = env.do("PUT", "/api/items/serve-a/step1", a, map[string]any{"needs_improvement": allNeeds(true), "seconds": 30})
	if rec.Code != 200 {
		t.Fatalf("step1: %d %s", rec.Code, rec.Body)
	}
	m := decodeMap(t, rec)
	if m["gpt_overall_feedback"] != "OVERALL-GPT-TEXT" || m["feedback_video_url"] == "" || m["rating"] == nil {
		t.Errorf("step1 response does not reveal GPT: %v", m)
	}
	assertNoForbidden(t, rec.Body.String())

	rec = env.do("PUT", "/api/items/serve-a/step1", a, map[string]any{"needs_improvement": allNeeds(false), "seconds": 1})
	if rec.Code != 409 || decodeMap(t, rec)["error"] != "step1_locked" {
		t.Fatalf("second step1: %d %s", rec.Code, rec.Body)
	}

	// Expert B is independent of expert A.
	rec = env.do("GET", "/api/items/serve-a", b, nil)
	if m := decodeMap(t, rec); m["step1_done"] != false || m["gpt_overall_feedback"] != nil {
		t.Errorf("expert B sees A's progress: %v", m)
	}
	if rec := env.do("PUT", "/api/items/serve-a/step2", b, step2Req(map[string]string{"1": "correct", "2": "correct"}, 3)); rec.Code != 409 {
		t.Errorf("expert B step2 without own step1: %d", rec.Code)
	}

	// Bad step 2, then good, then re-save.
	if rec := env.do("PUT", "/api/items/serve-a/step2", a, step2Req(map[string]string{"1": "correct"}, 3)); rec.Code != 400 {
		t.Errorf("incomplete cues: %d", rec.Code)
	}
	if rec := env.do("PUT", "/api/items/serve-a/step2", a, step2Req(map[string]string{"1": "correct", "2": "incorrect"}, 3)); rec.Code != 200 {
		t.Fatalf("step2: %d %s", rec.Code, rec.Body)
	}
	if rec := env.do("PUT", "/api/items/serve-a/step2", a, step2Req(map[string]string{"1": "partial", "2": "incorrect"}, 2)); rec.Code != 200 {
		t.Fatalf("step2 re-save: %d %s", rec.Code, rec.Body)
	}
	stored, err := env.store.GetRating(context.Background(), "serve-a", "expert_a")
	if err != nil || !stored.Completed || *stored.OverallScore != 2 || stored.CueRatings["1"] != "partial" || stored.BatchID != testBatch || stored.Skill != "serve" {
		t.Errorf("stored rating %+v %v", stored, err)
	}
	if _, err := env.store.GetRating(context.Background(), "serve-a", "expert_b"); !errors.Is(err, ErrNotFound) {
		t.Errorf("expert B rating exists: %v", err)
	}

	// Zero-cue item: empty map plus score.
	env.do("PUT", "/api/items/serve-none/step1", a, map[string]any{"needs_improvement": allNeeds(false), "seconds": 1})
	if rec := env.do("PUT", "/api/items/serve-none/step2", a, step2Req(map[string]string{}, 0)); rec.Code != 200 {
		t.Errorf("zero-cue step2: %d %s", rec.Code, rec.Body)
	}

	// Body limit.
	huge := `{"needs_improvement":{},"seconds":1,"pad":"` + strings.Repeat("x", MaxBodyBytes) + `"}`
	if rec := env.do("PUT", "/api/items/serve-a/step2", a, huge); rec.Code != 413 {
		t.Errorf("huge body: %d", rec.Code)
	}

	// Progress reflects both experts.
	rec = env.do("GET", "/api/admin/progress", "code-admin-9999", nil)
	var progress struct {
		Experts []ExpertProgress `json:"experts"`
	}
	decodeInto(t, rec.Body.Bytes(), &progress)
	want := []ExpertProgress{{"expert_a", 2, 2, 2}, {"expert_b", 2, 0, 0}}
	if len(progress.Experts) != 2 || progress.Experts[0] != want[0] || progress.Experts[1] != want[1] {
		t.Errorf("progress %+v", progress.Experts)
	}
}
