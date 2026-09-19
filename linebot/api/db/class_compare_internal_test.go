package db

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

func learner(skill string, totals ...float64) UserData {
	works := map[string]Work{}
	for i, total := range totals {
		works[string(rune('a'+i))] = Work{
			AnalysisStatus: "completed",
			GradingOutcome: commons.GradingOutcome{
				TotalGrade:     total,
				GradingDetails: []commons.GradingDetail{{Description: "手腕發力", Grade: total / 4, Maximum: 25}},
			},
		}
	}
	user := UserData{Portfolio: Portfolios{
		Serve: map[string]Work{}, Smash: map[string]Work{}, Clear: map[string]Work{}, Lift: map[string]Work{},
	}}
	switch skill {
	case "serve":
		user.Portfolio.Serve = works
	case "smash":
		user.Portfolio.Smash = works
	}
	return user
}

// A learner who uploads ten times must not outrank one who uploaded twice and
// scored higher, so everyone is counted by their own best attempt.
func TestClassBestAttemptsCountsEachLearnerOnce(t *testing.T) {
	users := []UserData{learner("smash", 40, 71.5, 55), learner("smash", 80), learner("serve", 99)}
	bests := classBestAttempts(users, "smash")
	if len(bests) != 2 {
		t.Fatalf("got %d learners, want the two with smash attempts", len(bests))
	}
	totals := map[float64]bool{bests[0].total: true, bests[1].total: true}
	if !totals[71.5] || !totals[80] {
		t.Errorf("kept the wrong attempts: %+v", bests)
	}
}

func TestClassBestAttemptsSkipsPendingAndUngradedWork(t *testing.T) {
	user := UserData{Portfolio: Portfolios{Smash: map[string]Work{
		"pending":  {AnalysisStatus: "pending", GradingOutcome: commons.GradingOutcome{TotalGrade: 99}},
		"failed":   {AnalysisStatus: "failed", GradingOutcome: commons.GradingOutcome{TotalGrade: 98}},
		"ungraded": {AnalysisStatus: "completed"},
		"real":     {AnalysisStatus: "completed", GradingOutcome: commons.GradingOutcome{TotalGrade: 61}},
	}}}
	bests := classBestAttempts([]UserData{user}, "smash")
	if len(bests) != 1 || bests[0].total != 61 {
		t.Fatalf("got %+v, want only the graded attempt", bests)
	}
}

func TestStandingRanksAndPercentiles(t *testing.T) {
	bests := []bestAttempt{{total: 40}, {total: 55}, {total: 71.5}, {total: 80}}
	for _, tc := range []struct {
		score      float64
		rank       int
		percentile float64
	}{
		{80, 1, 100},  // the best attempt in the class
		{71.5, 2, 75}, // better than two of four
		{40, 4, 25},   // the lowest is still counted as matching itself
		{90, 1, 100},  // a new personal best above everyone
	} {
		got := standing(bests, "smash", tc.score)
		if got.Rank != tc.rank || math.Abs(got.Percentile-tc.percentile) > 0.001 {
			t.Errorf("score %.1f: rank %d percentile %.1f, want rank %d percentile %.1f",
				tc.score, got.Rank, got.Percentile, tc.rank, tc.percentile)
		}
		if got.Classmates != 4 || got.ClassBest != 80 || math.Abs(got.ClassMean-61.625) > 0.001 {
			t.Errorf("score %.1f: class summary wrong: %+v", tc.score, got)
		}
	}
}

func TestStandingWithoutClassmates(t *testing.T) {
	got := standing(nil, "serve", 50)
	if got.Rank != 0 || got.Percentile != 0 || got.Classmates != 0 {
		t.Errorf("an empty class should not invent a ranking: %+v", got)
	}
}

// The comparison a learner sees must never carry whose attempt it was.
func TestClassBestAttemptCarriesNoIdentity(t *testing.T) {
	best := ClassBestAttempt{Skill: "smash", TotalGrade: 80,
		Criteria: []commons.GradingDetail{{Description: "手腕發力", Grade: 20, Maximum: 25}}}
	for _, field := range []string{"user", "id", "name"} {
		if hasJSONField(best, field) {
			t.Errorf("ClassBestAttempt exposes %q", field)
		}
	}
}

func hasJSONField(value any, field string) bool {
	encoded, err := jsonMarshal(value)
	if err != nil {
		return false
	}
	return containsKey(encoded, field)
}

func jsonMarshal(value any) (map[string]any, error) {
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	return out, json.Unmarshal(raw, &out)
}

func containsKey(value map[string]any, key string) bool {
	for name := range value {
		if strings.Contains(strings.ToLower(name), key) {
			return true
		}
	}
	return false
}
