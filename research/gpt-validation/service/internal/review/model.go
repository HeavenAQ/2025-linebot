package review

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"sort"
	"time"
)

// Firestore collections (see DESIGN.md).
const (
	ItemsCollection   = "gpt_validation_items"
	RatingsCollection = "gpt_validation_ratings"
)

// Cue rating values and their CSV scores.
const (
	CueCorrect   = "correct"
	CuePartial   = "partial"
	CueIncorrect = "incorrect"
)

// CueScore maps a cue rating to its numeric score (correct=2, partial=1,
// incorrect=0). ok is false for unknown values.
func CueScore(rating string) (score int, ok bool) {
	switch rating {
	case CueCorrect:
		return 2, true
	case CuePartial:
		return 1, true
	case CueIncorrect:
		return 0, true
	}
	return 0, false
}

// Criterion is one rubric checkpoint.
type Criterion struct {
	ID      string
	NameZh  string
	Grade   *float64
	Maximum *float64
}

// Cue is one GPT improvement cue.
type Cue struct {
	Index       int
	CriterionID string
	Title       string
	Feedback    string
}

// Item is a gpt_validation_items document.
type Item struct {
	ItemID             string
	BatchID            string
	Skill              string
	SourceFile         string
	WorkbookID         string
	DisplayCode        string
	Status             string
	Error              string
	Attempts           int
	Eligible           bool
	CoachingSource     string
	CoachingModel      string
	Handedness         string
	TotalGrade         *float64
	Criteria           []Criterion
	GPTOverallFeedback string
	GPTCues            []Cue
	GPTFlaggedCriteria []string
	Media              map[string]string
}

// ReviewableCoachingSources are the coaching outcomes experts rate: GPT's own
// feedback, and the score gate's "no issues found" (so experts can judge
// whether staying silent was right). Fallback text written without GPT is not.
var ReviewableCoachingSources = map[string]bool{"openai": true, "score_gate": true}

// ServableTo reports whether experts may see this item in batchID. The stored
// eligible flag is re-checked against its definition so a stale flag can never
// expose a fallback or failed item.
func (it Item) ServableTo(batchID string) bool {
	return it.BatchID == batchID && it.Eligible && it.Status == "ready" && ReviewableCoachingSources[it.CoachingSource]
}

// CriterionIDs lists rubric ids in rubric order.
func (it Item) CriterionIDs() []string {
	ids := make([]string, 0, len(it.Criteria))
	for _, c := range it.Criteria {
		ids = append(ids, c.ID)
	}
	return ids
}

// FlaggedSet is the set of criteria GPT raised a cue for.
func (it Item) FlaggedSet() map[string]bool {
	set := map[string]bool{}
	for _, id := range it.GPTFlaggedCriteria {
		set[id] = true
	}
	if len(it.GPTFlaggedCriteria) == 0 {
		for _, c := range it.GPTCues {
			if c.CriterionID != "" {
				set[c.CriterionID] = true
			}
		}
	}
	return set
}

// Rating is a gpt_validation_ratings document.
type Rating struct {
	ItemID           string
	ExpertID         string
	Skill            string
	BatchID          string
	NeedsImprovement map[string]bool
	Step1SubmittedAt *time.Time
	Step1Seconds     float64
	CueRatings       map[string]string
	OverallScore     *int
	Comment          string
	Step2SubmittedAt *time.Time
	Step2Seconds     *float64
	Completed        bool
	UpdatedAt        *time.Time
}

// Step1Done reports whether step 1 has been submitted (and is locked).
func (r *Rating) Step1Done() bool { return r != nil && r.Step1SubmittedAt != nil }

// IsCompleted reports whether step 2 has been submitted.
func (r *Rating) IsCompleted() bool { return r != nil && r.Completed }

// RatingDocID is the ratings document id for one item and expert.
func RatingDocID(itemID, expertID string) string { return itemID + "__" + expertID }

// Step2 is a validated step 2 submission.
type Step2 struct {
	CueRatings   map[string]string
	OverallScore int
	Comment      string
	Seconds      float64
}

// OrderForExpert returns items sorted by sha256(expert_id + ":" + item_id),
// a fixed pseudo-random order that differs between experts.
func OrderForExpert(expertID string, items []Item) []Item {
	type keyed struct {
		key  string
		item Item
	}
	ks := make([]keyed, len(items))
	for i, it := range items {
		sum := sha256.Sum256([]byte(expertID + ":" + it.ItemID))
		ks[i] = keyed{key: hex.EncodeToString(sum[:]), item: it}
	}
	sort.SliceStable(ks, func(i, j int) bool {
		if ks[i].key != ks[j].key {
			return ks[i].key < ks[j].key
		}
		return ks[i].item.ItemID < ks[j].item.ItemID
	})
	out := make([]Item, len(ks))
	for i, k := range ks {
		out[i] = k.item
	}
	return out
}

// --- decoding from Firestore document data -------------------------------

func asString(v any) string {
	s, _ := v.(string)
	return s
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func asFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, !math.IsNaN(x) && !math.IsInf(x, 0)
	case float32:
		return float64(x), true
	case int64:
		return float64(x), true
	case int:
		return float64(x), true
	case int32:
		return float64(x), true
	}
	return 0, false
}

func asFloatPtr(v any) *float64 {
	if f, ok := asFloat(v); ok {
		return &f
	}
	return nil
}

func asInt(v any) (int, bool) {
	f, ok := asFloat(v)
	if !ok || f != math.Trunc(f) {
		return 0, false
	}
	return int(f), true
}

func asTimePtr(v any) *time.Time {
	if t, ok := v.(time.Time); ok && !t.IsZero() {
		return &t
	}
	return nil
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func asSlice(v any) []any {
	s, _ := v.([]any)
	return s
}

// ItemFromData decodes an items document. Unknown or mistyped fields decode to
// zero values rather than failing the whole listing.
func ItemFromData(docID string, d map[string]any) Item {
	it := Item{
		ItemID:             asString(d["item_id"]),
		BatchID:            asString(d["batch_id"]),
		Skill:              asString(d["skill"]),
		SourceFile:         asString(d["source_file"]),
		WorkbookID:         asString(d["workbook_id"]),
		DisplayCode:        asString(d["display_code"]),
		Status:             asString(d["status"]),
		Error:              asString(d["error"]),
		Eligible:           asBool(d["eligible"]),
		CoachingSource:     asString(d["coaching_source"]),
		CoachingModel:      asString(d["coaching_model"]),
		Handedness:         asString(d["handedness"]),
		TotalGrade:         asFloatPtr(d["total_grade"]),
		GPTOverallFeedback: asString(d["gpt_overall_feedback"]),
		Media:              map[string]string{},
	}
	if it.ItemID == "" {
		it.ItemID = docID
	}
	it.Attempts, _ = asInt(d["attempts"])
	for _, raw := range asSlice(d["criteria"]) {
		m := asMap(raw)
		if m == nil {
			continue
		}
		it.Criteria = append(it.Criteria, Criterion{
			ID:      asString(m["id"]),
			NameZh:  asString(m["name_zh"]),
			Grade:   asFloatPtr(m["grade"]),
			Maximum: asFloatPtr(m["maximum"]),
		})
	}
	for _, raw := range asSlice(d["gpt_cues"]) {
		m := asMap(raw)
		if m == nil {
			continue
		}
		idx, _ := asInt(m["index"])
		it.GPTCues = append(it.GPTCues, Cue{
			Index:       idx,
			CriterionID: asString(m["criterion_id"]),
			Title:       asString(m["title"]),
			Feedback:    asString(m["feedback"]),
		})
	}
	for _, raw := range asSlice(d["gpt_flagged_criteria"]) {
		if s := asString(raw); s != "" {
			it.GPTFlaggedCriteria = append(it.GPTFlaggedCriteria, s)
		}
	}
	for k, v := range asMap(d["media"]) {
		if s := asString(v); s != "" {
			it.Media[k] = s
		}
	}
	return it
}

// RatingFromData decodes a ratings document.
func RatingFromData(d map[string]any) Rating {
	r := Rating{
		ItemID:           asString(d["item_id"]),
		ExpertID:         asString(d["expert_id"]),
		Skill:            asString(d["skill"]),
		BatchID:          asString(d["batch_id"]),
		Step1SubmittedAt: asTimePtr(d["step1_submitted_at"]),
		Comment:          asString(d["comment"]),
		Step2SubmittedAt: asTimePtr(d["step2_submitted_at"]),
		Step2Seconds:     asFloatPtr(d["step2_seconds"]),
		Completed:        asBool(d["completed"]),
		UpdatedAt:        asTimePtr(d["updated_at"]),
	}
	r.Step1Seconds, _ = asFloat(d["step1_seconds"])
	if m := asMap(d["needs_improvement"]); m != nil {
		r.NeedsImprovement = map[string]bool{}
		for k, v := range m {
			if b, ok := v.(bool); ok {
				r.NeedsImprovement[k] = b
			}
		}
	}
	if m := asMap(d["cue_ratings"]); m != nil {
		r.CueRatings = map[string]string{}
		for k, v := range m {
			if s := asString(v); s != "" {
				r.CueRatings[k] = s
			}
		}
	}
	if score, ok := asInt(d["overall_score"]); ok {
		r.OverallScore = &score
	}
	return r
}

// Step1Data is the document written when step 1 is submitted.
func (r Rating) Step1Data() map[string]any {
	ni := make(map[string]any, len(r.NeedsImprovement))
	for k, v := range r.NeedsImprovement {
		ni[k] = v
	}
	return map[string]any{
		"item_id":            r.ItemID,
		"expert_id":          r.ExpertID,
		"skill":              r.Skill,
		"batch_id":           r.BatchID,
		"needs_improvement":  ni,
		"step1_submitted_at": *r.Step1SubmittedAt,
		"step1_seconds":      r.Step1Seconds,
		"completed":          false,
		"updated_at":         *r.Step1SubmittedAt,
	}
}
