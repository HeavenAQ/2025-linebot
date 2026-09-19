package review

import (
	"sort"
	"time"
)

// Response shapes sent to experts. They are built field by field from the
// stored documents so internal fields (source_file, workbook_id, total_grade,
// coaching_source, other experts' ratings, ...) can never leak by accident.

// ItemSummary is one row of GET /api/items.
type ItemSummary struct {
	ItemID      string `json:"item_id"`
	DisplayCode string `json:"display_code"`
	Skill       string `json:"skill"`
	Step1Done   bool   `json:"step1_done"`
	Completed   bool   `json:"completed"`
}

// CriterionView is a rubric checkpoint as shown to experts.
type CriterionView struct {
	ID     string `json:"id"`
	NameZh string `json:"name_zh"`
}

// CueView is a GPT cue as shown to experts after step 1.
type CueView struct {
	Index       int    `json:"index"`
	CriterionID string `json:"criterion_id"`
	Title       string `json:"title"`
	Feedback    string `json:"feedback"`
}

// RatingView is the calling expert's own rating.
type RatingView struct {
	NeedsImprovement map[string]bool   `json:"needs_improvement"`
	Step1SubmittedAt *time.Time        `json:"step1_submitted_at"`
	CueRatings       map[string]string `json:"cue_ratings"`
	OverallScore     *int              `json:"overall_score"`
	Comment          string            `json:"comment"`
	Step2SubmittedAt *time.Time        `json:"step2_submitted_at"`
	Completed        bool              `json:"completed"`
}

// RevealedView holds what only appears once step 1 is submitted.
type RevealedView struct {
	GPTOverallFeedback string      `json:"gpt_overall_feedback"`
	GPTCues            []CueView   `json:"gpt_cues"`
	FeedbackVideoURL   string      `json:"feedback_video_url"`
	Rating             *RatingView `json:"rating"`
}

// ItemView is GET /api/items/{item_id}. The embedded pointer is nil before
// step 1, so none of its keys are serialized.
type ItemView struct {
	ItemID             string          `json:"item_id"`
	DisplayCode        string          `json:"display_code"`
	Skill              string          `json:"skill"`
	Criteria           []CriterionView `json:"criteria"`
	DetectedOverlayURL string          `json:"detected_overlay_url"`
	Step1Done          bool            `json:"step1_done"`
	Completed          bool            `json:"completed"`
	*RevealedView
}

// BuildItemSummaries lists servable items in the expert's fixed order.
func BuildItemSummaries(expertID, batchID string, items []Item, ratings []Rating) []ItemSummary {
	var eligible []Item
	for _, it := range items {
		if it.ServableTo(batchID) {
			eligible = append(eligible, it)
		}
	}
	own := map[string]*Rating{}
	for i := range ratings {
		if ratings[i].ExpertID == expertID {
			own[ratings[i].ItemID] = &ratings[i]
		}
	}
	out := make([]ItemSummary, 0, len(eligible))
	for _, it := range OrderForExpert(expertID, eligible) {
		r := own[it.ItemID]
		out = append(out, ItemSummary{
			ItemID:      it.ItemID,
			DisplayCode: it.DisplayCode,
			Skill:       it.Skill,
			Step1Done:   r.Step1Done(),
			Completed:   r.IsCompleted(),
		})
	}
	return out
}

// BuildItemView builds the expert's item page. rating must be the caller's own
// rating (or nil). sign returns "" when a URL cannot be produced.
func BuildItemView(item Item, rating *Rating, sign func(objectPath string) string) ItemView {
	v := ItemView{
		ItemID:      item.ItemID,
		DisplayCode: item.DisplayCode,
		Skill:       item.Skill,
		Criteria:    make([]CriterionView, 0, len(item.Criteria)),
		Step1Done:   rating.Step1Done(),
		Completed:   rating.IsCompleted(),
	}
	for _, c := range item.Criteria {
		v.Criteria = append(v.Criteria, CriterionView{ID: c.ID, NameZh: c.NameZh})
	}
	if p := item.Media["detected_overlay"]; p != "" {
		v.DetectedOverlayURL = sign(p)
	}
	if !rating.Step1Done() {
		return v
	}

	cues := make([]CueView, 0, len(item.GPTCues))
	for _, c := range item.GPTCues {
		cues = append(cues, CueView{Index: c.Index, CriterionID: c.CriterionID, Title: c.Title, Feedback: c.Feedback})
	}
	sort.SliceStable(cues, func(i, j int) bool { return cues[i].Index < cues[j].Index })
	rv := &RevealedView{
		GPTOverallFeedback: item.GPTOverallFeedback,
		GPTCues:            cues,
		Rating: &RatingView{
			NeedsImprovement: rating.NeedsImprovement,
			Step1SubmittedAt: rating.Step1SubmittedAt,
			CueRatings:       rating.CueRatings,
			OverallScore:     rating.OverallScore,
			Comment:          rating.Comment,
			Step2SubmittedAt: rating.Step2SubmittedAt,
			Completed:        rating.Completed,
		},
	}
	if rv.Rating.NeedsImprovement == nil {
		rv.Rating.NeedsImprovement = map[string]bool{}
	}
	if rv.Rating.CueRatings == nil {
		rv.Rating.CueRatings = map[string]string{}
	}
	if p := item.Media["feedback_video"]; p != "" {
		rv.FeedbackVideoURL = sign(p)
	}
	v.RevealedView = rv
	return v
}

// ExpertProgress is one row of GET /api/admin/progress.
type ExpertProgress struct {
	ExpertID  string `json:"expert_id"`
	Total     int    `json:"total"`
	Step1Done int    `json:"step1_done"`
	Completed int    `json:"completed"`
}

// BuildProgress counts each expert's progress over servable items.
func BuildProgress(expertIDs []string, batchID string, items []Item, ratings []Rating) []ExpertProgress {
	eligible := map[string]bool{}
	for _, it := range items {
		if it.ServableTo(batchID) {
			eligible[it.ItemID] = true
		}
	}
	byExpert := map[string]*ExpertProgress{}
	out := make([]ExpertProgress, len(expertIDs))
	for i, id := range expertIDs {
		out[i] = ExpertProgress{ExpertID: id, Total: len(eligible)}
		byExpert[id] = &out[i]
	}
	for i := range ratings {
		r := &ratings[i]
		p, ok := byExpert[r.ExpertID]
		if !ok || !eligible[r.ItemID] {
			continue
		}
		if r.Step1Done() {
			p.Step1Done++
		}
		if r.IsCompleted() {
			p.Completed++
		}
	}
	return out
}
