package review

import "fmt"

var fakeCriteria = map[string][]Criterion{
	"serve": {
		{ID: "serve_grip", NameZh: "握拍方式"},
		{ID: "serve_stance", NameZh: "站位與重心"},
		{ID: "serve_backswing", NameZh: "引拍動作"},
		{ID: "serve_contact", NameZh: "擊球點"},
		{ID: "serve_wrist", NameZh: "手腕發力"},
		{ID: "serve_follow", NameZh: "隨揮動作"},
	},
	"smash": {
		{ID: "smash_prep", NameZh: "準備姿勢"},
		{ID: "smash_turn", NameZh: "側身轉體"},
		{ID: "smash_elbow", NameZh: "手肘抬高"},
		{ID: "smash_contact", NameZh: "擊球點高度"},
		{ID: "smash_pronation", NameZh: "前臂內旋"},
		{ID: "smash_recovery", NameZh: "落地回位"},
	},
}

// FakeItems returns sample items for local UI development (LOCAL_FAKE_DATA=1).
func FakeItems(batchID string) []Item {
	var items []Item
	for i := 1; i <= 6; i++ {
		skill, prefix := "serve", "SV"
		if i%2 == 0 {
			skill, prefix = "smash", "SM"
		}
		crit := fakeCriteria[skill]
		id := fmt.Sprintf("%s-fake%05d", skill, i)
		it := Item{
			ItemID:             id,
			BatchID:            batchID,
			Skill:              skill,
			SourceFile:         fmt.Sprintf("student_%02d.mp4", i),
			WorkbookID:         fmt.Sprintf("EG%02d", i),
			DisplayCode:        fmt.Sprintf("%s-%03d", prefix, i),
			Status:             "ready",
			Eligible:           true,
			CoachingSource:     "openai",
			CoachingModel:      "fake-model",
			Handedness:         "right",
			Criteria:           crit,
			GPTOverallFeedback: "整體動作流暢，節奏掌握不錯。\n建議加強擊球前的準備，讓重心更穩定。",
			Media: map[string]string{
				"detected_overlay": "gpt-validation/" + batchID + "/renders/" + id + "/detected_overlay.mp4",
				"feedback_video":   "gpt-validation/" + batchID + "/renders/" + id + "/feedback.mp4",
			},
		}
		if i != 3 { // item 3 has no cues
			it.GPTCues = []Cue{
				{Index: 1, CriterionID: crit[1].ID, Title: crit[1].NameZh, Feedback: "重心偏後，擊球時身體向後倒。試著把重心放在前腳。"},
				{Index: 2, CriterionID: crit[3].ID, Title: crit[3].NameZh, Feedback: "擊球點太靠近身體，手臂沒有完全伸展。"},
			}
			it.GPTFlaggedCriteria = []string{crit[1].ID, crit[3].ID}
		}
		items = append(items, it)
	}
	// An ineligible item that must never be shown to experts.
	items = append(items, Item{
		ItemID: "serve-fakefallback", BatchID: batchID, Skill: "serve", DisplayCode: "SV-999",
		Status: "ready", Eligible: false, CoachingSource: "deterministic_fallback", Criteria: fakeCriteria["serve"],
	})
	return items
}
