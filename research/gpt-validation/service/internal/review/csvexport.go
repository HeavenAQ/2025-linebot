package review

import (
	"encoding/csv"
	"io"
	"sort"
	"strconv"
	"strings"
)

// utf8BOM makes Excel open the CSVs as UTF-8 (Chinese names and comments).
const utf8BOM = "\xef\xbb\xbf"

// CSV headers (DESIGN.md "CSV columns").
var (
	ItemsCSVHeader    = []string{"item_id", "display_code", "skill", "source_file", "workbook_id", "status", "eligible", "coaching_source", "coaching_model", "handedness", "total_grade", "n_cues", "gpt_flagged_criteria"}
	CriteriaCSVHeader = []string{"item_id", "skill", "expert_id", "criterion_id", "criterion_name", "expert_needs_improvement", "gpt_flagged"}
	CuesCSVHeader     = []string{"item_id", "skill", "expert_id", "cue_index", "criterion_id", "cue_rating", "cue_score"}
	OverallCSVHeader  = []string{"item_id", "skill", "expert_id", "overall_score", "comment", "step1_seconds", "step2_seconds", "completed"}
)

// ExportData is everything the exports need, loaded once per request.
type ExportData struct {
	BatchID   string
	ExpertIDs []string
	Items     []Item
	Ratings   []Rating
}

func (d ExportData) sortedItems() []Item {
	items := append([]Item(nil), d.Items...)
	sort.Slice(items, func(i, j int) bool { return items[i].ItemID < items[j].ItemID })
	return items
}

func (d ExportData) sortedExperts() []string {
	seen := map[string]bool{}
	var ids []string
	for _, id := range d.ExpertIDs {
		if !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	// Ratings from experts whose code was since removed are still data.
	for _, r := range d.Ratings {
		if r.ExpertID != "" && !seen[r.ExpertID] {
			seen[r.ExpertID] = true
			ids = append(ids, r.ExpertID)
		}
	}
	sort.Strings(ids)
	return ids
}

func (d ExportData) ratingIndex() map[string]*Rating {
	idx := make(map[string]*Rating, len(d.Ratings))
	for i := range d.Ratings {
		r := &d.Ratings[i]
		idx[RatingDocID(r.ItemID, r.ExpertID)] = r
	}
	return idx
}

func bit(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func formatFloat(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }

// excelSafe stops spreadsheet apps from evaluating free text as a formula.
func excelSafe(s string) string {
	if s != "" && strings.ContainsRune("=+@\t\r", rune(s[0])) {
		return "'" + s
	}
	return s
}

func writeCSV(w io.Writer, header []string, rows [][]string) error {
	if _, err := io.WriteString(w, utf8BOM); err != nil {
		return err
	}
	cw := csv.NewWriter(w)
	if err := cw.Write(header); err != nil {
		return err
	}
	if err := cw.WriteAll(rows); err != nil {
		return err
	}
	return cw.Error()
}

// WriteItemsCSV writes one row per item in the batch, eligible or not.
func WriteItemsCSV(w io.Writer, d ExportData) error {
	var rows [][]string
	for _, it := range d.sortedItems() {
		grade := ""
		if it.TotalGrade != nil {
			grade = formatFloat(*it.TotalGrade)
		}
		rows = append(rows, []string{
			it.ItemID, it.DisplayCode, it.Skill, it.SourceFile, it.WorkbookID, it.Status,
			bit(it.Eligible), it.CoachingSource, it.CoachingModel, it.Handedness,
			grade, strconv.Itoa(len(it.GPTCues)), strings.Join(it.GPTFlaggedCriteria, ";"),
		})
	}
	return writeCSV(w, ItemsCSVHeader, rows)
}

// WriteCriteriaCSV writes one row per item × expert × criterion for ratings
// whose step 1 was submitted.
func WriteCriteriaCSV(w io.Writer, d ExportData) error {
	idx := d.ratingIndex()
	var rows [][]string
	for _, it := range d.sortedItems() {
		flagged := it.FlaggedSet()
		for _, expert := range d.sortedExperts() {
			r := idx[RatingDocID(it.ItemID, expert)]
			if !r.Step1Done() {
				continue
			}
			for _, c := range it.Criteria {
				rows = append(rows, []string{
					it.ItemID, it.Skill, expert, c.ID, c.NameZh,
					bit(r.NeedsImprovement[c.ID]), bit(flagged[c.ID]),
				})
			}
		}
	}
	return writeCSV(w, CriteriaCSVHeader, rows)
}

// WriteCuesCSV writes one row per item × expert × rated cue.
func WriteCuesCSV(w io.Writer, d ExportData) error {
	idx := d.ratingIndex()
	var rows [][]string
	for _, it := range d.sortedItems() {
		cues := append([]Cue(nil), it.GPTCues...)
		sort.SliceStable(cues, func(i, j int) bool { return cues[i].Index < cues[j].Index })
		for _, expert := range d.sortedExperts() {
			r := idx[RatingDocID(it.ItemID, expert)]
			if !r.Step1Done() {
				continue
			}
			for _, c := range cues {
				key := strconv.Itoa(c.Index)
				rating, ok := r.CueRatings[key]
				score, valid := CueScore(rating)
				if !ok || !valid {
					continue
				}
				rows = append(rows, []string{
					it.ItemID, it.Skill, expert, key, c.CriterionID, rating, strconv.Itoa(score),
				})
			}
		}
	}
	return writeCSV(w, CuesCSVHeader, rows)
}

// WriteOverallCSV writes one row per eligible item × expert. Pairs the expert
// has not finished have blank score fields and completed=0.
func WriteOverallCSV(w io.Writer, d ExportData) error {
	idx := d.ratingIndex()
	var rows [][]string
	for _, it := range d.sortedItems() {
		for _, expert := range d.sortedExperts() {
			r := idx[RatingDocID(it.ItemID, expert)]
			if r == nil && !it.ServableTo(d.BatchID) {
				continue
			}
			score, comment, s1, s2 := "", "", "", ""
			completed := false
			if r != nil {
				if r.OverallScore != nil {
					score = strconv.Itoa(*r.OverallScore)
				}
				comment = excelSafe(r.Comment)
				if r.Step1Done() {
					s1 = formatFloat(r.Step1Seconds)
				}
				if r.Step2Seconds != nil {
					s2 = formatFloat(*r.Step2Seconds)
				}
				completed = r.Completed
			}
			rows = append(rows, []string{it.ItemID, it.Skill, expert, score, comment, s1, s2, bit(completed)})
		}
	}
	return writeCSV(w, OverallCSVHeader, rows)
}
