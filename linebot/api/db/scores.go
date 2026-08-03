package db

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// workKeyLayout is the format portfolio entries are keyed by.
const workKeyLayout = "2006-01-02-15-04"

// GetRecentSkillScores returns the learner's graded attempts for a skill,
// newest first. A limit of zero or less returns every attempt. An empty or
// missing portfolio is not an error: the learner simply has no scores yet.
func (client *FirestoreClient) GetRecentSkillScores(userID, skill string, limit int) ([]commons.SkillScore, error) {
	user, err := client.GetUserData(userID)
	if err != nil {
		return nil, err
	}
	return recentSkillScores(user.Portfolio.GetSkillPortfolio(strings.ToLower(strings.TrimSpace(skill))), limit), nil
}

func recentSkillScores(portfolio map[string]Work, limit int) []commons.SkillScore {
	scores := make([]commons.SkillScore, 0, len(portfolio))
	for key, work := range portfolio {
		scores = append(scores, commons.SkillScore{
			Date:        key,
			TotalGrade:  work.GradingOutcome.TotalGrade,
			ScoreStatus: work.GradingOutcome.ScoreStatus,
			Details:     work.GradingOutcome.GradingDetails,
		})
	}

	// Newest first. Keys that predate the current layout sort last rather than
	// failing the whole lookup, since a summary is still worth producing.
	sort.Slice(scores, func(i, j int) bool {
		left, leftErr := time.Parse(workKeyLayout, scores[i].Date)
		right, rightErr := time.Parse(workKeyLayout, scores[j].Date)
		switch {
		case leftErr == nil && rightErr == nil:
			return left.After(right)
		case leftErr == nil:
			return true
		case rightErr == nil:
			return false
		default:
			return scores[i].Date > scores[j].Date
		}
	})

	if limit > 0 && len(scores) > limit {
		scores = scores[:limit]
	}
	return scores
}

// ScoreFingerprint identifies a set of scores well enough to tell whether a
// cached summary was written before or after the learner's latest attempt.
func ScoreFingerprint(scores []commons.SkillScore) string {
	if len(scores) == 0 {
		return ""
	}
	parts := make([]string, 0, len(scores))
	for _, score := range scores {
		parts = append(parts, fmt.Sprintf("%s:%.2f", score.Date, score.TotalGrade))
	}
	return strings.Join(parts, "|")
}
