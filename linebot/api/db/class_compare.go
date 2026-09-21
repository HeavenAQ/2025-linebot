package db

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"google.golang.org/api/iterator"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// ClassStanding places one learner among their classmates for a skill.
//
// Every classmate is counted by their own best attempt, so a learner who
// uploads often is not ranked above one who uploaded twice and did better.
type ClassStanding struct {
	Skill string `json:"skill"`
	// The learner's own best total, or the attempt being placed.
	Score float64 `json:"score"`
	// 1 is the best in the class.
	Rank int `json:"rank"`
	// Share of classmates this score is at least as good as, 0-100.
	Percentile float64 `json:"percentile"`
	// Learners with at least one graded attempt at this skill, including this one.
	Classmates int     `json:"classmates"`
	ClassMean  float64 `json:"class_mean"`
	ClassBest  float64 `json:"class_best"`
}

// ClassBestAttempt is the strongest attempt anyone in the class has recorded
// for a skill, broken down by rubric criterion.
//
// It carries no name, no user ID and no date: a learner may see what a good
// attempt scores on each criterion, never whose attempt it was.
type ClassBestAttempt struct {
	Skill      string                  `json:"skill"`
	TotalGrade float64                 `json:"total_grade"`
	Criteria   []commons.GradingDetail `json:"criteria"`
}

// bestAttempt is one learner's strongest graded attempt, used while scanning.
type bestAttempt struct {
	total    float64
	criteria []commons.GradingDetail
}

// classBestAttempts reduces every learner's portfolio to their best attempt at
// one skill. Learners with no graded attempt are left out entirely.
func classBestAttempts(users []UserData, skill string) []bestAttempt {
	var bests []bestAttempt
	for _, user := range users {
		var best *bestAttempt
		for _, work := range user.Portfolio.GetSkillPortfolio(skill) {
			if work.AnalysisStatus != "" && work.AnalysisStatus != "completed" {
				continue
			}
			if work.GradingOutcome.TotalGrade <= 0 {
				continue
			}
			if best == nil || work.GradingOutcome.TotalGrade > best.total {
				best = &bestAttempt{
					total:    work.GradingOutcome.TotalGrade,
					criteria: work.GradingOutcome.GradingDetails,
				}
			}
		}
		if best != nil {
			bests = append(bests, *best)
		}
	}
	return bests
}

// standing places score among bests. Ties share the better rank, and the
// percentile counts classmates this score is at least as good as, so the top
// score is the 100th percentile and the bottom one is not zero.
func standing(bests []bestAttempt, skill string, score float64) ClassStanding {
	result := ClassStanding{Skill: skill, Score: score, Classmates: len(bests)}
	if len(bests) == 0 {
		return result
	}
	better, atLeast, sum := 0, 0, 0.0
	for _, item := range bests {
		sum += item.total
		if item.total > score {
			better++
		}
		if score >= item.total {
			atLeast++
		}
		if item.total > result.ClassBest {
			result.ClassBest = item.total
		}
	}
	result.Rank = better + 1
	result.Percentile = float64(atLeast) / float64(len(bests)) * 100
	result.ClassMean = sum / float64(len(bests))
	return result
}

// listUsers reads every learner once. A class is small enough that this is one
// pass over a few dozen documents; the nightly class-stats rebuild scans the
// same way.
func (client *FirestoreClient) listUsers(ctx context.Context) ([]UserData, error) {
	iter := client.Data.Documents(ctx)
	defer iter.Stop()
	var users []UserData
	for {
		doc, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("scan users: %w", err)
		}
		var user UserData
		if err := doc.DataTo(&user); err != nil {
			continue // one malformed document must not hide the whole class
		}
		if !client.CountsTowardClass(user.ID) {
			continue
		}
		users = append(users, user)
	}
	return users, nil
}

// ClassStandingFor places the learner's own best attempt among the class.
func (client *FirestoreClient) ClassStandingFor(ctx context.Context, userID, skill string) (ClassStanding, error) {
	skill = strings.ToLower(strings.TrimSpace(skill))
	users, err := client.listUsers(ctx)
	if err != nil {
		return ClassStanding{}, err
	}
	bests := classBestAttempts(users, skill)
	own, err := client.GetUserData(userID)
	if err != nil {
		return ClassStanding{}, err
	}
	mine := classBestAttempts([]UserData{*own}, skill)
	if len(mine) == 0 {
		return ClassStanding{Skill: skill, Classmates: len(bests)}, nil
	}
	return standing(bests, skill, mine[0].total), nil
}

// ClassBestAttemptFor returns the class's strongest attempt at a skill, with
// nothing in it that identifies whose attempt it is.
func (client *FirestoreClient) ClassBestAttemptFor(ctx context.Context, skill string) (ClassBestAttempt, error) {
	skill = strings.ToLower(strings.TrimSpace(skill))
	users, err := client.listUsers(ctx)
	if err != nil {
		return ClassBestAttempt{}, err
	}
	bests := classBestAttempts(users, skill)
	if len(bests) == 0 {
		return ClassBestAttempt{Skill: skill}, nil
	}
	sort.Slice(bests, func(i, j int) bool { return bests[i].total > bests[j].total })
	top := bests[0]
	criteria := make([]commons.GradingDetail, len(top.criteria))
	copy(criteria, top.criteria)
	return ClassBestAttempt{Skill: skill, TotalGrade: top.total, Criteria: criteria}, nil
}
