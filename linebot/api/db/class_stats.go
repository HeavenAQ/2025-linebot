package db

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"cloud.google.com/go/firestore"
	"google.golang.org/api/iterator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ClassStat is the running aggregate of every completed attempt of one skill
// on one day. The class chart reads these instead of scanning every learner's
// portfolio on each request.
//
// Count, sum and sum of squares are enough to recover the mean and standard
// deviation exactly; min and max only ever widen because attempts are never
// regraded. The nightly rebuild corrects anything that drifts (for example a
// deleted test user).
type ClassStat struct {
	Skill      string    `firestore:"skill"`
	Date       string    `firestore:"date"`
	Count      int64     `firestore:"count"`
	Sum        float64   `firestore:"sum"`
	SumSquares float64   `firestore:"sum_squares"`
	Min        float64   `firestore:"min"`
	Max        float64   `firestore:"max"`
	UpdatedAt  time.Time `firestore:"updated_at"`
}

// ClassStats is keyed by the deployment's data collection, preserving the
// isolation between the LLM and no-LLM variants.
func (c *FirestoreClient) ClassStats() *firestore.CollectionRef {
	return c.Client.Collection(c.Data.ID + "_class_stats")
}

func classStatID(skill, date string) string { return skill + "_" + date }

// workDay maps a portfolio key to the calendar day the class chart groups by.
func workDay(workKey string) (string, error) {
	parsed, err := ParseWorkTime(workKey)
	if err != nil {
		return "", fmt.Errorf("work key %q is not a date: %w", workKey, err)
	}
	return parsed.Format("2006-01-02"), nil
}

// Add folds one grade into the aggregate.
func (s *ClassStat) Add(grade float64) {
	if s.Count == 0 || grade < s.Min {
		s.Min = grade
	}
	if s.Count == 0 || grade > s.Max {
		s.Max = grade
	}
	s.Count++
	s.Sum += grade
	s.SumSquares += grade * grade
}

// Stats converts the aggregate into what the API returns (population standard
// deviation, as before).
func (s ClassStat) Stats() Stats {
	if s.Count == 0 {
		return Stats{}
	}
	n := float64(s.Count)
	mean := s.Sum / n
	variance := s.SumSquares/n - mean*mean
	return Stats{Avg: mean, Max: s.Max, Min: s.Min, Std: math.Sqrt(math.Max(0, variance))}
}

// readClassStat reads an aggregate inside a transaction; a missing one is an
// empty aggregate. Firestore requires every transactional read to precede the
// writes.
func (c *FirestoreClient) readClassStat(tx *firestore.Transaction, skill, date string) (*firestore.DocumentRef, ClassStat, error) {
	ref := c.ClassStats().Doc(classStatID(skill, date))
	stat := ClassStat{Skill: skill, Date: date}
	doc, err := tx.Get(ref)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return ref, stat, nil
		}
		return nil, ClassStat{}, err
	}
	if err := doc.DataTo(&stat); err != nil {
		return nil, ClassStat{}, err
	}
	return ref, stat, nil
}

// RebuildClassStats recomputes every aggregate from the portfolios and removes
// aggregates no attempt supports any more. It scans every learner, so it runs
// on a schedule outside class rather than per request.
func (c *FirestoreClient) RebuildClassStats(ctx context.Context) (int, error) {
	aggregates := map[string]*ClassStat{}
	users := c.Data.Documents(ctx)
	defer users.Stop()
	for {
		doc, err := users.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return 0, fmt.Errorf("scan users: %w", err)
		}
		var user UserData
		if err := doc.DataTo(&user); err != nil {
			continue // one malformed document must not block the rest
		}
		for _, skill := range SkillOrder {
			for key, work := range user.Portfolio.GetSkillPortfolio(skill) {
				if work.AnalysisStatus != "" && work.AnalysisStatus != "completed" {
					continue
				}
				day, err := workDay(key)
				if err != nil {
					continue
				}
				id := classStatID(skill, day)
				if aggregates[id] == nil {
					aggregates[id] = &ClassStat{Skill: skill, Date: day}
				}
				aggregates[id].Add(work.GradingOutcome.TotalGrade)
			}
		}
	}

	writer := c.Client.BulkWriter(ctx)
	now := time.Now()
	for id, stat := range aggregates {
		stat.UpdatedAt = now
		if _, err := writer.Set(c.ClassStats().Doc(id), *stat); err != nil {
			writer.End()
			return 0, err
		}
	}
	existing := c.ClassStats().Documents(ctx)
	defer existing.Stop()
	for {
		doc, err := existing.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			writer.End()
			return 0, fmt.Errorf("list class stats: %w", err)
		}
		if _, ok := aggregates[doc.Ref.ID]; !ok {
			if _, err := writer.Delete(doc.Ref); err != nil {
				writer.End()
				return 0, err
			}
		}
	}
	writer.End()
	return len(aggregates), nil
}

// GetClassSkillStats returns the class's per-day statistics for one skill.
func (c *FirestoreClient) GetClassSkillStats(skill string) (DateStats, error) {
	skill = strings.ToLower(strings.TrimSpace(skill))
	docs := c.ClassStats().Where("skill", "==", skill).Documents(*c.Ctx)
	defer docs.Stop()
	result := DateStats{}
	for {
		doc, err := docs.Next()
		if err == iterator.Done {
			return result, nil
		}
		if err != nil {
			return nil, fmt.Errorf("read class stats: %w", err)
		}
		var stat ClassStat
		if err := doc.DataTo(&stat); err != nil {
			return nil, err
		}
		if stat.Count > 0 {
			result[stat.Date] = stat.Stats()
		}
	}
}
