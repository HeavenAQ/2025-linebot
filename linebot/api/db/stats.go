package db

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// Stats represents aggregate statistics for grades
type DateStats = map[string]Stats

type Stats struct {
	Avg float64 `json:"avg"`
	Max float64 `json:"max"`
	Min float64 `json:"min"`
	Std float64 `json:"std"`
}

func computeDateStats(dateValues map[string][]float64) (DateStats, error) {
	dateStats := DateStats{}
	for date, values := range dateValues {
		stats, err := computeStats(values)
		if err != nil {
			return DateStats{}, fmt.Errorf("Stats on %v cannot be calculated", date)
		}
		dateStats[date] = stats
	}
	return dateStats, nil
}

func computeStats(values []float64) (Stats, error) {
	n := float64(len(values))
	if n == 0 {
		return Stats{}, errors.New("no values to compute stats")
	}
	min := values[0]
	max := values[0]
	sum := 0.0
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}
	avg := sum / n
	// population standard deviation by default
	var varSum float64
	for _, v := range values {
		d := v - avg
		varSum += d * d
	}
	std := math.Sqrt(varSum / n)
	return Stats{Avg: avg, Max: max, Min: min, Std: std}, nil
}

// GetUserSkillStats returns stats for a single user's grades for a given skill
func (client *FirestoreClient) GetUserSkillStats(userID string, skill string) (DateStats, error) {
	user, err := client.GetUserData(userID)
	if err != nil {
		return DateStats{}, err
	}
	portfolio := user.Portfolio.GetSkillPortfolio(skill)
	if portfolio == nil {
		return DateStats{}, errors.New("invalid or empty skill portfolio")
	}

	gradesOnDate := make(map[string][]float64)
	for date, work := range portfolio {
		// parse date to use YYYY-MM-DD only
		parsedDate, err := time.Parse(
			"2006-01-02-15-04", date,
		)
		if err != nil {
			return DateStats{}, errors.New("failed to parse the work's key as a date string")
		}

		date = parsedDate.Format("2006-01-02")
		gradesOnDate[date] = append(
			gradesOnDate[date],
			work.GradingOutcome.TotalGrade,
		)
	}
	return computeDateStats(gradesOnDate)
}

// GetClassSkillStats aggregates across all users for a given skill
func (client *FirestoreClient) GetClassSkillStats(skill string) (DateStats, error) {
	iter := client.Data.Documents(*client.Ctx)
	gradesOnDate := make(map[string][]float64)

	for {
		doc, err := iter.Next()
		if err != nil {
			break
		}
		var user UserData
		if err := doc.DataTo(&user); err != nil {
			continue
		}
		port := user.Portfolio.GetSkillPortfolio(skill)
		if port == nil {
			continue
		}
		for date, work := range port {
			// parse date to use YYYY-MM-DD only
			parsedDate, err := time.Parse(
				"2006-01-02-15-04", date,
			)
			if err != nil {
				return DateStats{}, errors.New("failed to parse the work's key as a date string")
			}

			date = parsedDate.Format("2006-01-02")
			gradesOnDate[date] = append(
				gradesOnDate[date],
				work.GradingOutcome.TotalGrade,
			)
		}
	}
	return computeDateStats(gradesOnDate)
}
