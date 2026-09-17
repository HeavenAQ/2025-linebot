package db

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

// The aggregate must reproduce exactly what the old full scan computed.
func TestClassStatMatchesDirectComputation(t *testing.T) {
	grades := []float64{45.5, 88, 61.25, 99.9, 0}
	var stat ClassStat
	for _, g := range grades {
		stat.Add(g)
	}
	want, err := computeStats(grades)
	require.NoError(t, err)
	got := stat.Stats()
	require.InDelta(t, want.Avg, got.Avg, 1e-9)
	require.InDelta(t, want.Std, got.Std, 1e-9)
	require.Equal(t, want.Min, got.Min)
	require.Equal(t, want.Max, got.Max)
	require.EqualValues(t, 5, stat.Count)
}

func TestClassStatSingleAndEmpty(t *testing.T) {
	require.Equal(t, Stats{}, ClassStat{}.Stats())
	var stat ClassStat
	stat.Add(70)
	got := stat.Stats()
	require.Equal(t, Stats{Avg: 70, Max: 70, Min: 70, Std: 0}, got)
	require.False(t, math.IsNaN(got.Std))
}

func TestWorkDayAcceptsBothKeyLayouts(t *testing.T) {
	day, err := workDay("2026-09-14-14-05-31-8f3a2c1d")
	require.NoError(t, err)
	require.Equal(t, "2026-09-14", day)
	day, err = workDay("2026-09-14-14-05")
	require.NoError(t, err)
	require.Equal(t, "2026-09-14", day)
	_, err = workDay("legacy")
	require.Error(t, err)
}
