package app

import (
	"errors"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
)

func TestBadHistoricalThumbnailDoesNotBlockCurrentPortfolio(t *testing.T) {
	user := &db.UserData{}
	user.Portfolio.Smash = map[string]db.Work{
		"old":    {DateTime: "old", Thumbnail: "missing"},
		"new":    {DateTime: "new", Thumbnail: "valid"},
		"hidden": {DateTime: "hidden", Thumbnail: "not-displayed"},
	}
	display := db.WithWeeklyReflectionNotes(user, nil)
	portfolio := display.Portfolio.Smash
	var attempted []string
	failures := signPortfolioThumbnails(portfolio, []db.Work{portfolio["old"], portfolio["new"]},
		func(value string) (commons.MediaRef, error) {
			attempted = append(attempted, value)
			if value == "missing" {
				return commons.MediaRef{}, errors.New("object not found")
			}
			return commons.MediaRef{SignedURL: "https://signed.example/image.jpeg"}, nil
		})
	require.Len(t, failures, 1)
	require.Equal(t, []string{"missing", "valid"}, attempted)
	require.Empty(t, portfolio["old"].Thumbnail)
	require.Equal(t, "https://signed.example/image.jpeg", portfolio["new"].Thumbnail)
	require.Equal(t, "not-displayed", portfolio["hidden"].Thumbnail)
	require.Equal(t, "missing", user.Portfolio.Smash["old"].Thumbnail)
	require.Equal(t, "valid", user.Portfolio.Smash["new"].Thumbnail)
}
