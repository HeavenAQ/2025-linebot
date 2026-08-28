package line

import (
	"net/url"
	"strings"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	linebotsdk "github.com/line/line-bot-sdk-go/v7/linebot"
	"github.com/stretchr/testify/require"
)

// A record carrying a long signed URL: nothing on the card may grow with it,
// which only holds while the buttons refer to the work rather than embed a URL.
func portfolioWork() db.Work {
	return db.Work{
		DateTime:  "2026-08-01-20-30",
		Thumbnail: "https://storage.example/thumbnail.jpeg?" + string(make([]byte, 500)),
	}
}

func TestPortfolioVideoPostbackStaysWithinLineLimit(t *testing.T) {
	client := &Client{}
	work := portfolioWork()

	buttons, err := client.createButtonActions(work, "serve")
	require.NoError(t, err)
	require.Len(t, buttons, 2)
	button, ok := buttons[1].(*linebotsdk.ButtonComponent)
	require.True(t, ok)
	action, ok := button.Action.(*linebotsdk.PostbackAction)
	require.True(t, ok)
	require.LessOrEqual(t, len(action.Data), 300)
	require.NotContains(t, action.Data, "https://")

	postback, err := client.HandleVideoPostbackData(action.Data)
	require.NoError(t, err)
	require.Equal(t, work.DateTime, postback.WorkDate)
	require.Equal(t, "serve", postback.Skill)
}

// The reflection button opens the web app at this one attempt, so the URI has
// to name the work -- and only the work: a signed playback URL belongs nowhere
// near a link that sits in the chat history for the rest of the semester.
func TestReflectionButtonLinksToTheWorkInTheReviewTab(t *testing.T) {
	client := &Client{reviewURL: "https://liff.example/personal?tab=review"}
	work := portfolioWork()

	buttons, err := client.createButtonActions(work, "serve")
	require.NoError(t, err)
	button, ok := buttons[0].(*linebotsdk.ButtonComponent)
	require.True(t, ok)
	action, ok := button.Action.(*linebotsdk.URIAction)
	require.True(t, ok)
	require.Equal(t, "更新學習反思", action.Label)

	parsed, err := url.Parse(action.URI)
	require.NoError(t, err)
	require.Equal(t, "https://liff.example/personal", parsed.Scheme+"://"+parsed.Host+parsed.Path)
	require.Equal(t, "review", parsed.Query().Get("tab"))
	require.Equal(t, "serve", parsed.Query().Get("skill"))
	require.Equal(t, work.DateTime, parsed.Query().Get("date"))
	require.NotContains(t, action.URI, work.Thumbnail)
}

// The configured URL already carries its own query, and a stray second "?"
// would hide tab= inside the value of date=.
func TestWorkReviewURLMergesIntoTheConfiguredQuery(t *testing.T) {
	merged := workReviewURL(
		"https://liff.example/personal?tab=review", "smash", "2026-08-01-20-30",
	)

	require.Equal(t, 1, strings.Count(merged, "?"))
	require.Equal(
		t,
		"https://liff.example/personal?date=2026-08-01-20-30&skill=smash&tab=review",
		merged,
	)
}

// Neither weekly menu entry has configuration of its own: each link is the
// review URL with its sub-tab merged into the query it already carries.
func TestReviewSectionURLMergesIntoTheConfiguredQuery(t *testing.T) {
	const configured = "https://liff.example/personal?tab=review"

	preview := reviewSectionURL(configured, previewSection)
	require.Equal(t, 1, strings.Count(preview, "?"))
	require.Equal(t, "https://liff.example/personal?section=preview&tab=review", preview)

	reflection := reviewSectionURL(configured, reflectionSection)
	require.Equal(t, 1, strings.Count(reflection, "?"))
	require.Equal(t, "https://liff.example/personal?section=reflection&tab=review", reflection)
}
