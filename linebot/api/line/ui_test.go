package line

import (
	"encoding/json"
	"fmt"
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

// The review card carries the link and a request for the coach's 課前預習 note.
// Postback payloads are told apart by their exact field set alone, so the
// second button has to come back as a preview request and as nothing else.
func TestWeeklyReviewCardOffersTheLinkAndAPreviewRequest(t *testing.T) {
	client := &Client{}
	bubble, err := weeklyReviewBubble("https://liff.example/personal?tab=review")
	require.NoError(t, err)
	require.Len(t, bubble.Footer.Contents, 2)

	link, ok := bubble.Footer.Contents[0].(*linebotsdk.ButtonComponent)
	require.True(t, ok)
	linkAction, ok := link.Action.(*linebotsdk.URIAction)
	require.True(t, ok)
	require.Equal(t, "前往每週回顧", linkAction.Label)
	require.Equal(t, "https://liff.example/personal?tab=review", linkAction.URI)

	button, ok := bubble.Footer.Contents[1].(*linebotsdk.ButtonComponent)
	require.True(t, ok)
	action, ok := button.Action.(*linebotsdk.PostbackAction)
	require.True(t, ok)
	require.Equal(t, "產生課前預習", action.Label)

	preview, err := client.HandleWeeklyPreviewPostbackData(action.Data)
	require.NoError(t, err)
	require.True(t, preview.Preview)

	// Every other postback the router tries before this one must refuse it.
	_, err = client.HandleStopGPTPostbackData(action.Data)
	require.Error(t, err)
	_, err = client.HandleWritingNotePostbackData(action.Data)
	require.Error(t, err)
	_, err = client.HandleVideoPostbackData(action.Data)
	require.Error(t, err)

	// And it must not claim theirs: 結束對話 is a single-field payload too.
	stop, err := json.Marshal(StopGPTPostback{Stop: true})
	require.NoError(t, err)
	_, err = client.HandleWeeklyPreviewPostbackData(string(stop))
	require.Error(t, err)
}

func TestPortfolioCardIncludesLiffReflectionAndPreview(t *testing.T) {
	client := &Client{}
	work := portfolioWork()
	work.Reflection = "This week's reflection"
	work.Preview = "Next lesson's preview"

	item := client.getCarouselItem(work, "serve", false)
	require.NotNil(t, item)
	require.Len(t, item.Body.Contents, 5)

	preview, ok := item.Body.Contents[3].(*linebotsdk.BoxComponent)
	require.True(t, ok)
	reflection, ok := item.Body.Contents[4].(*linebotsdk.BoxComponent)
	require.True(t, ok)
	require.Equal(t, "課前檢視要點：", preview.Contents[0].(*linebotsdk.TextComponent).Text)
	require.Equal(t, work.Preview, preview.Contents[1].(*linebotsdk.TextComponent).Text)
	require.Equal(t, "學習反思：", reflection.Contents[0].(*linebotsdk.TextComponent).Text)
	require.Equal(t, work.Reflection, reflection.Contents[1].(*linebotsdk.TextComponent).Text)
}

func TestPortfolioReturnsOnlyTheLatestTenWorks(t *testing.T) {
	client := &Client{}
	works := make(map[string]db.Work)
	for day := 1; day <= 12; day++ {
		date := fmt.Sprintf("2026-08-%02d-12-00", day)
		works[date] = db.Work{DateTime: date}
	}

	latest := client.latestPortfolioWorks(works)
	require.Len(t, latest, latestPortfolioWorkLimit)
	require.Equal(t, "2026-08-12-12-00", latest[0].DateTime)
	require.Equal(t, "2026-08-03-12-00", latest[len(latest)-1].DateTime)
}
