package line

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	linebotsdk "github.com/line/line-bot-sdk-go/v7/linebot"
	"github.com/stretchr/testify/require"
)

func TestPortfolioVideoPostbackStaysWithinLineLimit(t *testing.T) {
	client := &Client{}
	// A record carrying a long signed URL: the postback must still stay well
	// inside LINE's limit, which it only does by referring to the work rather
	// than embedding any URL.
	work := db.Work{
		DateTime:  "2026-08-01-20-30",
		Thumbnail: "https://storage.example/thumbnail.jpeg?" + string(make([]byte, 500)),
	}

	buttons, err := client.createButtonActions(work, "serve", "right")
	require.NoError(t, err)
	require.Len(t, buttons, 4)
	button, ok := buttons[3].(*linebotsdk.ButtonComponent)
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
