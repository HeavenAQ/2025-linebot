package line

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type ReplyAction func(db.UserState, db.BadmintonSkill) linebot.QuickReplyAction

func (client *LineBot) getQuickReplyAction() ReplyAction {
	return func(userState db.UserState, skill db.BadmintonSkill) linebot.QuickReplyAction {
		postbackData := `{"userState": "` + userState.String() + `", "skill": "` + skill.String() + `"}`
		return linebot.NewPostbackAction(
			skill.ChnString(),
			postbackData,
			"",
			skill.ChnString(),
			linebot.InputOption(""),
			"",
		)
	}
}

func (client *LineBot) getHandednessQuickReplyItems() *linebot.QuickReplyItems {
	items := []*linebot.QuickReplyButton{}
	for _, handedness := range []db.Handedness{db.Left, db.Right} {
		items = append(items, linebot.NewQuickReplyButton(
			"",
			linebot.NewPostbackAction(
				handedness.ChnString(),
				"handedness="+handedness.String(),
				"",
				handedness.ChnString(),
				"",
				"",
			),
		))
	}
	return linebot.NewQuickReplyItems(items...)
}
