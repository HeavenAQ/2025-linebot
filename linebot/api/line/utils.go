package line

import (
	"encoding/json"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type ReplyAction func(db.UserState, db.BadmintonSkill) linebot.QuickReplyAction

func (client *Client) getQuickReplyAction() ReplyAction {
	return func(userState db.UserState, skill db.BadmintonSkill) linebot.QuickReplyAction {
		postbackData, err := json.Marshal(SelectingSkillPostback{
			State: userState.String(),
			Skill: skill.String(),
		})
		if err != nil {
			return nil
		}

		return linebot.NewPostbackAction(
			skill.ChnString(),
			string(postbackData),
			"",
			skill.ChnString(),
			linebot.InputOption(""),
			"",
		)
	}
}
