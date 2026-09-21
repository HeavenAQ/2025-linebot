package line

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

// analysisResult is what a learner sees when their upload has been graded: a
// line of text and the portfolio card carrying the new entry.
func (client *Client) analysisResult(user *db.UserData, skill db.BadmintonSkill, text string, showBtns bool) []linebot.SendingMessage {
	works := user.Portfolio.GetSkillPortfolio(skill.String())
	if len(works) == 0 {
		return []linebot.SendingMessage{linebot.NewTextMessage(text)}
	}
	return []linebot.SendingMessage{
		linebot.NewTextMessage(text),
		client.portfolioCarousel(works, skill.String(), showBtns),
	}
}

// ReplyAnalysisResult answers the upload itself, so the learner reads the
// result in the same breath as sending the video. The token dies about a
// minute after the upload, which almost every analysis beats.
func (client *Client) ReplyAnalysisResult(replyToken string, user *db.UserData, skill db.BadmintonSkill, text string, showBtns bool) error {
	_, err := client.bot.ReplyMessage(replyToken, client.analysisResult(user, skill, text, showBtns)...).Do()
	return err
}

// PushAnalysisResult is for the analysis that outlived its reply token. It
// costs a push, which is why it is the fallback and not the rule.
func (client *Client) PushAnalysisResult(userID string, user *db.UserData, skill db.BadmintonSkill, text string, showBtns bool) error {
	_, err := client.bot.PushMessage(userID, client.analysisResult(user, skill, text, showBtns)...).Do()
	return err
}
