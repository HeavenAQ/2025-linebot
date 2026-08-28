package line

import (
	"fmt"
	"strings"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

// PushMessage sends to a user outside of any reply window. Unlike a reply it
// has no token to answer, so it is the only way to start a conversation the
// learner did not trigger.
func (client *Client) PushMessage(
	userID string,
	messages ...linebot.SendingMessage,
) (*linebot.BasicResponse, error) {
	res, err := client.bot.PushMessage(userID, messages...).Do()
	if err != nil {
		return nil, fmt.Errorf("failed to push message: %w", err)
	}
	return res, nil
}

// weeklyPreviewBubble renders the preview as its own card so it stands apart
// from the chat around it. The note arrives as written text rather than being
// split into components, because the coach's wording is the content.
func weeklyPreviewBubble(skill db.BadmintonSkill, note string) *linebot.BubbleContainer {
	return &linebot.BubbleContainer{
		Type: linebot.FlexContainerTypeBubble,
		Header: &linebot.BoxComponent{
			Type:            "box",
			Layout:          "vertical",
			BackgroundColor: "#1F6F4A",
			PaddingAll:      "16px",
			Contents: []linebot.FlexComponent{
				&linebot.TextComponent{
					Type:   "text",
					Text:   "本週課前預習",
					Color:  "#FFFFFF",
					Size:   "lg",
					Weight: linebot.FlexTextWeightTypeBold,
				},
				&linebot.TextComponent{
					Type:   "text",
					Text:   "重點加強：" + skill.ChnString(),
					Color:  "#DCEFE4",
					Size:   "sm",
					Margin: "sm",
				},
			},
		},
		Body: &linebot.BoxComponent{
			Type:       "box",
			Layout:     "vertical",
			PaddingAll: "16px",
			Contents: []linebot.FlexComponent{
				&linebot.TextComponent{
					Type:  "text",
					Text:  note,
					Size:  "sm",
					Wrap:  true,
					Color: "#333333",
				},
			},
		},
	}
}

// ReplyWeeklyPreview answers the 產生課前預習 button with the same card the
// scheduled run pushes, so a note the learner asked for and one they were sent
// read identically.
func (client *Client) ReplyWeeklyPreview(
	replyToken string,
	skill db.BadmintonSkill,
	note string,
) error {
	trimmed := strings.TrimSpace(note)
	if trimmed == "" {
		return fmt.Errorf("weekly preview note is empty")
	}
	_, err := client.ReplyMessage(
		replyToken,
		linebot.NewFlexMessage(
			"本週課前預習："+skill.ChnString(),
			weeklyPreviewBubble(skill, trimmed),
		),
	)
	return err
}

// PushWeeklyPreview delivers a learner's 課前預習 note.
func (client *Client) PushWeeklyPreview(
	userID string,
	skill db.BadmintonSkill,
	note string,
) error {
	trimmed := strings.TrimSpace(note)
	if trimmed == "" {
		return fmt.Errorf("weekly preview note is empty")
	}
	_, err := client.PushMessage(
		userID,
		linebot.NewFlexMessage(
			"本週課前預習："+skill.ChnString(),
			weeklyPreviewBubble(skill, trimmed),
		),
	)
	return err
}
