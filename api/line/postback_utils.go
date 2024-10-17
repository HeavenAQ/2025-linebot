package line

import (
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type WritingNotePostback struct {
	State      string `json:"state"`
	WorkDate   string `json:"work_date"`
	ActionStep string `json:"action_step"`
}

func (client *Client) PromptUploadVideo(event *linebot.Event) error {
	_, err := client.bot.ReplyMessage(
		event.ReplyToken,
		linebot.NewTextMessage("請上傳影片").WithQuickReplies(
			linebot.NewQuickReplyItems(
				linebot.NewQuickReplyButton(
					"",
					linebot.NewCameraAction("拍攝影片"),
				),
				linebot.NewQuickReplyButton(
					"",
					linebot.NewCameraRollAction("從相簿選擇"),
				),
			),
		),
	).Do()
	return err
}
