package line

import (
	"io"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

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
	if err != nil {
		return err
	}
	return nil
}

func (client *Client) GetVideoContent(msgID string) ([]byte, error) {
	contentResp, err := client.bot.GetMessageContent(msgID).Do()
	if err != nil {
		return nil, err
	}
	defer contentResp.Content.Close()

	// Read the body into a byte slice
	blob, err := io.ReadAll(contentResp.Content)
	if err != nil {
		return nil, err
	}
	return blob, nil
}
