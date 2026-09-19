package line

import (
	"encoding/json"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

// VideoContent is a playable video and its preview image.
type VideoContent struct {
	URL          string
	ThumbnailURL string
}

// ReplyChatAnalysis answers a video sent in coach chat with one reply: the
// analysis render to watch, then the coach's explanation.
//
// One reply carries both, which keeps a finished analysis off the account's
// push quota. An empty video -- a render that failed to sign -- still sends
// the explanation rather than nothing.
func (client *Client) ReplyChatAnalysis(replyToken string, video VideoContent, answer string) error {
	data, err := json.Marshal(StopGPTPostback{Stop: true})
	if err != nil {
		return err
	}
	text := linebot.NewTextMessage(answer).WithQuickReplies(&linebot.QuickReplyItems{
		Items: []*linebot.QuickReplyButton{
			linebot.NewQuickReplyButton("", linebot.NewPostbackAction(
				"結束對話", string(data), "", "結束對話", "OpenRichMenu", "",
			)),
		},
	})
	messages := []linebot.SendingMessage{}
	if video.URL != "" {
		messages = append(messages, linebot.NewVideoMessage(client.assetURL(video.URL), client.assetURL(video.ThumbnailURL)))
	}
	messages = append(messages, text)
	_, err = client.bot.ReplyMessage(replyToken, messages...).Do()
	return err
}
