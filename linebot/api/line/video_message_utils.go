package line

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

const (
	videoContentAttempts   = 16
	videoContentRetryDelay = 2 * time.Second
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
	return readVideoContent(
		func() (*linebot.MessageContentResponse, error) {
			return client.bot.GetMessageContent(msgID).Do()
		},
		videoContentAttempts,
		func() { time.Sleep(videoContentRetryDelay) },
	)
}

func readVideoContent(
	fetch func() (*linebot.MessageContentResponse, error),
	attempts int,
	wait func(),
) ([]byte, error) {
	if attempts < 1 {
		return nil, fmt.Errorf("video content attempts must be positive")
	}
	var lastContentType string
	for attempt := 1; attempt <= attempts; attempt++ {
		contentResp, err := fetch()
		if err != nil {
			return nil, err
		}
		lastContentType = contentResp.ContentType
		blob, readErr := io.ReadAll(contentResp.Content)
		closeErr := contentResp.Content.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if len(blob) > 0 && strings.HasPrefix(lastContentType, "video/") {
			return blob, nil
		}
		if attempt < attempts {
			wait()
		}
	}
	return nil, fmt.Errorf(
		"LINE video content is still processing after %d attempts (content-type %q)",
		attempts,
		lastContentType,
	)
}
