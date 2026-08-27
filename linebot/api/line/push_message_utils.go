package line

import (
	"fmt"

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
