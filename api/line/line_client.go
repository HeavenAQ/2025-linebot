package line

import (
	"fmt"
	"net/http"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type Client struct {
	bot *linebot.Client
}

// NewBotClient creates a new BotClient instance
func NewBotClient(channelSecret, channelToken string) (*Client, error) {
	bot, err := linebot.New(channelSecret, channelToken)
	if err != nil {
		return nil, fmt.Errorf("failed to create linebot client: %w", err)
	}

	return &Client{bot: bot}, nil
}

// ParseRequest wraps the linebot.Client's ParseRequest method
func (client *Client) ParseRequest(r *http.Request) ([]*linebot.Event, error) {
	res, err := client.bot.ParseRequest(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	return res, nil
}
