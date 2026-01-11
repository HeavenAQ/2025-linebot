package line

import (
    "fmt"
    "net/http"
    "strings"

    "github.com/line/line-bot-sdk-go/v7/linebot"
)

type Client struct {
	bot        *linebot.Client
	bucketName string
}

// NewBotClient creates a new BotClient instance
func NewBotClient(channelSecret, channelToken, bucketName string) (*Client, error) {
	bot, err := linebot.New(channelSecret, channelToken)
	if err != nil {
		return nil, fmt.Errorf("failed to create linebot client: %w", err)
	}

	return &Client{bot: bot, bucketName: bucketName}, nil
}

// ParseRequest wraps the linebot.Client's ParseRequest method
func (client *Client) ParseRequest(r *http.Request) ([]*linebot.Event, error) {
	res, err := client.bot.ParseRequest(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse request: %w", err)
	}
	return res, nil
}

// assetURL returns a fully-qualified HTTPS URL for a GCS object path.
// If the input already looks like an http(s) URL, it is returned as-is.
func (client *Client) assetURL(pathOrURL string) string {
    if strings.HasPrefix(pathOrURL, "http://") || strings.HasPrefix(pathOrURL, "https://") {
        return pathOrURL
    }
    p := strings.TrimLeft(pathOrURL, "/")
    return "https://storage.googleapis.com/" + client.bucketName + "/" + p
}
