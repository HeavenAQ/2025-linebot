package line

import (
	"net/http"

	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type LineBotClient interface {
	ParseRequest(*http.Request) ([]*linebot.Event, error)
	ReplyMessage(string, ...linebot.SendingMessage) *linebot.ReplyMessageCall
	ReplyWithTypeError(token string)
}

type LineBot struct {
	bot *linebot.Client
}

// NewBotClient creates a new BotClient instance
func NewBotClient(channelSecret, channelToken string) (*LineBot, error) {
	bot, err := linebot.New(channelSecret, channelToken)
	if err != nil {
		return nil, err
	}
	return &LineBot{bot: bot}, nil
}

// ParseRequest wraps the linebot.Client's ParseRequest method
func (b *LineBot) ParseRequest(r *http.Request) ([]*linebot.Event, error) {
	return b.bot.ParseRequest(r)
}

// ReplyMessage wraps the linebot.Client's ReplyMessage method
func (b *LineBot) ReplyMessage(replyToken string, messages ...linebot.SendingMessage) (*linebot.BasicResponse, error) {
	return b.bot.ReplyMessage(replyToken, messages...).Do()
}

func (b *LineBot) ReplyWithTypeError(replyToken string) {
	b.bot.ReplyMessage(replyToken, linebot.NewTextMessage("抱歉，您所輸入的訊息格式目前並未支援，請重試一次！")).Do()
}
