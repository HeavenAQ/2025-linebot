package line

import (
	"github.com/line/line-bot-sdk-go/v7/linebot"
)

type LineBotHandler struct {
	bot *linebot.Client
}

type CarouselBtn int8

const (
	VideoLink CarouselBtn = iota
	VideoDate
)

type VideoInfo struct {
	VideoId     string `json:"video_id"`
	ThumbnailId string `json:"thumbnail_id"`
}
