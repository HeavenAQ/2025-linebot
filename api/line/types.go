package line

type CarouselBtn int8

const (
	VideoLink CarouselBtn = iota
	VideoDate
)

type VideoInfo struct {
	VideoID     string `json:"video_id"`
	ThumbnailID string `json:"thumbnail_id"`
}
