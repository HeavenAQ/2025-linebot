package storage

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// SignThumbnailURL accepts stored object paths and historical GCS URLs, not
// arbitrary web URLs. Thumbnails remain private and are signed when displayed.
func (c *BucketClient) SignThumbnailURL(value, signer string) (commons.MediaRef, error) {
	object := value
	if strings.Contains(value, "://") {
		u, err := url.Parse(value)
		if err != nil {
			return commons.MediaRef{}, fmt.Errorf("invalid thumbnail reference")
		}
		switch {
		case u.Scheme == "gs" && u.Host == c.bucketName:
			object = strings.TrimPrefix(u.Path, "/")
		case u.Scheme == "https" && u.Host == "storage.googleapis.com":
			bucket, path, ok := strings.Cut(strings.TrimPrefix(u.Path, "/"), "/")
			if !ok || bucket != c.bucketName {
				return commons.MediaRef{}, fmt.Errorf("thumbnail bucket mismatch")
			}
			object = path
		default:
			return commons.MediaRef{}, fmt.Errorf("unsupported thumbnail reference")
		}
	}
	switch strings.ToLower(object[strings.LastIndex(object, ".")+1:]) {
	case "jpg", "jpeg", "png", "webp":
	default:
		return commons.MediaRef{}, fmt.Errorf("thumbnail must be an image")
	}
	return c.SignPlaybackURL(object, signer)
}
