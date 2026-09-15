package storage

import (
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// Read compatibility only. New uploads still use analyses/thumbnail.
// Do not permit arbitrary objects under a user directory.
var legacyThumbnailPath = regexp.MustCompile(`^U[0-9a-fA-F]{32}/thumbnail/[A-Za-z0-9_-]+\.(?i:jpg|jpeg|png|webp)$`)

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
	if !PlayableObject(object) && !legacyThumbnailPath.MatchString(object) {
		return commons.MediaRef{}, fmt.Errorf("unsupported thumbnail object path")
	}
	// Signing does not check existence. A missing historical image must be
	// omitted by the caller, not turned into a signed 404 in a LINE card.
	ctx, cancel := context.WithTimeout(c.ctx, 3*time.Second)
	defer cancel()
	if _, err := c.client.Bucket(c.bucketName).Object(object).Attrs(ctx); err != nil {
		return commons.MediaRef{}, fmt.Errorf("thumbnail unavailable: %w", err)
	}
	return c.signObjectURL(c.bucketName, object, signer)
}
