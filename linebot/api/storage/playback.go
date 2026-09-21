package storage

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	gcs "cloud.google.com/go/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"google.golang.org/api/googleapi"
)

// PlaybackURLTTL is how long a playback link stays valid. Long enough for a
// student to watch and rewatch, short enough that a leaked link expires.
const PlaybackURLTTL = 60 * time.Minute

// playablePrefixes are the only object trees a playback URL may be minted for.
// Object paths arrive from stored analyses, but signing is a capability worth
// fencing: a bug elsewhere must not be able to hand out a link to the bucket's
// private corners.
var playablePrefixes = []string{"analyses/", "experts/", "noai/analyses/", "no-ai/analyses/"}

func playbackContentType(objectPath string) string {
	switch strings.ToLower(objectPath[strings.LastIndex(objectPath, ".")+1:]) {
	case "jpg", "jpeg":
		return "image/jpeg"
	case "png":
		return "image/png"
	case "webp":
		return "image/webp"
	}
	if strings.HasSuffix(strings.ToLower(objectPath), ".mov") {
		return "video/quicktime"
	}
	return "video/mp4"
}

// PlayableObject reports whether a path may be signed for playback.
func PlayableObject(objectPath string) bool {
	if objectPath == "" || strings.Contains(objectPath, "..") {
		return false
	}
	for _, prefix := range playablePrefixes {
		if strings.HasPrefix(objectPath, prefix) {
			return true
		}
	}
	return false
}

func BucketFromGCSURI(uri string) string {
	rest, found := strings.CutPrefix(strings.TrimSpace(uri), "gs://")
	if !found {
		return ""
	}
	bucket, _, _ := strings.Cut(rest, "/")
	return bucket
}

// SignPlaybackURLIn mints a read URL for one stored object, in the bucket the
// analysis recorded (empty means this client's). Go signs it rather than the
// GPU service, so playback never waits on a scaled-to-zero GPU; on Cloud Run
// that signing goes through IAM, needing serviceAccountTokenCreator on itself.
func (c *BucketClient) SignPlaybackURLIn(bucketName, objectPath, serviceAccountEmail string) (commons.MediaRef, error) {
	if strings.TrimSpace(bucketName) == "" {
		bucketName = c.bucketName
	}
	if !PlayableObject(objectPath) {
		return commons.MediaRef{}, fmt.Errorf("object path is not playable: %q", objectPath)
	}
	return c.signObjectURL(bucketName, objectPath, serviceAccountEmail)
}

// Signing goes through the IAM signBytes API, which now and then answers 5xx.
// A retry costs milliseconds; not retrying costs the learner their playback.
const (
	signAttempts     = 3
	signRetryBackoff = 100 * time.Millisecond
)

// signRetryable tells a bad moment at the signing backend from a refusal: a
// 5xx or a throttle is worth another try, a 401/403/404 never will be.
func signRetryable(err error) bool {
	var apiErr *googleapi.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.Code >= 500 || apiErr.Code == http.StatusTooManyRequests
}

// Only callers that have validated their specific media path may use this.
func (c *BucketClient) signObjectURL(bucketName, objectPath, serviceAccountEmail string) (commons.MediaRef, error) {
	expires := time.Now().Add(PlaybackURLTTL)
	opts := &gcs.SignedURLOptions{
		Method:  "GET",
		Expires: expires,
		Scheme:  gcs.SigningSchemeV4,
		// Older expert objects have no GCS Content-Type. LIFF's embedded browser
		// does not consistently sniff an octet-stream as video, so make the
		// signed response explicitly playable.
		QueryParameters: url.Values{
			"response-content-type": []string{playbackContentType(objectPath)},
		},
	}
	if trimmed := strings.TrimSpace(serviceAccountEmail); trimmed != "" {
		opts.GoogleAccessID = trimmed
	}
	var url string
	var err error
	for attempt := 0; attempt < signAttempts; attempt++ {
		url, err = c.client.Bucket(bucketName).SignedURL(objectPath, opts)
		if err == nil || !signRetryable(err) {
			break
		}
		time.Sleep(time.Duration(attempt+1) * signRetryBackoff)
	}
	if err != nil {
		return commons.MediaRef{}, fmt.Errorf("sign playback URL for %q: %w", objectPath, err)
	}
	return commons.MediaRef{
		ObjectPath:       objectPath,
		GCSURI:           fmt.Sprintf("gs://%s/%s", bucketName, objectPath),
		SignedURL:        url,
		SignedURLExpires: expires.Unix(),
	}, nil
}
