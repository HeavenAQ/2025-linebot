package storage

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	gcs "cloud.google.com/go/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
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
// analysis recorded (an empty name means this client's bucket).
//
// Go signs these itself rather than asking the analysis service to, so opening
// a video never depends on the GPU service being awake; its minimum capacity is
// scheduled and is zero outside class.
//
// Signing uses whatever credentials the process has: a key file signs locally,
// while on Cloud Run the metadata credentials sign through IAM, which needs the
// service account to hold roles/iam.serviceAccountTokenCreator on itself.
func (c *BucketClient) SignPlaybackURLIn(bucketName, objectPath, serviceAccountEmail string) (commons.MediaRef, error) {
	if strings.TrimSpace(bucketName) == "" {
		bucketName = c.bucketName
	}
	if !PlayableObject(objectPath) {
		return commons.MediaRef{}, fmt.Errorf("object path is not playable: %q", objectPath)
	}
	return c.signObjectURL(bucketName, objectPath, serviceAccountEmail)
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
	url, err := c.client.Bucket(bucketName).SignedURL(objectPath, opts)
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
