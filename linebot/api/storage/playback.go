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
var playablePrefixes = []string{"analyses/", "experts/"}

func playbackContentType(objectPath string) string {
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

// SignPlaybackURL mints a read URL for one stored object.
//
// Go signs these itself rather than asking the analysis service to, so opening
// a video never depends on a GPU instance being awake. That service scales to
// zero; routing playback through it would make the first view after an idle
// period wait for a cold start.
//
// Signing uses whatever credentials the process has: a key file signs locally,
// while on Cloud Run the metadata credentials sign through IAM, which needs the
// service account to hold roles/iam.serviceAccountTokenCreator on itself.
func (c *BucketClient) SignPlaybackURL(objectPath string, serviceAccountEmail string) (commons.MediaRef, error) {
	if !PlayableObject(objectPath) {
		return commons.MediaRef{}, fmt.Errorf("object path is not playable: %q", objectPath)
	}
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
	url, err := c.client.Bucket(c.bucketName).SignedURL(objectPath, opts)
	if err != nil {
		return commons.MediaRef{}, fmt.Errorf("sign playback URL for %q: %w", objectPath, err)
	}
	return commons.MediaRef{
		ObjectPath:       objectPath,
		GCSURI:           fmt.Sprintf("gs://%s/%s", c.bucketName, objectPath),
		SignedURL:        url,
		SignedURLExpires: expires.Unix(),
	}, nil
}
