package storage

import (
	"fmt"
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
// BucketFromGCSURI returns the bucket named by a "gs://bucket/object" URI, or
// "" when the URI is empty or malformed.
//
// Analyses are written by the analysis service into whichever bucket that
// service is configured with, which is not necessarily this deployment's own:
// a deployment that shares the analysis service shares its output bucket too.
// The bucket that actually holds an object is therefore a property of the
// stored media reference, not of the bot's configuration.
func BucketFromGCSURI(uri string) string {
	rest, found := strings.CutPrefix(strings.TrimSpace(uri), "gs://")
	if !found {
		return ""
	}
	bucket, _, _ := strings.Cut(rest, "/")
	return bucket
}

// SignPlaybackURL signs against this client's own bucket.
func (c *BucketClient) SignPlaybackURL(objectPath string, serviceAccountEmail string) (commons.MediaRef, error) {
	return c.SignPlaybackURLIn(c.bucketName, objectPath, serviceAccountEmail)
}

// SignPlaybackURLIn signs an object in an explicit bucket. Callers pass the
// bucket recorded alongside the object so playback still works for analyses
// written elsewhere; an empty bucket falls back to this client's own.
func (c *BucketClient) SignPlaybackURLIn(
	bucketName string, objectPath string, serviceAccountEmail string,
) (commons.MediaRef, error) {
	if !PlayableObject(objectPath) {
		return commons.MediaRef{}, fmt.Errorf("object path is not playable: %q", objectPath)
	}
	if strings.TrimSpace(bucketName) == "" {
		bucketName = c.bucketName
	}
	expires := time.Now().Add(PlaybackURLTTL)
	opts := &gcs.SignedURLOptions{
		Method:  "GET",
		Expires: expires,
		Scheme:  gcs.SigningSchemeV4,
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
