package storage

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func playbackClient(t *testing.T) (*BucketClient, *fakeBucket) {
	t.Helper()
	client := newFakeClient()
	bucket := client.Bucket("nstc-2025-storage").(*fakeBucket)
	return &BucketClient{client: client, bucketName: "nstc-2025-storage"}, bucket
}

func TestSignPlaybackURLReturnsATimeLimitedLink(t *testing.T) {
	t.Parallel()

	client, bucket := playbackClient(t)
	media, err := client.SignPlaybackURL(
		"analyses/v1/U123/req/student_corrected.mp4",
		"svc@project.iam.gserviceaccount.com",
	)

	require.NoError(t, err)
	require.Contains(t, media.SignedURL, "analyses/v1/U123/req/student_corrected.mp4")
	require.Contains(t, media.SignedURL, "as=svc@project.iam.gserviceaccount.com")
	require.Equal(t, "analyses/v1/U123/req/student_corrected.mp4", media.ObjectPath)
	require.Equal(t, "gs://nstc-2025-storage/analyses/v1/U123/req/student_corrected.mp4", media.GCSURI)
	require.Positive(t, media.SignedURLExpires)
	require.Equal(t, []string{"analyses/v1/U123/req/student_corrected.mp4"}, bucket.signed)
	require.Equal(t, "video/mp4", bucket.signedOptions[0].QueryParameters.Get("response-content-type"))
}

func TestSignPlaybackURLMarksMovExpertAsQuickTimeVideo(t *testing.T) {
	t.Parallel()

	client, bucket := playbackClient(t)
	_, err := client.SignPlaybackURL(
		"experts/v3/serve/videos/expert.mov",
		"svc@project.iam.gserviceaccount.com",
	)

	require.NoError(t, err)
	require.Equal(t, "video/quicktime", bucket.signedOptions[0].QueryParameters.Get("response-content-type"))
}

// Signing is a capability: only the trees that hold playable output may be
// signed, so a bad path elsewhere cannot mint a link into the bucket.
func TestSignPlaybackURLRefusesPathsOutsidePlayableTrees(t *testing.T) {
	t.Parallel()

	client, bucket := playbackClient(t)
	for _, path := range []string{
		"", "secrets/key.json", "analyses/../secrets/key.json",
		"U123/thumbnail.jpeg", "/analyses/v1/x.mp4",
	} {
		_, err := client.SignPlaybackURL(path, "svc@project.iam.gserviceaccount.com")
		require.Error(t, err, path)
	}
	require.Empty(t, bucket.signed, "nothing outside a playable tree should reach the signer")
}

func TestPlayableObjectAcceptsAnalysisAndExpertTrees(t *testing.T) {
	t.Parallel()

	require.True(t, PlayableObject("analyses/v1/U1/r/student_corrected.mp4"))
	require.True(t, PlayableObject("experts/v1/serve/videos/nstc_left_1.mov"))
	require.False(t, PlayableObject("analyses/../experts/x.mp4"))
}

func TestSignPlaybackURLSurfacesSignerFailures(t *testing.T) {
	t.Parallel()

	client, bucket := playbackClient(t)
	bucket.signErr = errors.New("iam: permission denied")

	_, err := client.SignPlaybackURL("analyses/v1/U1/r/student_corrected.mp4", "svc@x.iam")
	require.ErrorContains(t, err, "iam: permission denied")
}

// Without an explicit account the signer auto-detects, so the option is left
// empty rather than set to a blank string.
func TestSignPlaybackURLOmitsAnEmptyServiceAccount(t *testing.T) {
	t.Parallel()

	client, _ := playbackClient(t)
	media, err := client.SignPlaybackURL("analyses/v1/U1/r/student_corrected.mp4", "  ")

	require.NoError(t, err)
	require.Contains(t, media.SignedURL, "as=")
	require.NotContains(t, media.SignedURL, "as=  ")
}
