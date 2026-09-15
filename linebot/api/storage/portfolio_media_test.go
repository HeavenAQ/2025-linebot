package storage

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPortfolioMediaPrefixesAndBucket(t *testing.T) {
	c, _ := playbackClient(t)
	for _, prefix := range []string{"analyses/", "noai/analyses/", "no-ai/analyses/"} {
		media, err := c.SignPlaybackURLIn("shared-bucket", prefix+"v1/U1/r/student.mp4", "svc@project.iam")
		require.NoError(t, err)
		require.Contains(t, media.SignedURL, "/shared-bucket/"+prefix)
	}
	for _, path := range []string{"noai/secrets/key.json", "no-ai/../secrets", "noai/analyses/../private"} {
		require.False(t, PlayableObject(path))
	}
}

func TestPrivatePortfolioThumbnailSignedWithImageContentType(t *testing.T) {
	c, bucket := playbackClient(t)
	for _, value := range []string{
		"no-ai/analyses/thumbnail/U1/t.jpeg",
		"https://storage.googleapis.com/nstc-2025-storage/no-ai/analyses/thumbnail/U1/t.jpeg?old=expired",
		"gs://nstc-2025-storage/analyses/thumbnail/U1/t.jpeg",
	} {
		media, err := c.SignThumbnailURL(value, "svc@project.iam")
		require.NoError(t, err)
		require.Contains(t, media.SignedURL, "https://signed.test/")
		require.NotContains(t, media.SignedURL, "old=expired")
		require.Equal(t, "image/jpeg", bucket.signedOptions[len(bucket.signedOptions)-1].QueryParameters.Get("response-content-type"))
	}
	for _, value := range []string{
		"https://evil.test/analyses/thumbnail/t.jpeg",
		"gs://another-bucket/analyses/thumbnail/t.jpeg",
		"no-ai/secrets/t.jpeg", "analyses/private.json",
	} {
		_, err := c.SignThumbnailURL(value, "")
		require.Error(t, err)
	}
}
