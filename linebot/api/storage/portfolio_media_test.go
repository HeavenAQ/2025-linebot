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
	bucket.objects["no-ai/analyses/thumbnail/U1/t.jpeg"] = &fakeObject{}
	bucket.objects["analyses/thumbnail/U1/t.jpeg"] = &fakeObject{}
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

func TestLegacyThumbnailReadCompatibilityDoesNotBroadenVideoSigning(t *testing.T) {
	c, bucket := playbackClient(t)
	object := "U0123456789abcdef0123456789abcdef/thumbnail/2026-08-02-02-22.jpeg"
	bucket.objects[object] = &fakeObject{}
	for _, value := range []string{object, "gs://nstc-2025-storage/" + object,
		"https://storage.googleapis.com/nstc-2025-storage/" + object + "?old=expired"} {
		media, err := c.SignThumbnailURL(value, "svc@project.iam")
		require.NoError(t, err)
		require.Equal(t, object, media.ObjectPath)
		require.Equal(t, "image/jpeg", bucket.signedOptions[len(bucket.signedOptions)-1].QueryParameters.Get("response-content-type"))
	}
	require.False(t, PlayableObject(object))
	bucket.signed = nil
	for _, value := range []string{
		"U0123456789abcdef0123456789abcdef/private/file.jpeg",
		"U0123456789abcdef0123456789abcdef/thumbnail/../file.jpeg",
		"U0123456789abcdef0123456789abcdef/thumbnail/file.mp4",
		"unknown/thumbnail/file.jpeg",
	} {
		_, err := c.SignThumbnailURL(value, "")
		require.Error(t, err)
	}
	require.Empty(t, bucket.signed)
	delete(bucket.objects, object)
	_, err := c.SignThumbnailURL(object, "")
	require.ErrorContains(t, err, "thumbnail unavailable")
	require.Empty(t, bucket.signed, "a missing thumbnail must not become a signed 404")
}
