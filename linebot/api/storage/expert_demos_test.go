package storage

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func demoClient(t *testing.T, objects ...string) (*BucketClient, *fakeBucket) {
	t.Helper()
	resetDemoCache()
	client := newFakeClient()
	bucket := client.Bucket("demo-bucket").(*fakeBucket)
	for _, name := range objects {
		bucket.objects[name] = &fakeObject{data: []byte("x")}
	}
	return NewBucketClientWithClient(context.Background(), client, "demo-bucket"), bucket
}

func resetDemoCache() {
	demoCacheMu.Lock()
	defer demoCacheMu.Unlock()
	demoCache = map[string]demoListing{}
}

func TestExpertDemosReadsTheHandednessAndSkillPrefix(t *testing.T) {
	client, bucket := demoClient(t,
		"expert-demo-video/right/serve/expert-2.mp4",
		"expert-demo-video/right/serve/expert-2.jpg",
		"expert-demo-video/right/serve/expert-1.mp4",
		"expert-demo-video/right/serve/expert-1.jpg",
		// A different hand and a different stroke must not leak in.
		"expert-demo-video/left/serve/expert-1.mp4",
		"expert-demo-video/left/serve/expert-1.jpg",
		"expert-demo-video/right/smash/expert-9.mp4",
		"expert-demo-video/right/smash/expert-9.jpg",
	)

	demos, err := client.ExpertDemos("right", "serve", "signer@example.com", 0)
	require.NoError(t, err)
	require.Equal(t, []string{"expert-1", "expert-2"}, []string{demos[0].Expert, demos[1].Expert},
		"experts come back in a stable order, so a learner sees the same list each time")
	require.Equal(t, []string{"expert-demo-video/right/serve/"}, bucket.listed)
	require.Equal(t, "expert-demo-video/right/serve/expert-1.mp4", demos[0].Video.ObjectPath)
	require.Equal(t, "expert-demo-video/right/serve/expert-1.jpg", demos[0].Thumbnail.ObjectPath)
	require.NotEmpty(t, demos[0].Video.SignedURL)
	require.NotEmpty(t, demos[0].Thumbnail.SignedURL)
}

// LINE renders a video message as a transparent rectangle until the file
// loads, so a demonstration without a thumbnail is worse than one fewer.
func TestExpertDemosSkipsAVideoWithNoThumbnail(t *testing.T) {
	client, _ := demoClient(t,
		"expert-demo-video/right/serve/expert-1.mp4",
		"expert-demo-video/right/serve/expert-2.mp4",
		"expert-demo-video/right/serve/expert-2.jpg",
	)

	demos, err := client.ExpertDemos("right", "serve", "signer@example.com", 0)
	require.NoError(t, err)
	require.Len(t, demos, 1)
	require.Equal(t, "expert-2", demos[0].Expert)
}

func TestExpertDemosStopsAtTheLimit(t *testing.T) {
	client, _ := demoClient(t,
		"expert-demo-video/right/serve/expert-1.mp4",
		"expert-demo-video/right/serve/expert-1.jpg",
		"expert-demo-video/right/serve/expert-2.mp4",
		"expert-demo-video/right/serve/expert-2.jpg",
		"expert-demo-video/right/serve/expert-3.mp4",
		"expert-demo-video/right/serve/expert-3.jpg",
	)

	demos, err := client.ExpertDemos("right", "serve", "signer@example.com", 2)
	require.NoError(t, err)
	require.Len(t, demos, 2, "a reply carries five messages, one of them the intro text")
}

func TestExpertDemosIsEmptyWhenNothingIsUploaded(t *testing.T) {
	client, _ := demoClient(t)

	demos, err := client.ExpertDemos("right", "clear", "signer@example.com", 4)
	require.NoError(t, err, "a stroke with no demonstrations is not a failure")
	require.Empty(t, demos)
}

func TestExpertDemosReportsAFailedListing(t *testing.T) {
	client, bucket := demoClient(t)
	bucket.listErr = errors.New("permission denied")

	_, err := client.ExpertDemos("right", "serve", "signer@example.com", 4)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expert-demo-video/right/serve/")
}

// The listing changes only when someone uploads, so it is read once an hour;
// the signed URLs inside it are minted fresh on every reply.
func TestExpertDemosCachesTheListingButNotTheURLs(t *testing.T) {
	client, bucket := demoClient(t,
		"expert-demo-video/right/serve/expert-1.mp4",
		"expert-demo-video/right/serve/expert-1.jpg",
	)

	first, err := client.ExpertDemos("right", "serve", "signer@example.com", 4)
	require.NoError(t, err)
	second, err := client.ExpertDemos("right", "serve", "signer@example.com", 4)
	require.NoError(t, err)

	require.Len(t, bucket.listed, 1, "the second reply reuses the cached listing")
	require.Len(t, bucket.signed, 4, "but every reply signs its own URLs")
	require.Equal(t, first[0].Expert, second[0].Expert)
}

func TestExpertDemosRefusesAnEmptySkill(t *testing.T) {
	client, _ := demoClient(t)

	_, err := client.ExpertDemos("right", "", "signer@example.com", 4)
	require.Error(t, err)
}
