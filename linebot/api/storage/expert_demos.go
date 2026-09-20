package storage

import (
	"fmt"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

// Demonstrations live at expert-demo-video/<handedness>/<skill>/<expert>.mp4,
// with a .jpg beside each: a prefix listing, so adding an expert is an upload.
const DemoPrefix = "expert-demo-video"

// LINE draws a video message as a transparent rectangle until the file loads,
// so a demonstration without a thumbnail is skipped rather than sent bare.
const demoThumbnailExtension = ".jpg"

// The listing changes only on upload; the signed URLs inside it do not cache.
const demoListingTTL = time.Hour

// ExpertDemo is one expert's demonstration, ready to send.
type ExpertDemo struct {
	Expert    string
	Video     commons.MediaRef
	Thumbnail commons.MediaRef
}

type demoListing struct {
	experts []string
	read    time.Time
}

var (
	demoCacheMu sync.Mutex
	demoCache   = map[string]demoListing{}
)

// DemoObjectPath is where one expert's demonstration of a stroke lives.
func DemoObjectPath(handedness, skill, expert, extension string) string {
	return path.Join(DemoPrefix, handedness, skill, expert+extension)
}

// ExpertDemos returns up to limit demonstrations for one handedness and
// stroke, in a stable order, each signed for playback. An empty result is not
// an error: a stroke may have no demonstrations recorded yet.
func (c *BucketClient) ExpertDemos(handedness, skill, serviceAccountEmail string, limit int) ([]ExpertDemo, error) {
	experts, err := c.demoExperts(handedness, skill)
	if err != nil {
		return nil, err
	}

	demos := make([]ExpertDemo, 0, len(experts))
	for _, expert := range experts {
		if limit > 0 && len(demos) == limit {
			break
		}
		video, err := c.signObjectURL(c.bucketName,
			DemoObjectPath(handedness, skill, expert, ".mp4"), serviceAccountEmail)
		if err != nil {
			return nil, err
		}
		thumbnail, err := c.signObjectURL(c.bucketName,
			DemoObjectPath(handedness, skill, expert, demoThumbnailExtension), serviceAccountEmail)
		if err != nil {
			return nil, err
		}
		demos = append(demos, ExpertDemo{Expert: expert, Video: video, Thumbnail: thumbnail})
	}
	return demos, nil
}

// demoExperts lists the experts who have both a video and a thumbnail for this
// stroke, newest listing cached for an hour.
func (c *BucketClient) demoExperts(handedness, skill string) ([]string, error) {
	if strings.TrimSpace(handedness) == "" || strings.TrimSpace(skill) == "" {
		return nil, fmt.Errorf("demo listing needs a handedness and a skill")
	}
	key := c.bucketName + "/" + handedness + "/" + skill

	demoCacheMu.Lock()
	defer demoCacheMu.Unlock()
	if cached, ok := demoCache[key]; ok && time.Since(cached.read) < demoListingTTL {
		return cached.experts, nil
	}

	prefix := path.Join(DemoPrefix, handedness, skill) + "/"
	names, err := c.client.Bucket(c.bucketName).ObjectNames(c.ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("list demonstrations under %q: %w", prefix, err)
	}

	videos, thumbnails := map[string]bool{}, map[string]bool{}
	for _, name := range names {
		base := strings.TrimPrefix(name, prefix)
		if base == "" || strings.Contains(base, "/") {
			continue
		}
		switch strings.ToLower(path.Ext(base)) {
		case ".mp4":
			videos[strings.TrimSuffix(base, path.Ext(base))] = true
		case demoThumbnailExtension:
			thumbnails[strings.TrimSuffix(base, path.Ext(base))] = true
		}
	}

	experts := make([]string, 0, len(videos))
	for expert := range videos {
		if thumbnails[expert] {
			experts = append(experts, expert)
		}
	}
	sort.Strings(experts)

	demoCache[key] = demoListing{experts: experts, read: time.Now()}
	return experts, nil
}
