package review

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"
)

var (
	// ErrNotFound means the document does not exist.
	ErrNotFound = errors.New("not found")
	// ErrStep1Locked means step 1 was already submitted and cannot change.
	ErrStep1Locked = errors.New("step 1 already submitted")
	// ErrStep1Required means step 2 was attempted before step 1.
	ErrStep1Required = errors.New("step 1 not submitted")
)

// Store is the persistence the handlers need. The lock rules (step 1 is
// write-once, step 2 requires step 1) live in the implementations' atomic
// operations so concurrent requests cannot bypass them.
type Store interface {
	// ListItems returns every item in the batch (eligible or not).
	ListItems(ctx context.Context, batchID string) ([]Item, error)
	// GetItem returns ErrNotFound when the item does not exist.
	GetItem(ctx context.Context, itemID string) (Item, error)
	// GetRating returns ErrNotFound when the expert has no rating yet.
	GetRating(ctx context.Context, itemID, expertID string) (Rating, error)
	// ListRatings returns ratings in the batch; expertID "" means all experts.
	ListRatings(ctx context.Context, batchID, expertID string) ([]Rating, error)
	// SubmitStep1 creates the rating atomically; ErrStep1Locked if step 1 exists.
	SubmitStep1(ctx context.Context, r Rating) error
	// SubmitStep2 updates the rating atomically; ErrStep1Required before step 1.
	SubmitStep2(ctx context.Context, itemID, expertID string, s Step2, now time.Time) error
}

// step2Update computes the fields a step 2 save writes. Seconds accumulate so
// revisits add to the time spent on step 2 rather than replacing it.
func step2Update(existing Rating, s Step2, now time.Time) map[string]any {
	cues := make(map[string]any, len(s.CueRatings))
	for k, v := range s.CueRatings {
		cues[k] = v
	}
	seconds := s.Seconds
	if existing.Step2Seconds != nil {
		seconds += *existing.Step2Seconds
	}
	return map[string]any{
		"cue_ratings":        cues,
		"overall_score":      int64(s.OverallScore),
		"comment":            s.Comment,
		"step2_submitted_at": now,
		"step2_seconds":      seconds,
		"completed":          true,
		"updated_at":         now,
	}
}

// MemoryStore is an in-memory Store for tests and local development.
type MemoryStore struct {
	mu      sync.Mutex
	items   map[string]Item
	ratings map[string]Rating
}

// NewMemoryStore returns a store holding items.
func NewMemoryStore(items ...Item) *MemoryStore {
	m := &MemoryStore{items: map[string]Item{}, ratings: map[string]Rating{}}
	for _, it := range items {
		m.items[it.ItemID] = it
	}
	return m
}

func (m *MemoryStore) ListItems(_ context.Context, batchID string) ([]Item, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Item
	for _, it := range m.items {
		if it.BatchID == batchID {
			out = append(out, it)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ItemID < out[j].ItemID })
	return out, nil
}

func (m *MemoryStore) GetItem(_ context.Context, itemID string) (Item, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	it, ok := m.items[itemID]
	if !ok {
		return Item{}, ErrNotFound
	}
	return it, nil
}

func (m *MemoryStore) GetRating(_ context.Context, itemID, expertID string) (Rating, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	r, ok := m.ratings[RatingDocID(itemID, expertID)]
	if !ok {
		return Rating{}, ErrNotFound
	}
	return r, nil
}

func (m *MemoryStore) ListRatings(_ context.Context, batchID, expertID string) ([]Rating, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []Rating
	for _, r := range m.ratings {
		if r.BatchID == batchID && (expertID == "" || r.ExpertID == expertID) {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return RatingDocID(out[i].ItemID, out[i].ExpertID) < RatingDocID(out[j].ItemID, out[j].ExpertID)
	})
	return out, nil
}

func (m *MemoryStore) SubmitStep1(_ context.Context, r Rating) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := RatingDocID(r.ItemID, r.ExpertID)
	if existing, ok := m.ratings[id]; ok && existing.Step1Done() {
		return ErrStep1Locked
	}
	// Round-trip through the document shape so the fake matches Firestore.
	m.ratings[id] = RatingFromData(r.Step1Data())
	return nil
}

func (m *MemoryStore) SubmitStep2(_ context.Context, itemID, expertID string, s Step2, now time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := RatingDocID(itemID, expertID)
	existing, ok := m.ratings[id]
	if !ok || !existing.Step1Done() {
		return ErrStep1Required
	}
	data := existing.Step1Data()
	if existing.Step2Seconds != nil {
		data["step2_seconds"] = *existing.Step2Seconds
	}
	for k, v := range step2Update(existing, s, now) {
		data[k] = v
	}
	m.ratings[id] = RatingFromData(data)
	return nil
}
