package review

import (
	"context"
	"errors"
	"fmt"
	"time"

	"cloud.google.com/go/firestore"
	"google.golang.org/api/iterator"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// FirestoreStore implements Store on Cloud Firestore.
type FirestoreStore struct {
	client *firestore.Client
}

// NewFirestoreStore wraps a Firestore client.
func NewFirestoreStore(client *firestore.Client) *FirestoreStore {
	return &FirestoreStore{client: client}
}

func isNotFound(err error) bool { return status.Code(err) == codes.NotFound }

func (s *FirestoreStore) ListItems(ctx context.Context, batchID string) ([]Item, error) {
	iter := s.client.Collection(ItemsCollection).Where("batch_id", "==", batchID).Documents(ctx)
	defer iter.Stop()
	var items []Item
	for {
		snap, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("list items: %w", err)
		}
		items = append(items, ItemFromData(snap.Ref.ID, snap.Data()))
	}
	return items, nil
}

func (s *FirestoreStore) GetItem(ctx context.Context, itemID string) (Item, error) {
	ref := s.client.Collection(ItemsCollection).Doc(itemID)
	if ref == nil {
		return Item{}, ErrNotFound
	}
	snap, err := ref.Get(ctx)
	if isNotFound(err) {
		return Item{}, ErrNotFound
	}
	if err != nil {
		return Item{}, fmt.Errorf("get item: %w", err)
	}
	return ItemFromData(snap.Ref.ID, snap.Data()), nil
}

func (s *FirestoreStore) ratingRef(itemID, expertID string) *firestore.DocumentRef {
	return s.client.Collection(RatingsCollection).Doc(RatingDocID(itemID, expertID))
}

func (s *FirestoreStore) GetRating(ctx context.Context, itemID, expertID string) (Rating, error) {
	ref := s.ratingRef(itemID, expertID)
	if ref == nil {
		return Rating{}, ErrNotFound
	}
	snap, err := ref.Get(ctx)
	if isNotFound(err) {
		return Rating{}, ErrNotFound
	}
	if err != nil {
		return Rating{}, fmt.Errorf("get rating: %w", err)
	}
	return RatingFromData(snap.Data()), nil
}

func (s *FirestoreStore) ListRatings(ctx context.Context, batchID, expertID string) ([]Rating, error) {
	q := s.client.Collection(RatingsCollection).Where("batch_id", "==", batchID)
	if expertID != "" {
		q = q.Where("expert_id", "==", expertID)
	}
	iter := q.Documents(ctx)
	defer iter.Stop()
	var ratings []Rating
	for {
		snap, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("list ratings: %w", err)
		}
		ratings = append(ratings, RatingFromData(snap.Data()))
	}
	return ratings, nil
}

func (s *FirestoreStore) SubmitStep1(ctx context.Context, r Rating) error {
	ref := s.ratingRef(r.ItemID, r.ExpertID)
	if ref == nil {
		return ErrNotFound
	}
	return s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		snap, err := tx.Get(ref)
		if err != nil && !isNotFound(err) {
			return err
		}
		if snap != nil && snap.Exists() {
			existing := RatingFromData(snap.Data())
			if existing.Step1Done() {
				return ErrStep1Locked
			}
		}
		return tx.Set(ref, r.Step1Data())
	})
}

func (s *FirestoreStore) SubmitStep2(ctx context.Context, itemID, expertID string, st Step2, now time.Time) error {
	ref := s.ratingRef(itemID, expertID)
	if ref == nil {
		return ErrStep1Required
	}
	return s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		snap, err := tx.Get(ref)
		if isNotFound(err) {
			return ErrStep1Required
		}
		if err != nil {
			return err
		}
		existing := RatingFromData(snap.Data())
		if !existing.Step1Done() {
			return ErrStep1Required
		}
		fields := step2Update(existing, st, now)
		updates := make([]firestore.Update, 0, len(fields))
		for path, value := range fields {
			updates = append(updates, firestore.Update{Path: path, Value: value})
		}
		return tx.Update(ref, updates)
	})
}
