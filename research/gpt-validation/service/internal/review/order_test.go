package review

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func ids(items []Item) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.ItemID
	}
	return out
}

func TestOrderForExpertDeterministic(t *testing.T) {
	var items []Item
	for i := 0; i < 40; i++ {
		items = append(items, Item{ItemID: fmt.Sprintf("serve-%010d", i)})
	}
	first := ids(OrderForExpert("expert_a", items))

	reversed := make([]Item, len(items))
	for i := range items {
		reversed[len(items)-1-i] = items[i]
	}
	if again := ids(OrderForExpert("expert_a", reversed)); !reflect.DeepEqual(first, again) {
		t.Error("order depends on input order")
	}
	if other := ids(OrderForExpert("expert_b", items)); reflect.DeepEqual(first, other) {
		t.Error("two experts got the same order")
	}

	// Matches the documented key: sha256(expert_id + ":" + item_id).
	want := ids(items)
	key := func(id string) string {
		s := sha256.Sum256([]byte("expert_a:" + id))
		return hex.EncodeToString(s[:])
	}
	sort.Slice(want, func(i, j int) bool { return key(want[i]) < key(want[j]) })
	if !reflect.DeepEqual(first, want) {
		t.Error("order does not follow sha256(expert_id:item_id)")
	}
}

func TestItemListOrderAndFiltering(t *testing.T) {
	ineligible := testItem("serve-fallback", 0)
	ineligible.Eligible = false
	ineligible.CoachingSource = "deterministic_fallback"
	staleFlag := testItem("serve-stale", 0)
	staleFlag.CoachingSource = "score_gate" // eligible flag true but definition says no
	otherBatch := testItem("serve-other", 1)
	otherBatch.BatchID = "another-batch"
	env := newTestEnv(t, testItem("serve-a", 1), testItem("serve-b", 2), testItem("smash-c", 0), ineligible, staleFlag, otherBatch)

	rec := env.do("GET", "/api/items", "code-alpha-1234", nil)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	body := rec.Body.String()
	assertNoForbidden(t, body)
	for _, hidden := range []string{"serve-fallback", "serve-stale", "serve-other"} {
		if contains(body, hidden) {
			t.Errorf("list includes %s", hidden)
		}
	}
	var got struct {
		Items []ItemSummary `json:"items"`
	}
	decodeInto(t, rec.Body.Bytes(), &got)
	all, _ := env.store.ListItems(context.Background(), testBatch)
	var eligible []Item
	for _, it := range all {
		if it.ServableTo(testBatch) {
			eligible = append(eligible, it)
		}
	}
	want := ids(OrderForExpert("expert_a", eligible))
	var gotIDs []string
	for _, s := range got.Items {
		gotIDs = append(gotIDs, s.ItemID)
	}
	if !reflect.DeepEqual(gotIDs, want) {
		t.Errorf("list order %v, want %v", gotIDs, want)
	}
}
