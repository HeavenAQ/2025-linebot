package db

import (
	"context"
	"strings"

	"cloud.google.com/go/firestore"
	firebase "firebase.google.com/go"
)

type FirestoreClient struct {
	Ctx            *context.Context
	Client         *firestore.Client
	Data           *firestore.CollectionRef
	Sessions       *firestore.CollectionRef
	ChatHistory    *firestore.CollectionRef
	DailySummaries *firestore.CollectionRef
	WeeklyPreviews *firestore.CollectionRef
	// WeeklyReflections holds what learners write in the LIFF review tab.
	WeeklyReflections *firestore.CollectionRef
	// excludedLearners are accounts whose attempts must not reach the class
	// figures: the instructor's and the developer's own test uploads, which are
	// real documents in the same collection as the students'.
	excludedLearners map[string]bool
}

// ExcludeFromClassFigures keeps the listed learners out of every class
// aggregate, standing and best attempt. Their own portfolios are untouched.
func (c *FirestoreClient) ExcludeFromClassFigures(userIDs []string) {
	excluded := make(map[string]bool, len(userIDs))
	for _, id := range userIDs {
		if id = strings.TrimSpace(id); id != "" {
			excluded[id] = true
		}
	}
	c.excludedLearners = excluded
}

// CountsTowardClass reports whether a learner's attempts belong in the class
// figures.
func (c *FirestoreClient) CountsTowardClass(userID string) bool {
	return !c.excludedLearners[userID]
}

func NewFirestoreClient(projectID string, dataCollection string, sessionCollection string) (*FirestoreClient, error) {
	// initialize firebase app
	ctx := context.Background()
	conf := &firebase.Config{ProjectID: projectID}
	app, err := firebase.NewApp(ctx, conf)
	if err != nil {
		return nil, err
	}

	// instantiate firestore client
	client, err := app.Firestore(ctx)
	if err != nil {
		return nil, err
	}

	// return firestore client
	return &FirestoreClient{
		Ctx:               &ctx,
		Client:            client,
		Data:              client.Collection(dataCollection),
		Sessions:          client.Collection(sessionCollection),
		ChatHistory:       client.Collection("chat_history"),
		DailySummaries:    client.Collection("daily_summaries"),
		WeeklyPreviews:    client.Collection("weekly_previews"),
		WeeklyReflections: client.Collection("weekly_reflections"),
	}, nil
}
