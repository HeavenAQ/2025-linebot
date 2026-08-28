package db

import (
	"context"

	"cloud.google.com/go/firestore"
)

type FirestoreClient struct {
	Ctx      *context.Context
	Client   *firestore.Client
	Data     *firestore.CollectionRef
	Sessions *firestore.CollectionRef
	// WeeklyReflections holds what learners write in the LIFF review tab.
	WeeklyReflections *firestore.CollectionRef
}

// DefaultDatabase is the database a project gets without asking for one. An
// empty FIREBASE_DATABASE_ID means it, so a deployment that predates the
// setting keeps talking to the same data.
const DefaultDatabase = "(default)"

// NewFirestoreClient connects to one named database and takes its collections
// from there.
//
// The database, not the collection names, is what separates deployments that
// share a GCP project. Collection names alone cannot do it: WeeklyReflections
// is fixed below, so two deployments pointed at the same database would write
// every learner's reflection into one collection however their other
// collections were named.
func NewFirestoreClient(projectID string, databaseID string, dataCollection string, sessionCollection string) (*FirestoreClient, error) {
	ctx := context.Background()
	if databaseID == "" {
		databaseID = DefaultDatabase
	}

	client, err := firestore.NewClientWithDatabase(ctx, projectID, databaseID)
	if err != nil {
		return nil, err
	}

	// return firestore client
	return &FirestoreClient{
		Ctx:               &ctx,
		Client:            client,
		Data:              client.Collection(dataCollection),
		Sessions:          client.Collection(sessionCollection),
		WeeklyReflections: client.Collection("weekly_reflections"),
	}, nil
}
