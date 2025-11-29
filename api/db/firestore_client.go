package db

import (
	"context"
	"log"

	"cloud.google.com/go/firestore"
)

type FirestoreClient struct {
	Ctx      *context.Context
	Client   *firestore.Client
	Data     *firestore.CollectionRef
	Sessions *firestore.CollectionRef
}

func NewFirestoreClient(credentials []byte, projectID string, databaseID string, dataCollection string, sessionCollection string) (*FirestoreClient, error) {
	// initialize firebase app
	ctx := context.Background()

	// instantiate firestore client
	client, err := firestore.NewClientWithDatabase(ctx, projectID, databaseID)
	if err != nil {
		log.Fatal("Error initializing firebase database client:", err)
	}

	// return firestore client
	return &FirestoreClient{
		&ctx,
		client,
		client.Collection(dataCollection),
		client.Collection(sessionCollection),
	}, nil
}
