package db

import (
	"context"
	"log"

	"cloud.google.com/go/firestore"
	firebase "firebase.google.com/go"
)

type FirestoreClient struct {
	Ctx      *context.Context
	Client   *firestore.Client
	Data     *firestore.CollectionRef
	Sessions *firestore.CollectionRef
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
