package storage

import "context"

// NewBucketClientWithClient is a test-only helper for constructing a BucketClient
// with a custom StorageClient implementation (e.g., fakes/mocks).
func NewBucketClientWithClient(ctx context.Context, client StorageClient, bucketName string) *BucketClient {
    return &BucketClient{client: client, bucketName: bucketName, ctx: ctx}
}

