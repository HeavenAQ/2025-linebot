package storage

import (
    "context"
    "io"

    gcs "cloud.google.com/go/storage"
)

// StorageClient abstracts the subset of Cloud Storage client used by BucketClient.
// This enables unit testing with fakes without hitting GCP.
type StorageClient interface {
    Bucket(name string) BucketHandle
    Close() error
}

type BucketHandle interface {
    Object(name string) ObjectHandle
}

type ObjectHandle interface {
    NewWriter(ctx context.Context) ObjectWriter
    Delete(ctx context.Context) error
    ObjectName() string
}

type ObjectWriter interface {
    io.WriteCloser
    SetContentType(ct string)
}

// gcsClient is the production implementation backed by cloud.google.com/go/storage.
type gcsClient struct{ *gcs.Client }

func (c *gcsClient) Bucket(name string) BucketHandle { return &gcsBucket{c.Client.Bucket(name)} }
func (c *gcsClient) Close() error                     { return c.Client.Close() }

type gcsBucket struct{ *gcs.BucketHandle }

func (b *gcsBucket) Object(name string) ObjectHandle { return &gcsObject{b.BucketHandle.Object(name)} }

type gcsObject struct{ *gcs.ObjectHandle }

func (o *gcsObject) NewWriter(ctx context.Context) ObjectWriter { return &gcsWriter{o.ObjectHandle.NewWriter(ctx)} }
func (o *gcsObject) Delete(ctx context.Context) error           { return o.ObjectHandle.Delete(ctx) }
func (o *gcsObject) ObjectName() string                         { return o.ObjectHandle.ObjectName() }

type gcsWriter struct{ *gcs.Writer }

func (w *gcsWriter) SetContentType(ct string) { w.Writer.ContentType = ct }

