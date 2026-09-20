package storage

import (
	"context"
	"errors"
	"io"

	gcs "cloud.google.com/go/storage"
	"google.golang.org/api/iterator"
)

// StorageClient abstracts the subset of Cloud Storage client used by BucketClient.
// This enables unit testing with fakes without hitting GCP.
type StorageClient interface {
	Bucket(name string) BucketHandle
	Close() error
}

type BucketHandle interface {
	Object(name string) ObjectHandle
	// ObjectNames lists the objects under a prefix. The demonstration videos
	// are found this way rather than named in code, so adding an expert is an
	// upload rather than a deploy.
	ObjectNames(ctx context.Context, prefix string) ([]string, error)
	// SignedURL mints a time-limited read URL. It is on the interface so the
	// playback path can be tested without reaching GCP.
	SignedURL(object string, opts *gcs.SignedURLOptions) (string, error)
}

type ObjectHandle interface {
	NewReader(ctx context.Context) (io.ReadCloser, error)
	Attrs(ctx context.Context) (*gcs.ObjectAttrs, error)
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
func (c *gcsClient) Close() error                    { return c.Client.Close() }

type gcsBucket struct{ *gcs.BucketHandle }

func (b *gcsBucket) Object(name string) ObjectHandle { return &gcsObject{b.BucketHandle.Object(name)} }

func (b *gcsBucket) ObjectNames(ctx context.Context, prefix string) ([]string, error) {
	var names []string
	it := b.BucketHandle.Objects(ctx, &gcs.Query{Prefix: prefix})
	for {
		attrs, err := it.Next()
		if errors.Is(err, iterator.Done) {
			return names, nil
		}
		if err != nil {
			return nil, err
		}
		names = append(names, attrs.Name)
	}
}

func (b *gcsBucket) SignedURL(object string, opts *gcs.SignedURLOptions) (string, error) {
	return b.BucketHandle.SignedURL(object, opts)
}

type gcsObject struct{ *gcs.ObjectHandle }

func (o *gcsObject) NewReader(ctx context.Context) (io.ReadCloser, error) {
	return o.ObjectHandle.NewReader(ctx)
}

func (o *gcsObject) NewWriter(ctx context.Context) ObjectWriter {
	return &gcsWriter{o.ObjectHandle.NewWriter(ctx)}
}
func (o *gcsObject) Delete(ctx context.Context) error { return o.ObjectHandle.Delete(ctx) }
func (o *gcsObject) ObjectName() string               { return o.ObjectHandle.ObjectName() }

type gcsWriter struct{ *gcs.Writer }

func (w *gcsWriter) SetContentType(ct string) { w.Writer.ContentType = ct }
