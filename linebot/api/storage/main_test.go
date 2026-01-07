package storage

import (
    "bytes"
    "context"
    "errors"
    "os"
    "path/filepath"
    "testing"

    "github.com/stretchr/testify/require"
)

// --- Fakes for StorageClient interfaces ---

type fakeClient struct{ buckets map[string]*fakeBucket }

func newFakeClient() *fakeClient { return &fakeClient{buckets: map[string]*fakeBucket{}} }

func (c *fakeClient) Bucket(name string) BucketHandle {
    b, ok := c.buckets[name]
    if !ok {
        b = &fakeBucket{name: name, objects: map[string]*fakeObject{}}
        c.buckets[name] = b
    }
    return b
}
func (c *fakeClient) Close() error { return nil }

type fakeBucket struct {
    name    string
    objects map[string]*fakeObject
}

func (b *fakeBucket) Object(name string) ObjectHandle {
    return &fakeObjectRef{bucket: b, name: name}
}

type fakeObject struct {
    data        []byte
    contentType string
}

type fakeObjectRef struct {
    bucket *fakeBucket
    name   string
}

func (r *fakeObjectRef) ensure() *fakeObject {
    obj, ok := r.bucket.objects[r.name]
    if !ok {
        obj = &fakeObject{}
        r.bucket.objects[r.name] = obj
    }
    return obj
}

func (r *fakeObjectRef) NewWriter(ctx context.Context) ObjectWriter {
    return &fakeWriter{ref: r}
}
func (r *fakeObjectRef) Delete(ctx context.Context) error {
    if _, ok := r.bucket.objects[r.name]; !ok {
        return errors.New("object not found")
    }
    delete(r.bucket.objects, r.name)
    return nil
}
func (r *fakeObjectRef) ObjectName() string { return r.name }

type fakeWriter struct {
    ref *fakeObjectRef
    buf bytes.Buffer
    ct  string
}

func (w *fakeWriter) Write(p []byte) (int, error) { return w.buf.Write(p) }
func (w *fakeWriter) Close() error {
    obj := w.ref.ensure()
    obj.data = append([]byte(nil), w.buf.Bytes()...)
    obj.contentType = w.ct
    return nil
}
func (w *fakeWriter) SetContentType(ct string) { w.ct = ct }

// --- Tests ---

func TestCreateUserFolders(t *testing.T) {
    fake := newFakeClient()
    bc := NewBucketClientWithClient(context.Background(), fake, "test-bucket")

    uf, err := bc.CreateUserFolders("user123", "alice")
    require.NoError(t, err)
    require.Equal(t, "user123/", uf.RootPath)

    // verify placeholder exists
    b := fake.buckets["test-bucket"]
    obj, ok := b.objects["user123/.folder_placeholder"]
    require.True(t, ok)
    require.Empty(t, obj.data)
}

func TestUploadVideo(t *testing.T) {
    fake := newFakeClient()
    bc := NewBucketClientWithClient(context.Background(), fake, "test-bucket")

    fi := &FileInfo{}
    fi.Bucket.VideoPath = "user123/videos/v1.mp4"
    fi.Local.VideoBlob = []byte{1, 2, 3, 4, 5}

    uploaded, err := bc.UploadVideo(fi)
    require.NoError(t, err)
    require.Equal(t, fi.Bucket.VideoPath, uploaded.Path)
    require.Equal(t, "user123/videos/v1.mp4", uploaded.Name)

    obj := fake.buckets["test-bucket"].objects[fi.Bucket.VideoPath]
    require.Equal(t, []byte{1, 2, 3, 4, 5}, obj.data)
    require.Equal(t, "video/mp4", obj.contentType)
}

func TestUploadThumbnail(t *testing.T) {
    fake := newFakeClient()
    bc := NewBucketClientWithClient(context.Background(), fake, "test-bucket")

    thumbPath := filepath.Join("test_files", "test_thumbnail.jpg")
    fi := &FileInfo{}
    fi.Bucket.ThumbnailPath = "user123/thumbnails/t1.jpg"
    fi.Local.ThumbnailPath = thumbPath

    uploaded, err := bc.UploadThumbnail(fi)
    require.NoError(t, err)
    require.Equal(t, fi.Bucket.ThumbnailPath, uploaded.Path)

    // Verify bytes were written
    disk, _ := os.ReadFile(thumbPath)
    obj := fake.buckets["test-bucket"].objects[fi.Bucket.ThumbnailPath]
    require.Equal(t, len(disk), len(obj.data))
    require.Equal(t, "image/jpeg", obj.contentType)
}

func TestDeleteFile(t *testing.T) {
    fake := newFakeClient()
    bc := NewBucketClientWithClient(context.Background(), fake, "test-bucket")

    // Seed an object via upload
    fi := &FileInfo{}
    fi.Bucket.VideoPath = "user123/videos/v2.mp4"
    fi.Local.VideoBlob = []byte{9, 9, 9}
    _, err := bc.UploadVideo(fi)
    require.NoError(t, err)

    // Delete it
    err = bc.DeleteFile(fi.Bucket.VideoPath)
    require.NoError(t, err)

    // Verify it no longer exists
    _, ok := fake.buckets["test-bucket"].objects[fi.Bucket.VideoPath]
    require.False(t, ok)

    // Delete again should error
    err = bc.DeleteFile(fi.Bucket.VideoPath)
    require.Error(t, err)
}
