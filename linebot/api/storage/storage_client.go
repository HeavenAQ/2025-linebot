// Package storage implements the basic CRUD functions for the GCP cloud storage
package storage

import (
    "bytes"
    "context"
    "fmt"
    "io"
    "os"

    gcs "cloud.google.com/go/storage"
)

type BucketClient struct {
    client     StorageClient
    bucketName string
    ctx        context.Context
}

type UserFolders struct {
	UserID   string
	UserName string
	RootPath string
}

func NewBucketClient(bucketName string) (*BucketClient, error) {
    ctx := context.Background()

    // init Google Cloud Storage client
    client, err := gcs.NewClient(ctx)
    if err != nil {
        return nil, err
    }

    return &BucketClient{
        &gcsClient{client},
        bucketName,
        ctx,
    }, nil
}

// NewBucketClientWithClient was used for tests; moved to a _test.go helper.

func (c *BucketClient) Close() error {
	return c.client.Close()
}

func (c *BucketClient) CreateUserFolders(userID string, userName string) (*UserFolders, error) {
    rootPath := fmt.Sprintf("%s/", userID)
    // a placeholder for UI display
    placeholderPath := fmt.Sprintf("%s.folder_placeholder", rootPath)
    bucket := c.client.Bucket(c.bucketName)
    obj := bucket.Object(placeholderPath)

    writer := obj.NewWriter(c.ctx)
    if _, err := writer.Write([]byte("")); err != nil {
        _ = writer.Close()
        return nil, fmt.Errorf("failed to create user folder placeholder: %w", err)
    }
    if err := writer.Close(); err != nil {
        return nil, fmt.Errorf("failed to close writer for user folder placeholder: %w", err)
    }

	return &UserFolders{
		UserID:   userID,
		UserName: userName,
		RootPath: rootPath,
	}, nil
}

type FileInfo struct {
	Bucket struct {
		VideoPath     string
		ThumbnailPath string
	}
	Local struct {
		ThumbnailPath string
		VideoBlob     []byte
	}
}

type UploadedFile struct {
	Name string
	Path string
}

func (c *BucketClient) UploadVideo(fileInfo *FileInfo) (*UploadedFile, error) {
    bucket := c.client.Bucket(c.bucketName)
    obj := bucket.Object(fileInfo.Bucket.VideoPath)

    writer := obj.NewWriter(c.ctx)
    writer.SetContentType("video/mp4")

    blobReader := bytes.NewReader(fileInfo.Local.VideoBlob)
    if _, err := io.Copy(writer, blobReader); err != nil {
        _ = writer.Close()
        return nil, fmt.Errorf("failed to upload video: %w", err)
    }

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close video writer: %w", err)
	}

    return &UploadedFile{
        Name: obj.ObjectName(),
        Path: fileInfo.Bucket.VideoPath,
    }, nil
}

func (c *BucketClient) UploadThumbnail(fileInfo *FileInfo) (*UploadedFile, error) {
	// upload video thumbnail to google drive
	thumbnailData, err := os.ReadFile(fileInfo.Local.ThumbnailPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read thumbnail file: %w", err)
	}

    // get bucket and create an object
    bucket := c.client.Bucket(c.bucketName)
    obj := bucket.Object(fileInfo.Bucket.ThumbnailPath)

	// Initiate a writer of the object
    writer := obj.NewWriter(c.ctx)
    writer.SetContentType("image/jpeg")

	// write in-memory media blob to the object
	blobReader := bytes.NewReader(thumbnailData)
    if _, err := io.Copy(writer, blobReader); err != nil {
        _ = writer.Close()
        return nil, fmt.Errorf("failed to copy in-memory thumbnail data to object writer: %w", err)
    }

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close thumbnail writer: %w", err)
	}

	return &UploadedFile{
		Name: obj.ObjectName(),
		Path: fileInfo.Bucket.ThumbnailPath,
	}, nil
}

func (c *BucketClient) DeleteFile(filePath string) error {
    bucket := c.client.Bucket(c.bucketName)
    obj := bucket.Object(filePath)

    if err := obj.Delete(c.ctx); err != nil {
        return fmt.Errorf("failed to delete file %s: %w", filePath, err)
    }
    return nil
}
