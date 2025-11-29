package storage

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/option"
)

type BucketClient struct {
	client     *storage.Client
	bucketName string
	ctx        context.Context
}

type UserFolders struct {
	UserID   string
	UserName string
	RootPath string
}

func NewBucketClient(credentials []byte, bucketName string) (*BucketClient, error) {
	ctx := context.Background()

	// init google cloud storage client
	client, err := storage.NewClient(ctx, option.WithCredentialsJSON(credentials))
	if err != nil {
		return nil, err
	}

	return &BucketClient{
		client:     client,
		bucketName: bucketName,
		ctx:        ctx,
	}, nil
}

// Close closes the bucket client
func (c *BucketClient) Close() error {
	return c.client.Close()
}

// CreateUserFolders creates a user folder structure in the bucket
// Note: GCS doesn't have actual folders, so we just return the user's root path
func (c *BucketClient) CreateUserFolders(userID string, userName string) (*UserFolders, error) {
	// In GCS, folders are virtual - they're just part of the object path
	// We don't need to explicitly create them, but we can create a placeholder object
	// to ensure the "folder" appears in the UI

	rootPath := fmt.Sprintf("%s/", userID)

	// Create a placeholder object to establish the user's root folder
	placeholderPath := fmt.Sprintf("%s.folder_placeholder", rootPath)
	bucket := c.client.Bucket(c.bucketName)
	obj := bucket.Object(placeholderPath)

	writer := obj.NewWriter(c.ctx)
	if _, err := writer.Write([]byte("")); err != nil {
		writer.Close()
		return nil, fmt.Errorf("failed to create user folder placeholder: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer for user folder: %w", err)
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

// UploadVideo uploads a video to the bucket
func (c *BucketClient) UploadVideo(fileInfo *FileInfo) (*UploadedFile, error) {
	bucket := c.client.Bucket(c.bucketName)
	obj := bucket.Object(fileInfo.Bucket.VideoPath)

	writer := obj.NewWriter(c.ctx)
	writer.ContentType = "video/mp4"

	blob := bytes.NewReader(fileInfo.Local.VideoBlob)
	if _, err := io.Copy(writer, blob); err != nil {
		writer.Close()
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

// UploadThumbnail uploads a thumbnail to the bucket
func (c *BucketClient) UploadThumbnail(fileInfo *FileInfo) (*UploadedFile, error) {
	thumbnailData, err := os.ReadFile(fileInfo.Local.ThumbnailPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read thumbnail file: %w", err)
	}

	bucket := c.client.Bucket(c.bucketName)
	obj := bucket.Object(fileInfo.Bucket.ThumbnailPath)

	writer := obj.NewWriter(c.ctx)
	writer.ContentType = "image/jpeg"

	if _, err := writer.Write(thumbnailData); err != nil {
		writer.Close()
		return nil, fmt.Errorf("failed to upload thumbnail: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close thumbnail writer: %w", err)
	}

	return &UploadedFile{
		Name: obj.ObjectName(),
		Path: fileInfo.Bucket.ThumbnailPath,
	}, nil
}

// DeleteFile deletes a file from the bucket
func (c *BucketClient) DeleteFile(filePath string) error {
	bucket := c.client.Bucket(c.bucketName)
	obj := bucket.Object(filePath)

	if err := obj.Delete(c.ctx); err != nil {
		return fmt.Errorf("failed to delete file: %w", err)
	}

	return nil
}

// GetPublicURL returns the public URL for a file in the bucket
// Note: This assumes the bucket is publicly readable. For private buckets,
// you would need to generate a signed URL instead.
func (c *BucketClient) GetPublicURL(filePath string) string {
	return fmt.Sprintf("https://storage.googleapis.com/%s/%s", c.bucketName, filePath)
}

// GetSignedURL generates a signed URL for private file access
func (c *BucketClient) GetSignedURL(filePath string, expirationMinutes int) (string, error) {
	// Generate a signed URL valid for the specified duration
	opts := &storage.SignedURLOptions{
		Scheme:  storage.SigningSchemeV4,
		Method:  "GET",
		Expires: time.Now().Add(time.Duration(expirationMinutes) * time.Minute),
	}

	url, err := c.client.Bucket(c.bucketName).SignedURL(filePath, opts)
	if err != nil {
		return "", fmt.Errorf("failed to generate signed URL: %w", err)
	}

	return url, nil
}
