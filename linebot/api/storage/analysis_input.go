package storage

import (
	"context"
	"fmt"
	"io"
	"strings"
)

func (c *BucketClient) ReadAnalysisInput(ctx context.Context, object string) ([]byte, error) {
	if !strings.HasPrefix(object, "analyses/input/") || strings.Contains(object, "..") {
		return nil, fmt.Errorf("invalid analysis input")
	}
	reader, err := c.client.Bucket(c.bucketName).Object(object).NewReader(ctx)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	const maximum = 150 * 1024 * 1024
	data, err := io.ReadAll(io.LimitReader(reader, maximum+1))
	if len(data) > maximum {
		return nil, fmt.Errorf("video too large")
	}
	return data, err
}
