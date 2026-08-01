package analysis

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAnalyzeVideoRejectsEmptyInputBeforeStartingStream(t *testing.T) {
	client := &Client{}
	result, err := client.AnalyzeVideo(
		context.Background(),
		"request-id",
		"user-id",
		"video.mp4",
		"serve",
		"right",
		nil,
	)

	require.Nil(t, result)
	require.EqualError(t, err, "video is empty")
}
