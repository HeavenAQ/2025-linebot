package analysis

import (
	"context"
	analysisv1 "github.com/HeavenAQ/nstc-linebot-2025/api/analysis/v1"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"testing"
)

type healthClient struct {
	analysisv1.BadmintonAnalysisClient
	key string
}

func (h *healthClient) Health(ctx context.Context, _ *analysisv1.HealthRequest, _ ...grpc.CallOption) (*analysisv1.HealthResponse, error) {
	md, _ := metadata.FromOutgoingContext(ctx)
	h.key = md.Get("x-api-key")[0]
	return &analysisv1.HealthResponse{Status: "serving"}, nil
}

func TestHealthAndWarmupAttachAPIKey(t *testing.T) {
	rpc := &healthClient{}
	client := &Client{service: rpc, apiKey: "test-key"}
	require.NoError(t, client.Health(context.Background()))
	require.Equal(t, "test-key", rpc.key)
	rpc.key = ""
	require.NoError(t, client.Warmup(context.Background()))
	require.Equal(t, "test-key", rpc.key)
}
