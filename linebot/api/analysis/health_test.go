package analysis

import (
	"context"
	analysisv1 "github.com/HeavenAQ/nstc-linebot-2025/api/analysis/v1"
	"github.com/HeavenAQ/nstc-linebot-2025/api/obs"
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

type metadataClient struct {
	analysisv1.BadmintonAnalysisClient
	md metadata.MD
}

func (m *metadataClient) Health(ctx context.Context, _ *analysisv1.HealthRequest, _ ...grpc.CallOption) (*analysisv1.HealthResponse, error) {
	m.md, _ = metadata.FromOutgoingContext(ctx)
	return &analysisv1.HealthResponse{Status: "serving"}, nil
}

// The analysis service logs under the caller's request ID and trace, so they
// must travel with every call.
func TestCallsForwardRequestIDAndTrace(t *testing.T) {
	rpc := &metadataClient{}
	client := &Client{service: rpc, apiKey: "test-key"}
	ctx := obs.WithRequest(context.Background(), obs.FromHeaders("105445aa7843bc8bf206b12000100000/42;o=1", "job-123"))

	require.NoError(t, client.Health(ctx))
	require.Equal(t, []string{"job-123"}, rpc.md.Get("x-request-id"))
	require.Equal(t, []string{"105445aa7843bc8bf206b12000100000/42;o=1"}, rpc.md.Get("x-cloud-trace-context"))
	require.Equal(t, []string{"test-key"}, rpc.md.Get("x-api-key"))

	require.NoError(t, client.Health(context.Background()))
	require.Empty(t, rpc.md.Get("x-request-id"), "no correlation IDs are invented without a request")
}
