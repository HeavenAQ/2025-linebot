package obs

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseTraceHeader(t *testing.T) {
	traceID, spanID, sampled, ok := ParseTraceHeader("105445AA7843BC8BF206B120001000/123;o=1")
	require.False(t, ok, "31 hex characters is not a trace id")

	traceID, spanID, sampled, ok = ParseTraceHeader("105445aa7843bc8bf206b12000100000/2147483647;o=1")
	require.True(t, ok)
	require.Equal(t, "105445aa7843bc8bf206b12000100000", traceID)
	require.Equal(t, "2147483647", spanID)
	require.True(t, sampled)

	traceID, spanID, sampled, ok = ParseTraceHeader("105445aa7843bc8bf206b12000100000")
	require.True(t, ok)
	require.Empty(t, spanID)
	require.False(t, sampled)

	for _, bad := range []string{"", "nonsense", "105445aa7843bc8bf206b12000100000/abc", "105445aa7843bc8bf206b12000100000/1;o=7"} {
		_, _, _, ok := ParseTraceHeader(bad)
		require.False(t, ok, bad)
	}
}

func TestFromHeadersStartsATraceWhenMissing(t *testing.T) {
	request := FromHeaders("", "")
	require.Len(t, request.TraceID, 32)
	require.Equal(t, request.TraceID, request.ID)

	request = FromHeaders("105445aa7843bc8bf206b12000100000/9;o=1", "job-1")
	require.Equal(t, "job-1", request.ID)
	require.Equal(t, "105445aa7843bc8bf206b12000100000/9;o=1", OutgoingTraceHeader(WithRequest(context.Background(), request)))

	// An unprintable client-supplied ID is replaced, not logged verbatim.
	unsafe := FromHeaders("", "bad id\n")
	require.Equal(t, unsafe.TraceID, unsafe.ID)
}

func TestWithRequestIDKeepsTheTrace(t *testing.T) {
	ctx := WithRequest(context.Background(), FromHeaders("105445aa7843bc8bf206b12000100000/9;o=0", "http"))
	ctx = WithRequestID(ctx, "job-7")
	request, ok := RequestFrom(ctx)
	require.True(t, ok)
	require.Equal(t, "job-7", request.ID)
	require.Equal(t, "105445aa7843bc8bf206b12000100000", request.TraceID)
}

func TestEventWritesCloudLoggingFields(t *testing.T) {
	var buffer bytes.Buffer
	SetOutput(&buffer)
	Configure("demo-project")
	t.Cleanup(func() { SetOutput(nil); Configure("") })

	ctx := WithRequest(context.Background(), FromHeaders("105445aa7843bc8bf206b12000100000/9;o=1", "req-1"))
	Event(ctx, Warning, "analysis job finished", map[string]any{"job_id": "abc", "error": context.Canceled})

	var entry map[string]any
	require.NoError(t, json.Unmarshal(buffer.Bytes(), &entry))
	require.Equal(t, "WARNING", entry["severity"])
	require.Equal(t, "analysis job finished", entry["message"])
	require.Equal(t, "req-1", entry["request_id"])
	require.Equal(t, "abc", entry["job_id"])
	require.Equal(t, "context canceled", entry["error"])
	require.Equal(t, "projects/demo-project/traces/105445aa7843bc8bf206b12000100000", entry["logging.googleapis.com/trace"])
	require.Equal(t, "0000000000000009", entry["logging.googleapis.com/spanId"])
}

func TestStdLoggerLinesBecomeStructured(t *testing.T) {
	var buffer bytes.Buffer
	SetOutput(&buffer)
	t.Cleanup(func() { SetOutput(nil) })

	NewStdLogger(Error).Printf("failed to reply: %v", "boom")

	var entry map[string]any
	require.NoError(t, json.Unmarshal(buffer.Bytes(), &entry))
	require.Equal(t, "ERROR", entry["severity"])
	require.Equal(t, "failed to reply: boom", entry["message"])
	location := entry["logging.googleapis.com/sourceLocation"].(map[string]any)
	require.True(t, strings.HasSuffix(location["file"].(string), "_test.go"))
}
