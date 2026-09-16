package analysisqueue

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

type transport func(*http.Request) (*http.Response, error)

func (f transport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestTaskContainsOnlyJobIDAndUsesStableNameAndOIDC(t *testing.T) {
	for _, code := range []int{200, 409} {
		c := &Client{queue: "projects/p/locations/r/queues/llm", workerURL: "https://worker.test", serviceAccount: "worker@p.iam.gserviceaccount.com"}
		c.http = &http.Client{Transport: transport(func(r *http.Request) (*http.Response, error) {
			var body map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
			task := body["task"].(map[string]any)
			require.Equal(t, c.queue+"/tasks/stable", task["name"])
			require.Equal(t, "1200s", task["dispatchDeadline"])
			h := task["httpRequest"].(map[string]any)
			require.Equal(t, c.workerURL+"/internal/analysis/task", h["url"])
			decoded, err := base64.StdEncoding.DecodeString(h["body"].(string))
			require.NoError(t, err)
			require.JSONEq(t, `{"job_id":"stable"}`, string(decoded))
			require.Equal(t, c.workerURL, h["oidcToken"].(map[string]any)["audience"])
			return &http.Response{StatusCode: code, Body: io.NopCloser(strings.NewReader("{}"))}, nil
		})}
		require.NoError(t, c.Enqueue(context.Background(), "stable"))
	}
}
