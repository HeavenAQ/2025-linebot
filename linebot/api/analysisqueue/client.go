// Package analysisqueue publishes small, durable HTTP tasks, never video bytes.
package analysisqueue

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/oauth2/google"
)

type Client struct {
	http                             *http.Client
	queue, workerURL, serviceAccount string
}

// Scheduled service-level capacity does not create or replace a revision.
func (c *Client) SetGPUCapacity(ctx context.Context, project string, minimum int) error {
	if minimum < 0 || minimum > 1 || strings.ContainsAny(project, "/?# ") {
		return fmt.Errorf("invalid capacity")
	}
	body := fmt.Sprintf(`{"scaling":{"minInstanceCount":%d}}`, minimum)
	uri := "https://run.googleapis.com/v2/projects/" + project + "/locations/asia-southeast1/services/badminton-analysis-ai?updateMask=scaling.minInstanceCount"
	req, err := http.NewRequestWithContext(ctx, "PATCH", uri, strings.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("GPU capacity update: HTTP %d", response.StatusCode)
	}
	return nil
}

func New(ctx context.Context, queue, workerURL, serviceAccount string) (*Client, error) {
	parsed, err := url.Parse(workerURL)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || serviceAccount == "" || !strings.HasPrefix(queue, "projects/") {
		return nil, fmt.Errorf("invalid analysis queue configuration")
	}
	h, err := google.DefaultClient(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, err
	}
	h.Timeout = 15 * time.Second
	return &Client{h, queue, strings.TrimRight(workerURL, "/"), serviceAccount}, nil
}

func (c *Client) Enqueue(ctx context.Context, jobID string) error {
	if strings.ContainsAny(jobID, "/?# ") || jobID == "" {
		return fmt.Errorf("invalid job id")
	}
	payload, _ := json.Marshal(map[string]string{"job_id": jobID})
	body, _ := json.Marshal(map[string]any{"task": map[string]any{
		"name":             c.queue + "/tasks/" + jobID,
		"dispatchDeadline": "1200s",
		"httpRequest": map[string]any{
			"httpMethod": "POST", "url": c.workerURL + "/internal/analysis/task",
			"headers":   map[string]string{"Content-Type": "application/json"},
			"body":      base64.StdEncoding.EncodeToString(payload),
			"oidcToken": map[string]string{"serviceAccountEmail": c.serviceAccount, "audience": c.workerURL},
		},
	}})
	req, err := http.NewRequestWithContext(ctx, "POST", "https://cloudtasks.googleapis.com/v2/"+c.queue+"/tasks", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	// Deterministic names make repeated LINE webhooks and outbox retries safe.
	if response.StatusCode == http.StatusConflict || response.StatusCode/100 == 2 {
		return nil
	}
	return fmt.Errorf("enqueue analysis: HTTP %d", response.StatusCode)
}
