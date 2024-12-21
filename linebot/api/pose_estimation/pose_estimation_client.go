package poseestimation

import (
	"bytes"
	"fmt"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/go-resty/resty/v2"
)

// Client represents the configuration for the pose estimation service.
type Client struct {
	Client    *resty.Client
	Username  string
	Password  string
	URL       string
	VideoBlob []byte
}

// ResponseData holds the data returned from the video processing API.

type VideoAnalysisResponse struct {
	ProcessedVideo string                 `json:"processed_video"`
	Grade          commons.GradingOutcome `json:"grade"`
}

// NewClient initializes a new Client with the specified URL, credentials, and video blob.
func NewClient(username, password, url string, videoBlob []byte) *Client {
	client := resty.New()
	client.SetTimeout(120 * time.Second) // Set a 120-second timeout to handle large files
	client.SetRetryCount(3)              // Retry the request up to 3 times in case of failures

	return &Client{
		Client:    client,
		Username:  username,
		Password:  password,
		URL:       url,
		VideoBlob: videoBlob,
	}
}

// uploadVideo uploads the video blob specified in the Client struct and returns a ResponseData struct.
func (c *Client) uploadVideo(skill string, handedness string) (*VideoAnalysisResponse, error) {
	// Prepare the request.
	req := c.Client.R().
		SetBasicAuth(c.Username, c.Password).                              // Set Basic Authentication credentials
		SetFileReader("video", "video.mp4", bytes.NewReader(c.VideoBlob)). // Attach the video blob as a file
		SetFormData(map[string]string{
			"skill":      skill,
			"handedness": handedness,
		}).
		SetResult(&VideoAnalysisResponse{}). // Expect a ResponseData struct
		SetError(&map[string]interface{}{})  // Capture error response if any

	// Send the request.
	resp, err := req.Post(c.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check if the server returned a successful status code.
	if resp.IsError() {
		return nil, fmt.Errorf("server returned error status code: %d, response: %s",
			resp.StatusCode(), resp.String())
	}

	// Extract response data into a ResponseData struct.
	responseData, ok := resp.Result().(*VideoAnalysisResponse)
	if !ok || responseData == nil {
		return nil, fmt.Errorf("failed to parse response data")
	}

	return responseData, nil
}

// ProcessVideo handles the entire video processing workflow: uploading, retrieving the grading score, and returning the processed video as bytes.
func (c *Client) ProcessVideo(skill string, handedness string) (*VideoAnalysisResponse, error) {
	// Upload the video and retrieve response data.
	responseData, err := c.uploadVideo(skill, handedness)
	if err != nil {
		return nil, fmt.Errorf("failed to upload video: %w", err)
	}

	return responseData, nil
}
