package gpt

import (
	"context"
	"errors"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// How do will send request to custom assistant GPT?
// 1. Createa a new thread
// 2. Add message to thread
// 3. Run thread
// 4. List the thread and the latest message will be the response from the assistant

type Client struct {
	Ctx         *context.Context
	Client      *openai.Client
	APIKey      string
	AssistantID string
}

func NewGPTClient(apiKey, assistantID string) *Client {
	ctx := context.Background()
	client := openai.NewClient(apiKey)
	return &Client{
		Ctx:         &ctx,
		Client:      client,
		APIKey:      apiKey,
		AssistantID: assistantID,
	}
}

func (client *Client) CreateThread() (*openai.Thread, error) {
	threadReq := openai.ThreadRequest{
		Messages:      []openai.ThreadMessage{},
		Metadata:      map[string]any{},
		ToolResources: nil,
	}

	thread, err := client.Client.CreateThread(*client.Ctx, threadReq)
	if err != nil {
		return nil, fmt.Errorf("error creating thread: %w", err)
	}
	return &thread, nil
}

func (client *Client) RetrieveThread(threadID string) (*openai.Thread, error) {
	thread, err := client.Client.RetrieveThread(*client.Ctx, threadID)
	if err != nil {
		return nil, fmt.Errorf("error retrieving thread: %w", err)
	}
	return &thread, nil
}

func (client *Client) AddMessageToThread(threadID, message string) error {
	messageReq := openai.MessageRequest{
		Role:     string(openai.ThreadMessageRoleUser),
		Content:  message,
		FileIds:  []string{},
		Metadata: map[string]any{},
	}

	_, err := client.Client.CreateMessage(*client.Ctx, threadID, messageReq)
	if err != nil {
		return fmt.Errorf("error adding message to thread: %w", err)
	}
	return nil
}

var ErrRunStatus = errors.New("error running thread")

func (client *Client) RunThread(threadID string) (string, error) {
	runReq := openai.RunRequest{AssistantID: client.AssistantID}
	run, err := client.Client.CreateRun(*client.Ctx, threadID, runReq)
	if err != nil {
		return "", fmt.Errorf("error running thread: %w", err)
	}

	for run.Status != openai.RunStatusCompleted {
		if run.Status == openai.RunStatusFailed {
			return "", ErrRunStatus
		}

		run, err = client.Client.RetrieveRun(*client.Ctx, threadID, run.ID)
		if err != nil {
			return "", fmt.Errorf("error retrieving run: %w", err)
		}
	}
	return run.ID, nil
}

func (client *Client) GetAssistantResponse(threadID, runID string) (string, error) {
	limit := 1
	order := "desc"
	res, err := client.Client.ListMessage(*client.Ctx, threadID, &limit, &order, nil, nil, &runID)
	if err != nil {
		return "", fmt.Errorf("error getting assistant response: %w", err)
	}

	return res.Messages[0].Content[0].Text.Value, nil
}
