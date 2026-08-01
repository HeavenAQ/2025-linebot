package gpt

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/conversations"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

// Flow for sending requests using the Responses API:
// 1. Create a conversation once and store its ID.
// 2. For each user message, call Responses.New with the conversation ID and prompt.
// 3. Read the generated text directly from the returned Response.

type Client struct {
	Ctx          *context.Context
	Client       *openai.Client
	PromptID     string
	RewriteModel string
}

func NewGPTClient(apiKey, promptID, rewriteModel string) *Client {
	ctx := context.Background()
	if rewriteModel == "" {
		rewriteModel = "gpt-5.6-terra"
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	return &Client{
		Ctx:          &ctx,
		Client:       &client,
		PromptID:     promptID,
		RewriteModel: rewriteModel,
	}
}

type HistoryMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

func (client *Client) RewriteQuery(history []HistoryMessage, query string) (string, error) {
	if len(history) == 0 {
		return query, nil
	}
	if len(history) > 12 {
		history = history[len(history)-12:]
	}
	payload, err := json.Marshal(struct {
		History []HistoryMessage `json:"history"`
		Query   string           `json:"query"`
	}{History: history, Query: query})
	if err != nil {
		return "", fmt.Errorf("marshal query rewrite context: %w", err)
	}
	req := responses.ResponseNewParams{
		Model:        client.RewriteModel,
		Instructions: param.Opt[string]{Value: "Rewrite the latest user query as one standalone query using only necessary context from the conversation history. Preserve the user's language and intent. Resolve pronouns and omitted badminton skill references. Do not answer the query, add advice, or mention the history. Return only the rewritten query."},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{Value: string(payload)},
		},
		MaxOutputTokens: param.Opt[int64]{Value: 300},
		Store:           param.Opt[bool]{Value: false},
	}
	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("rewrite query: %w", err)
	}
	rewritten := resp.OutputText()
	if rewritten == "" {
		return "", fmt.Errorf("query rewrite returned empty output")
	}
	return rewritten, nil
}

func (client *Client) CreateConversation() (*conversations.Conversation, error) {
	conversationReq := conversations.ConversationNewParams{
		Items:    []responses.ResponseInputItemUnionParam{},
		Metadata: shared.Metadata{},
	}

	conversation, err := client.Client.Conversations.New(*client.Ctx, conversationReq)
	if err != nil {
		return nil, fmt.Errorf("error creating conversation: %w", err)
	}
	return conversation, nil
}

func (client *Client) RetrieveConversation(conversationID string) (*conversations.Conversation, error) {
	conversation, err := client.Client.Conversations.Get(*client.Ctx, conversationID)
	if err != nil {
		return nil, fmt.Errorf("error retrieving conversation: %w", err)
	}
	return conversation, nil
}

// AddMessageToConversation sends a user message via Responses API
// and returns the assistant's generated text output.
func (client *Client) AddMessageToConversation(conversationID, message string) (string, error) {
	req := responses.ResponseNewParams{
		Prompt: responses.ResponsePromptParam{
			ID: client.PromptID,
		},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{
				Value: message,
			},
		},
		Conversation: responses.ResponseNewParamsConversationUnion{
			OfString: param.Opt[string]{
				Value: conversationID,
			},
		},
	}

	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("error creating response: %w", err)
	}

	// Extract the assistant's text output
	output := resp.OutputText()
	if output == "" {
		return "", fmt.Errorf("no assistant text output available")
	}
	return output, nil
}

// Summarize takes arbitrary text and returns a short summary using the configured prompt.
func (client *Client) Summarize(content string) (string, error) {
	req := responses.ResponseNewParams{
		Prompt: responses.ResponsePromptParam{
			ID: client.PromptID,
		},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{
				Value: "Summarize this learning summary under 100 words: " + content,
			},
		},
	}

	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("error creating summary response: %w", err)
	}

	output := resp.OutputText()
	if output == "" {
		return "", fmt.Errorf("no assistant text output available")
	}
	return output, nil
}
