package gpt

import (
	"context"
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
	Ctx      *context.Context
	Client   *openai.Client
	PromptID string
}

func NewGPTClient(apiKey, promptID string) *Client {
	ctx := context.Background()
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	return &Client{
		Ctx:      &ctx,
		Client:   &client,
		PromptID: promptID,
	}
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
