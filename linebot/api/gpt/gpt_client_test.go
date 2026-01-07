package gpt_test

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCreateConversation verifies that a conversation can be successfully created
func TestCreateConversation(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	t.Parallel()
    // Create a new conversation
    conv, err := gptClient.CreateConversation()
    require.NoError(t, err, "Expected no error when creating conversation")
    require.NotNil(t, conv, "Expected conversation to be created")
}

// TestAddMessageToConversation verifies that a message can be added to a conversation
func TestAddMessageToConversation(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	t.Parallel()
    // Create a new conversation
    conv, err := gptClient.CreateConversation()
    require.NoError(t, err, "Expected no error when creating conversation")

    // Add a message to the conversation and get assistant's reply
    reply, err := gptClient.AddMessageToConversation(conv.ID, "Hello, this is a test message.")
    require.NoError(t, err, "Expected no error when adding message to conversation")
    require.NotEmpty(t, reply, "Expected assistant to reply with text")
}

// TestGetAssistantResponse verifies that the assistant's response can be retrieved from the conversation
func TestGetAssistantResponse(t *testing.T) {
	if !runIntegration {
		t.Skip("Skipping GPT integration test")
	}
	t.Parallel()
    // Create a new conversation
    conv, err := gptClient.CreateConversation()
    require.NoError(t, err, "Expected no error when creating conversation")

    // Add a message to the conversation and verify assistant's response directly from Responses API
    response, err := gptClient.AddMessageToConversation(conv.ID, "What can you do?")
    require.NoError(t, err, "Expected no error when retrieving assistant response")
    require.NotEmpty(t, response, "Expected a non-empty response from the assistant")
    t.Log("Assistant's response:", response)
}
