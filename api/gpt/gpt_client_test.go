package gpt_test

import (
    "testing"

    "github.com/stretchr/testify/require"
)

// TestCreateThread verifies that a thread can be successfully created
func TestCreateThread(t *testing.T) {
    if !runIntegration {
        t.Skip("Skipping GPT integration test")
    }
    t.Parallel()
    // Create a new thread
    thread, err := gptClient.CreateThread()
    require.NoError(t, err, "Expected no error when creating thread")
    require.NotNil(t, thread, "Expected thread to be created")
}

// TestAddMessageToThread verifies that a message can be added to a thread
func TestAddMessageToThread(t *testing.T) {
    if !runIntegration {
        t.Skip("Skipping GPT integration test")
    }
    t.Parallel()
    // Create a new thread
    thread, err := gptClient.CreateThread()
    require.NoError(t, err, "Expected no error when creating thread")

    // Add a message to the thread and get assistant's reply
    reply, err := gptClient.AddMessageToThread(thread.ID, "Hello, this is a test message.")
    require.NoError(t, err, "Expected no error when adding message to thread")
    require.NotEmpty(t, reply, "Expected assistant to reply with text")
}

// TestGetAssistantResponse verifies that the assistant's response can be retrieved from the thread
func TestGetAssistantResponse(t *testing.T) {
    if !runIntegration {
        t.Skip("Skipping GPT integration test")
    }
    t.Parallel()
    // Create a new thread
    thread, err := gptClient.CreateThread()
    require.NoError(t, err, "Expected no error when creating thread")

    // Add a message to the thread and verify assistant's response directly from Responses API
    response, err := gptClient.AddMessageToThread(thread.ID, "What can you do?")
    require.NoError(t, err, "Expected no error when retrieving assistant response")
    require.NotEmpty(t, response, "Expected a non-empty response from the assistant")
    t.Log("Assistant's response:", response)
}
