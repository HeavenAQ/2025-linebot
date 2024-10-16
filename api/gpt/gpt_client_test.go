package gpt_test

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCreateThread verifies that a thread can be successfully created
func TestCreateThread(t *testing.T) {
	t.Parallel()
	// Create a new thread
	thread, err := gptClient.CreateThread()
	require.NoError(t, err, "Expected no error when creating thread")
	require.NotNil(t, thread, "Expected thread to be created")
}

// TestAddMessageToThread verifies that a message can be added to a thread
func TestAddMessageToThread(t *testing.T) {
	t.Parallel()
	// Create a new thread
	thread, err := gptClient.CreateThread()
	require.NoError(t, err, "Expected no error when creating thread")

	// Add a message to the thread
	err = gptClient.AddMessageToThread(thread.ID, "Hello, this is a test message.")
	require.NoError(t, err, "Expected no error when adding message to thread")
}

// TestRunThread verifies that a thread can be run and processed by the assistant
func TestRunThread(t *testing.T) {
	t.Parallel()
	// Create a new thread
	thread, err := gptClient.CreateThread()
	require.NoError(t, err, "Expected no error when creating thread")

	// Add a message to the thread
	err = gptClient.AddMessageToThread(thread.ID, "Please process this message.")
	require.NoError(t, err, "Expected no error when adding message to thread")

	// Run the thread and get a run ID
	runID, err := gptClient.RunThread(thread.ID)
	require.NoError(t, err, "Expected no error when running thread")
	require.NotEmpty(t, runID, "Expected a valid run ID")
}

// TestGetAssistantResponse verifies that the assistant's response can be retrieved from the thread
func TestGetAssistantResponse(t *testing.T) {
	t.Parallel()
	// Create a new thread
	thread, err := gptClient.CreateThread()
	require.NoError(t, err, "Expected no error when creating thread")

	// Add a message to the thread
	err = gptClient.AddMessageToThread(thread.ID, "What can you do?")
	require.NoError(t, err, "Expected no error when adding message to thread")

	// Run the thread and get a run ID
	runID, err := gptClient.RunThread(thread.ID)
	require.NoError(t, err, "Expected no error when running thread")
	require.NotEmpty(t, runID, "Expected a valid run ID")

	// Retrieve the assistant's response
	response, err := gptClient.GetAssistantResponse(thread.ID, runID)
	require.NoError(t, err, "Expected no error when retrieving assistant response")
	require.NotEmpty(t, response, "Expected a non-empty response from the assistant")
	t.Log("Assistant's response:", response)
}
