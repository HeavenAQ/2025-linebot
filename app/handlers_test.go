package app

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/line/line-bot-sdk-go/v7/linebot"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockLineBot is a mock implementation of LineBotClient for testing
type MockLineBot struct {
	mock.Mock
}

func (m *MockLineBot) ParseRequest(r *http.Request) ([]*linebot.Event, error) {
	args := m.Called(r)
	return args.Get(0).([]*linebot.Event), args.Error(1)
}

func (m *MockLineBot) ReplyMessage(replyToken string, messages ...linebot.SendingMessage) (*linebot.BasicResponse, error) {
	args := m.Called(replyToken, messages)
	return args.Get(0).(*linebot.BasicResponse), args.Error(1)
}

func (b *MockLineBot) ReplyWithTypeError(replyToken string) {
	// Do nothing
}

func TestLineWebhookHandler(t *testing.T) {
	// Create a mock LineBotClient
	mockBot := new(MockLineBot)

	// Create a mock app with the mock LineBotClient
	mockApp := &App{
		LineBot: mockBot,
	}

	// Set up the handler with the mock app
	handler := mockApp.LineWebhookHandler()

	// Create a mock request
	req := httptest.NewRequest(http.MethodPost, "/callback", nil)
	rr := httptest.NewRecorder()

	// Set expectations for the mock LineBotClient
	mockBot.On("ParseRequest", req).Return([]*linebot.Event{}, nil)

	// Call the handler
	handler.ServeHTTP(rr, req)

	// Check the response status code
	require.Equal(t, http.StatusOK, rr.Code)

	// Ensure the expectations were met
	mockBot.AssertExpectations(t)
}
