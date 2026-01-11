package db

import (
    "context"
    "time"

    "cloud.google.com/go/firestore"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// ChatMessage represents a single chat turn message
type ChatMessage struct {
    Role           string    `json:"role" firestore:"role"`                         // "user" or "assistant"
    Text           string    `json:"text" firestore:"text"`                         // message text
    Skill          string    `json:"skill" firestore:"skill"`                       // skill context, e.g., serve/smash/clear
    ConversationID string    `json:"conversation_id" firestore:"conversation_id"`   // optional conversation id
    Timestamp      time.Time `json:"timestamp" firestore:"timestamp"`               // server timestamp when stored
}

// ChatHistory document stored under collection "chat_history" with doc ID = userID
type ChatHistory struct {
    UserID   string        `json:"user_id" firestore:"user_id"`
    Messages []ChatMessage `json:"messages" firestore:"messages"`
}

// AppendChatExchange appends a user/assistant message pair to the user's chat history
func (client *FirestoreClient) AppendChatExchange(userID, skill, conversationID, userText, assistantText string) error {
    ctx := *client.Ctx
    docRef := client.ChatHistory.Doc(userID)

    return client.Client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
        var history ChatHistory
        snap, err := tx.Get(docRef)
        if err != nil {
            // If not found, start a new history document
            if status.Code(err) == codes.NotFound {
                history = ChatHistory{UserID: userID, Messages: []ChatMessage{}}
            } else {
                return err
            }
        } else {
            // If doc exists, decode it
            if err := snap.DataTo(&history); err != nil {
                // If decode fails, start fresh but keep userID
                history = ChatHistory{UserID: userID, Messages: []ChatMessage{}}
            }
        }

        now := time.Now().UTC()
        history.Messages = append(history.Messages,
            ChatMessage{Role: "user", Text: userText, Skill: skill, ConversationID: conversationID, Timestamp: now},
            ChatMessage{Role: "assistant", Text: assistantText, Skill: skill, ConversationID: conversationID, Timestamp: now},
        )

        return tx.Set(docRef, history)
    })
}

// GetChatHistory returns the full chat history for a user.
func (client *FirestoreClient) GetChatHistory(userID string) (*ChatHistory, error) {
    ctx := *client.Ctx
    docRef := client.ChatHistory.Doc(userID)
    snap, err := docRef.Get(ctx)
    if err != nil {
        if status.Code(err) == codes.NotFound {
            // Return empty history if none exists
            return &ChatHistory{UserID: userID, Messages: []ChatMessage{}}, nil
        }
        return nil, err
    }
    var history ChatHistory
    if err := snap.DataTo(&history); err != nil {
        return nil, err
    }
    return &history, nil
}
