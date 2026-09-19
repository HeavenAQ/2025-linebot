package app

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestRegistrationBlocksActionsUntilSaved(t *testing.T) {
	registered := func() *db.UserData {
		return &db.UserData{ID: "U1", RealName: "王小明",
			ExperimentNumber: "01", RegistrationVersion: 1, RegistrationCompletedAt: time.Now()}
	}
	for _, tc := range []struct {
		name        string
		kind        linebot.EventType
		message     linebot.Message
		existing    bool
		lookupError bool
		links       int
		allowed     bool
		reply       string
	}{
		{"follow", linebot.EventTypeFollow, nil, false, false, 1, false, ""},
		{"video", linebot.EventTypeMessage, &linebot.VideoMessage{}, false, false, 1, false, ""},
		{"postback", linebot.EventTypePostback, nil, false, false, 1, false, ""},
		{"rich menu", linebot.EventTypeMessage, linebot.NewTextMessage("動作分析"), false, false, 1, false, ""},
		// Details typed into the chat are no longer a registration; the learner
		// is sent to the form like everyone else.
		{"details in chat", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), false, false, 1, false, ""},
		{"outage", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), false, true, 0, false, "暫時"},
		{"registered video", linebot.EventTypeMessage, &linebot.VideoMessage{}, true, false, 0, true, ""},
		{"registration replay", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), true, false, 0, false, "已完成登記"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			links := 0
			message := ""
			event := &linebot.Event{Type: tc.kind, Message: tc.message,
				Source: &linebot.EventSource{Type: linebot.EventSourceTypeUser, UserID: "U1"}}
			_, allowed := gateExperimentEvent(event,
				func(string) (*db.UserData, error) {
					if tc.lookupError {
						return nil, errors.New("outage")
					}
					if tc.existing {
						return registered(), nil
					}
					return nil, status.Error(codes.NotFound, "absent")
				},
				func(_ *linebot.Event, text string) { message = text },
				func(*linebot.Event) { links++ })
			if allowed != tc.allowed || links != tc.links ||
				(tc.reply != "" && !strings.Contains(message, tc.reply)) {
				t.Fatalf("allowed=%v links=%d reply=%s", allowed, links, message)
			}
		})
	}
}

func TestGroupRegistrationDoesNotAccessDatabase(t *testing.T) {
	called := false
	_, allowed := gateExperimentEvent(&linebot.Event{Type: linebot.EventTypeMessage,
		Source: &linebot.EventSource{Type: linebot.EventSourceTypeGroup, UserID: "U1"}},
		func(string) (*db.UserData, error) { called = true; return nil, nil },
		func(_ *linebot.Event, message string) {
			if !strings.Contains(message, "一對一") {
				t.Fatal(message)
			}
		},
		func(*linebot.Event) { t.Fatal("group request was offered the registration form") })
	if allowed || called {
		t.Fatal("group request reached registration")
	}
}
