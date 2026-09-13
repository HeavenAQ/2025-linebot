package app

import (
	"errors"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"strings"
	"testing"
	"time"
)

func TestRegistrationBlocksActionsUntilSaved(t *testing.T) {
	registered := func() *db.UserData {
		return &db.UserData{ID: "U1", RealName: "王小明",
			ExperimentNumber: "01", RegistrationVersion: 1, RegistrationCompletedAt: time.Now()}
	}
	for _, tc := range []struct {
		name                   string
		kind                   linebot.EventType
		message                linebot.Message
		existing               bool
		lookupError, saveError bool
		creates, saves         int
		allowed                bool
		reply                  string
	}{
		{"follow", linebot.EventTypeFollow, nil, false, false, false, 0, 0, false, "請先"},
		{"video", linebot.EventTypeMessage, &linebot.VideoMessage{}, false, false, false, 0, 0, false, "請先"},
		{"postback", linebot.EventTypePostback, nil, false, false, false, 0, 0, false, "請先"},
		{"bad name", linebot.EventTypeMessage, linebot.NewTextMessage("01 王123"), false, false, false, 0, 0, false, "格式不正確"},
		{"rich menu", linebot.EventTypeMessage, linebot.NewTextMessage("動作分析"), false, false, false, 0, 0, false, "格式不正確"},
		{"register", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), false, false, false, 1, 1, false, "登記完成"},
		{"save failure", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), false, false, true, 1, 1, false, "登記尚未完成"},
		{"outage", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), false, true, false, 0, 0, false, "暫時"},
		{"registered video", linebot.EventTypeMessage, &linebot.VideoMessage{}, true, false, false, 0, 0, true, ""},
		{"registration replay", linebot.EventTypeMessage, linebot.NewTextMessage("01 王小明"), true, false, false, 0, 0, false, "已完成登記"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			creates, saves := 0, 0
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
				func(string) *db.UserData { creates++; return &db.UserData{ID: "U1"} },
				func(id string, r db.ExperimentRegistration) (*db.UserData, error) {
					saves++
					if id != "U1" || r.RealName != "王小明" || r.ExperimentNumber != "01" {
						t.Fatal("wrong registration")
					}
					if tc.saveError {
						return nil, errors.New("write failed")
					}
					return registered(), nil
				}, func(_ *linebot.Event, text string) { message = text })
			if allowed != tc.allowed || creates != tc.creates || saves != tc.saves ||
				(tc.reply != "" && !strings.Contains(message, tc.reply)) {
				t.Fatalf("allowed=%v creates=%d saves=%d reply=%s", allowed, creates, saves, message)
			}
		})
	}
}

func TestGroupRegistrationDoesNotAccessDatabase(t *testing.T) {
	called := false
	_, allowed := gateExperimentEvent(&linebot.Event{Type: linebot.EventTypeMessage,
		Source: &linebot.EventSource{Type: linebot.EventSourceTypeGroup, UserID: "U1"}},
		func(string) (*db.UserData, error) { called = true; return nil, nil }, nil, nil,
		func(_ *linebot.Event, message string) {
			if !strings.Contains(message, "一對一") {
				t.Fatal(message)
			}
		})
	if allowed || called {
		t.Fatal("group request reached registration")
	}
}
