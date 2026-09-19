package gpt

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// toolServer answers with the queued responses in order, recording every
// request body so a test can see what travelled back to the model.
type toolServer struct {
	replies  []string
	requests []map[string]any
}

func (t *toolServer) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		t.requests = append(t.requests, body)
		reply := `{"id":"resp","object":"response","status":"completed","output":[
			{"type":"message","role":"assistant","status":"completed",
			 "content":[{"type":"output_text","text":"最後回覆"}]}]}`
		if len(t.requests) <= len(t.replies) {
			reply = t.replies[len(t.requests)-1]
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}
}

func callReply(callID, name, arguments string) string {
	return `{"id":"resp","object":"response","status":"completed","output":[
		{"type":"function_call","id":"fc","call_id":"` + callID + `","name":"` + name + `","arguments":` +
		mustJSON(arguments) + `,"status":"completed"}]}`
}

func mustJSON(value string) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(encoded)
}

func testClient(t *testing.T, server *toolServer) *Client {
	t.Helper()
	httpServer := httptest.NewServer(server.handler())
	t.Cleanup(httpServer.Close)
	ctx := context.Background()
	openaiClient := openai.NewClient(option.WithAPIKey("test"), option.WithBaseURL(httpServer.URL))
	return &Client{Ctx: &ctx, Client: &openaiClient, Model: DefaultModel}
}

// The learner asks something no fixed prompt could carry, so the coach looks it
// up and answers from what came back.
func TestCoachRunsAToolAndAnswersFromItsResult(t *testing.T) {
	server := &toolServer{replies: []string{callReply("call-1", "get_class_standing", `{"skill":"smash"}`)}}
	client := testClient(t, server)

	var seen []ToolCallRecord
	reply, err := client.CoachWithTools(context.Background(), nil, "我在班上排第幾？", "殺球", nil,
		[]Tool{{
			Name:       "get_class_standing",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
			Handler: func(context.Context, string) (string, error) {
				return `{"rank":2,"percentile":75}`, nil
			},
		}},
		func(record ToolCallRecord) { seen = append(seen, record) })
	if err != nil {
		t.Fatal(err)
	}
	if reply != "最後回覆" {
		t.Errorf("reply %q, want the answer after the lookup", reply)
	}
	if len(seen) != 1 || seen[0].Name != "get_class_standing" {
		t.Fatalf("tool calls recorded: %+v", seen)
	}
	if len(server.requests) != 2 {
		t.Fatalf("made %d requests, want the call and the follow-up", len(server.requests))
	}
	// The second request must carry both the model's call and its result.
	// The tool's own JSON travels as a string, so it appears escaped here.
	encoded, _ := json.Marshal(server.requests[1]["input"])
	for _, want := range []string{"call-1", "function_call_output", `rank\":2`} {
		if !strings.Contains(string(encoded), want) {
			t.Errorf("follow-up input missing %s: %s", want, encoded)
		}
	}
}

// A name the model invented must come back as a failed lookup it can work
// around, not as an error that loses the learner's question.
func TestUnknownToolIsReportedToTheModel(t *testing.T) {
	server := &toolServer{replies: []string{callReply("call-1", "delete_everything", `{}`)}}
	client := testClient(t, server)
	reply, err := client.CoachWithTools(context.Background(), nil, "?", "殺球", nil,
		[]Tool{{Name: "get_class_standing", Parameters: map[string]any{"type": "object"},
			Handler: func(context.Context, string) (string, error) { return "{}", nil }}}, nil)
	if err != nil || reply != "最後回覆" {
		t.Fatalf("reply %q err %v", reply, err)
	}
	encoded, _ := json.Marshal(server.requests[1]["input"])
	if !strings.Contains(string(encoded), "no such tool") {
		t.Errorf("the model was not told the tool does not exist: %s", encoded)
	}
}

// A model that keeps asking must still produce an answer.
func TestToolRoundsAreBounded(t *testing.T) {
	replies := make([]string, 12)
	for i := range replies {
		replies[i] = callReply("call", "loop", `{}`)
	}
	server := &toolServer{replies: replies}
	client := testClient(t, server)
	calls := 0
	_, err := client.CoachWithTools(context.Background(), nil, "?", "殺球", nil,
		[]Tool{{Name: "loop", Parameters: map[string]any{"type": "object"},
			Handler: func(context.Context, string) (string, error) { calls++; return "{}", nil }}}, nil)
	if err == nil {
		t.Fatal("a model that only ever calls tools has no answer to give")
	}
	if calls > maxToolRounds {
		t.Errorf("ran %d lookups, want at most %d", calls, maxToolRounds)
	}
}

// Without tools the request must look exactly as it did before.
func TestCoachWithoutToolsSendsNoToolsField(t *testing.T) {
	server := &toolServer{}
	client := testClient(t, server)
	if _, err := client.Coach(nil, "怎麼練發球", "發球", nil); err != nil {
		t.Fatal(err)
	}
	if _, ok := server.requests[0]["tools"]; ok {
		t.Error("a plain question should not offer tools")
	}
}
