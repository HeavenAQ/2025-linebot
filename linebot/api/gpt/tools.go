package gpt

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

// Tool is one thing the coach may look up while answering. The handler reads
// the learner's own records; which learner is decided by whoever builds the
// tool, never by the model, so a question cannot reach another learner's data.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON schema object. Keep it closed: every property
	// listed, additionalProperties false, so a malformed call is rejected by
	// OpenAI rather than reaching the handler.
	Parameters map[string]any
	// Handler returns the tool's answer as JSON. An error is reported to the
	// model as a failed lookup, which it can then work around, rather than
	// failing the learner's whole question.
	Handler func(ctx context.Context, arguments string) (string, error)
}

// maxToolRounds bounds the conversation: the coach may look things up a few
// times, but a learner waiting on a LINE message cannot wait on a loop that
// decides to keep going.
const maxToolRounds = 4

// toolCallTimeout bounds one lookup.
const toolCallTimeout = 10 * time.Second

func toolParams(tools []Tool) []responses.ToolUnionParam {
	params := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		params = append(params, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				Parameters:  tool.Parameters,
				Strict:      param.NewOpt(true),
			},
		})
	}
	return params
}

// runTool executes one call by name. An unknown name is reported back to the
// model rather than raised: the model chose it, and it can choose again.
func runTool(ctx context.Context, tools []Tool, name, arguments string) string {
	for _, tool := range tools {
		if tool.Name != name {
			continue
		}
		ctx, cancel := context.WithTimeout(ctx, toolCallTimeout)
		defer cancel()
		result, err := tool.Handler(ctx, arguments)
		if err != nil {
			return toolError(fmt.Sprintf("lookup failed: %v", err))
		}
		return result
	}
	return toolError(fmt.Sprintf("no such tool: %s", name))
}

func toolError(message string) string {
	encoded, err := json.Marshal(map[string]string{"error": message})
	if err != nil {
		return `{"error":"lookup failed"}`
	}
	return string(encoded)
}

// ToolCallRecord is one lookup the coach made, for the request log.
type ToolCallRecord struct {
	Name      string
	Arguments string
	Seconds   float64
}
