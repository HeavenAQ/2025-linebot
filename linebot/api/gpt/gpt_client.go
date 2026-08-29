package gpt

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/HeavenAQ/nstc-linebot-2025/commons"
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

// DefaultModel is what every request runs on unless OPENAI_MODEL says
// otherwise.
//
// Named here rather than left to a stored OpenAI prompt. A stored prompt pins
// whichever model it was saved against, and OpenAI retires those: when
// gpt-5.2-chat-latest was withdrawn, every summary started coming back 404
// with nothing in this repository naming the model, so there was no way to fix
// it from here. The system prompts live in this file for the same reason --
// they are reviewable, versioned with the code that sends them, and cannot
// change under the service without a deploy.
const DefaultModel = "gpt-5.6-terra"

type Client struct {
	Ctx    *context.Context
	Client *openai.Client
	Model  string
}

func NewGPTClient(apiKey, model string) *Client {
	ctx := context.Background()
	if model == "" {
		model = DefaultModel
	}
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
	)

	return &Client{
		Ctx:    &ctx,
		Client: &client,
		Model:  model,
	}
}

// coachInstruction is the persona behind the bot's replies to learners.
//
// It is deliberately explicit that questions about a learner's own progress
// are in scope: the stored prompt this replaced used to refuse them outright
// and tell the student to go find a real coach, which is the one thing a
// coaching bot must not do when it has the scores in front of it.
const coachInstruction = "你是一位羽球教練，正在指導大學體育課的學生。" +
	"學生會問你關於自己練習的問題，訊息中通常附有系統的動作評分。\n" +
	"- 一律使用繁體中文，語氣直接、鼓勵，像在球場邊說話。\n" +
	"- 你的工作就是評估與給建議。學生問自己的學習進度、動作好不好、" +
	"該怎麼改進時，一律直接給出評估與具體練法。\n" +
	"- 絕對不要說自己無法分析或評估動作表現，也不要叫學生去問別的教練或找專業人士——" +
	"你就是他的教練。\n" +
	"- 有分數時，先說目前的水準與趨勢，點出最弱的項目，再給那個項目的練法。\n" +
	"- 建議要具體到身體部位與練得到的動作，例如「擊球瞬間手腕先放鬆再快速前甩」，" +
	"而不是「多多練習」。\n" +
	"- 只根據提供的分數與對話內容說話，不要杜撰沒有出現過的數字。" +
	"真的沒有任何資料時，問一個具體的問題把資料問出來，不要空泛地拒絕。\n" +
	"- 訊息開頭會標明本次討論的動作，只針對那個動作回答；" +
	"不要改談其他動作，也不要用其他動作的技術要點來解釋。\n" +
	"- 這是 LINE 訊息，控制在 200 字以內。逐項回饋時每項一行、以數字開頭。"

type HistoryMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

func (client *Client) RewriteQuery(history []HistoryMessage, query string) (string, error) {
	if len(history) == 0 {
		return query, nil
	}
	if len(history) > 12 {
		history = history[len(history)-12:]
	}
	payload, err := json.Marshal(struct {
		History []HistoryMessage `json:"history"`
		Query   string           `json:"query"`
	}{History: history, Query: query})
	if err != nil {
		return "", fmt.Errorf("marshal query rewrite context: %w", err)
	}
	req := responses.ResponseNewParams{
		Model:        client.Model,
		Instructions: param.Opt[string]{Value: "Rewrite the latest user query as one standalone query using only necessary context from the conversation history. Preserve the user's language and intent. Resolve pronouns and omitted badminton skill references. Do not answer the query, add advice, or mention the history. Return only the rewritten query."},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{Value: string(payload)},
		},
		MaxOutputTokens: param.Opt[int64]{Value: 300},
		Store:           param.Opt[bool]{Value: false},
	}
	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("rewrite query: %w", err)
	}
	rewritten := resp.OutputText()
	if rewritten == "" {
		return "", fmt.Errorf("query rewrite returned empty output")
	}
	return rewritten, nil
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

// AddMessageToConversation sends a learner's question through their skill
// conversation and returns the coach's reply.
//
// The recent grades ride along with the question. Without them the coach has
// nothing to evaluate and falls back on asking the learner what they have been
// practising, which is a poor answer to "how am I doing" when the scores are
// sitting in Firestore.
func (client *Client) AddMessageToConversation(
	conversationID, message, skillChn string, scores []commons.SkillScore,
) (string, error) {
	var input strings.Builder
	// Name the stroke. Each skill has its own conversation, but a fresh one
	// carries no prior turns, and the grades below are only criterion names
	// and numbers -- nothing in them says which stroke they belong to. Asked
	// "what went wrong this time" with no stroke named, the model has to guess,
	// and it has answered about the wrong one.
	if strings.TrimSpace(skillChn) != "" {
		input.WriteString(fmt.Sprintf("[本次討論的動作] %s\n", skillChn))
	}
	writeScores(&input, scores)
	if input.Len() > 0 {
		input.WriteString("\n\n")
	}
	input.WriteString(message)

	req := responses.ResponseNewParams{
		Model:        client.Model,
		Instructions: param.Opt[string]{Value: coachInstruction},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{
				Value: input.String(),
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

const summaryInstruction = "Summarize the learner's badminton progress in under 100 words, in the language the learner uses. " +
	"Ground the summary in the recent scores below: state the trend across attempts, the latest total, and the criterion that scores lowest. " +
	"Where the conversation and the scores disagree, trust the scores. Do not invent scores that are not listed."

// writeScores renders the learner's recent grades. The coach and the summary
// share it so a learner cannot be told two different things about the same
// numbers depending on which one they asked.
func writeScores(b *strings.Builder, scores []commons.SkillScore) {
	if len(scores) == 0 {
		return
	}
	b.WriteString("[Recent scores, newest first]")
	for _, score := range scores {
		b.WriteString(fmt.Sprintf("\n- %s: total %.1f", score.Date, score.TotalGrade))
		if strings.TrimSpace(score.ScoreStatus) != "" {
			b.WriteString(fmt.Sprintf(" (%s)", score.ScoreStatus))
		}
		for _, detail := range score.Details {
			b.WriteString(fmt.Sprintf("\n    * %s: %.1f/%.1f", detail.Description, detail.Grade, detail.Maximum))
		}
	}
}

// buildSummaryPrompt lays the learner's recent grades alongside the chat
// content so the summary reflects how they are actually scoring, not only what
// they talked about. Either section may be missing.
func buildSummaryPrompt(content string, scores []commons.SkillScore) string {
	var b strings.Builder
	writeScores(&b, scores)

	if strings.TrimSpace(content) != "" {
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString("[Conversation]\n")
		b.WriteString(content)
	}
	return b.String()
}

const weeklyPreviewInstruction = "你是羽球教練，正在為學生準備本週的課前預習提醒。" +
	"請用繁體中文，先用一行說明這個動作為什麼最需要加強（根據分數趨勢與最弱的細項），" +
	"再列出兩到三個具體可練習的重點，每點一行、以「・」開頭。" +
	"全部不超過 150 字，只根據提供的分數，不要杜撰沒有列出的數據。"

// buildWeeklyPreviewPrompt lays out every skill the learner has attempted, so
// the reasons can draw on the whole picture even though the focus is fixed.
// The caller picks the focus skill rather than the model, because the push
// names that skill in its header and the two must not disagree.
func buildWeeklyPreviewPrompt(
	displayName string,
	focusSkill string,
	history []commons.SkillHistory,
) string {
	var b strings.Builder
	b.WriteString("本週要加強的動作：")
	b.WriteString(focusSkill)
	if strings.TrimSpace(displayName) != "" {
		b.WriteString("\n學生：")
		b.WriteString(displayName)
	}
	for _, skill := range history {
		b.WriteString(fmt.Sprintf("\n\n[%s]", skill.Skill))
		for _, score := range skill.Scores {
			b.WriteString(fmt.Sprintf("\n- %s: 總分 %.1f", score.Date, score.TotalGrade))
			for _, detail := range score.Details {
				b.WriteString(fmt.Sprintf("\n    * %s: %.1f/%.1f", detail.Description, detail.Grade, detail.Maximum))
			}
		}
	}
	return b.String()
}

// WeeklyPreview writes the 課前預習 note for a learner from their past scores.
func (client *Client) WeeklyPreview(
	displayName string,
	focusSkill string,
	history []commons.SkillHistory,
) (string, error) {
	if len(history) == 0 {
		return "", fmt.Errorf("weekly preview needs at least one graded skill")
	}
	if strings.TrimSpace(focusSkill) == "" {
		return "", fmt.Errorf("weekly preview needs a focus skill")
	}
	req := responses.ResponseNewParams{
		Model:        client.Model,
		Instructions: param.Opt[string]{Value: weeklyPreviewInstruction},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{
				Value: buildWeeklyPreviewPrompt(displayName, focusSkill, history),
			},
		},
	}

	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("error creating weekly preview response: %w", err)
	}
	output := resp.OutputText()
	if output == "" {
		return "", fmt.Errorf("no assistant text output available")
	}
	return output, nil
}

// Summarize turns the learner's conversation and their recent grades into a
// short summary using the configured prompt.
func (client *Client) Summarize(content string, scores []commons.SkillScore) (string, error) {
	req := responses.ResponseNewParams{
		Model:        client.Model,
		Instructions: param.Opt[string]{Value: summaryInstruction},
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{
				Value: buildSummaryPrompt(content, scores),
			},
		},
	}

	resp, err := client.Client.Responses.New(*client.Ctx, req)
	if err != nil {
		return "", fmt.Errorf("error creating summary response: %w", err)
	}

	output := resp.OutputText()
	if output == "" {
		return "", fmt.Errorf("no assistant text output available")
	}
	return output, nil
}
