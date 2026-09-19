package app

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	linebot "github.com/line/line-bot-sdk-go/v7/linebot"
)

// chatAnalysisAcknowledgement is what the learner sees while their video is
// analysed. The answer follows as a push once the analysis lands, because by
// then the reply token is long expired.
const chatAnalysisAcknowledgement = "影片收到了，正在分析動作。分析完成後我會把骨架影片和說明一起傳給你。"

const chatQuestionQueued = "收到你的問題，分析完成後一起回答。"

// handleChatVideo takes a video sent while the learner is talking to the coach.
//
// The analysis runs without its own coaching stage: the learner gets the
// correction skeleton overlay, and the explaining happens here, in the
// conversation, against whatever they actually asked.
func (app *App) handleChatVideo(event *linebot.Event, session *db.UserSession, user *db.UserData, replyToken string) {
	if !db.IsSupportedSkill(session.Skill) {
		app.rejectUnsupportedSkill(user.ID, session.Skill, replyToken)
		return
	}
	if app.AnalysisQueue == nil || !app.Config.AnalysisServer.AsyncAccept {
		_, err := app.LineBot.SendReply(replyToken, "動作分析暫時暫停中，影片未被接收。請稍後再上傳一次。")
		handleLineMessageResponseError(err)
		return
	}
	video, err := app.getVideoContent(event, user.ID)
	if err != nil {
		app.handleGetVideoError(err, replyToken)
		return
	}
	// A question asked just before the video belongs with it.
	app.enqueueVideoAnalysis(event, session, user, video, replyToken, chatAnalysisRequest{
		question: session.PendingChatQuestion(),
	})
}

// chatAnalysisRequest marks an analysis as belonging to a conversation.
type chatAnalysisRequest struct {
	question string
}

// queueChatQuestion attaches a question to the analysis the learner is waiting
// on, so the two are answered together. It reports whether one was waiting.
func (app *App) queueChatQuestion(ctx context.Context, userID, question string) bool {
	job, err := app.FirestoreClient.PendingChatAnalysis(ctx, userID)
	if err != nil || job == nil {
		return false
	}
	if err := app.FirestoreClient.SetAnalysisJobQuestion(ctx, job.ID, question); err != nil {
		app.Logger.Warn.Printf("attach chat question job=%s: %v", job.ID, err)
		return false
	}
	return true
}

// answerChatAnalysis sends the finished analysis back into the conversation:
// the skeleton overlay to watch, then the coach's answer.
func (app *App) answerChatAnalysis(ctx context.Context, job db.AnalysisJob) {
	user, err := app.FirestoreClient.GetUserData(job.UserID)
	if err != nil {
		app.Logger.Error.Printf("chat analysis user lookup job=%s: %v", job.ID, err)
		return
	}
	work, ok := user.Portfolio.GetSkillPortfolio(job.Skill)[job.WorkDate]
	if !ok {
		app.Logger.Error.Printf("chat analysis work missing job=%s date=%s", job.ID, job.WorkDate)
		return
	}
	if err := app.sendChatOverlayVideo(job, work); err != nil {
		app.Logger.Error.Printf("chat analysis video job=%s: %v", job.ID, err)
	}

	answer, err := app.chatAnalysisAnswer(ctx, job, work)
	if err != nil {
		app.Logger.Error.Printf("chat analysis answer job=%s: %v", job.ID, err)
		answer = "分析完成了，但我這邊暫時無法產生說明。你可以再問我一次。"
	}
	if _, err := app.LineBot.PushGPTChattingModeReply(job.UserID, answer); err != nil {
		app.Logger.Error.Printf("chat analysis push job=%s: %v", job.ID, err)
		return
	}
	if err := app.FirestoreClient.AppendChatExchange(job.UserID, job.Skill, chatQuestion(job), answer); err != nil {
		app.Logger.Warn.Printf("append chat analysis exchange job=%s: %v", job.ID, err)
	}
}

func chatQuestion(job db.AnalysisJob) string {
	if question := strings.TrimSpace(job.Question); question != "" {
		return question
	}
	return "（上傳了一支影片）"
}

// sendChatOverlayVideo shows the correction skeleton overlay -- the analysis
// render without any coaching text burned into it.
func (app *App) sendChatOverlayVideo(job db.AnalysisJob, work db.Work) error {
	overlay := work.SkeletonOverlayVideo
	if overlay.ObjectPath == "" {
		overlay = work.StudentVideo
	}
	if overlay.ObjectPath == "" {
		return fmt.Errorf("analysis produced no video")
	}
	signed, err := app.StorageClient.SignPlaybackURLIn(
		storageBucketFrom(overlay.GCSURI), overlay.ObjectPath, app.Config.GCP.ServiceAccountEmail,
	)
	if err != nil {
		return fmt.Errorf("sign overlay: %w", err)
	}
	thumbnail, err := app.StorageClient.SignThumbnailURL(work.Thumbnail, app.Config.GCP.ServiceAccountEmail)
	if err != nil {
		// A missing preview image is not worth withholding the video over.
		app.Logger.Warn.Printf("chat analysis thumbnail job=%s: %v", job.ID, err)
	}
	_, err = app.LineBot.PushVideoMessage(job.UserID, signed.SignedURL, thumbnail.SignedURL)
	return err
}

// chatAnalysisAnswer asks the coach to explain this attempt, with the learner's
// own records available to look up.
func (app *App) chatAnalysisAnswer(ctx context.Context, job db.AnalysisJob, work db.Work) (string, error) {
	history, err := app.FirestoreClient.GetChatHistory(job.UserID)
	if err != nil {
		app.Logger.Warn.Printf("chat analysis history job=%s: %v", job.ID, err)
	}
	var turns []gpt.HistoryMessage
	if history != nil {
		for _, message := range history.Messages {
			if message.Skill == job.Skill {
				turns = append(turns, gpt.HistoryMessage{Role: message.Role, Text: message.Text})
			}
		}
	}

	var prompt strings.Builder
	prompt.WriteString("[剛上傳的影片分析結果]\n")
	prompt.WriteString(fmt.Sprintf("日期：%s\n總分：%.1f\n", job.WorkDate, work.GradingOutcome.TotalGrade))
	for _, detail := range work.GradingOutcome.GradingDetails {
		prompt.WriteString(fmt.Sprintf("- %s：%.1f/%.1f\n", detail.Description, detail.Grade, detail.Maximum))
	}
	prompt.WriteString("\n這次沒有另外產生影片內的建議，請你根據上面的評分標準說明。\n\n")
	if question := strings.TrimSpace(job.Question); question != "" {
		prompt.WriteString(question)
	} else {
		prompt.WriteString("請說明這次動作可以改進的地方。")
	}

	scores, err := app.FirestoreClient.GetRecentSkillScores(job.UserID, job.Skill, coachingScoreLimit)
	if err != nil {
		app.Logger.Warn.Printf("chat analysis scores job=%s: %v", job.ID, err)
	}
	started := time.Now()
	answer, err := app.GPTClient.CoachWithTools(
		ctx, turns, prompt.String(), db.SkillStrToEnum(job.Skill).ChnString(), scores,
		app.CoachingTools(job.UserID),
		func(record gpt.ToolCallRecord) {
			app.Logger.Info.Printf("[chat.tool] job=%s tool=%s args=%s took=%.2fs",
				job.ID, record.Name, record.Arguments, record.Seconds)
		},
	)
	app.Logger.Info.Printf("[chat.analysis] job=%s answered in %s", job.ID, time.Since(started))
	return answer, err
}

// storageBucketFrom keeps the bucket of a stored object, so a variant's own
// bucket is signed rather than the default one.
func storageBucketFrom(gcsURI string) string {
	return storage.BucketFromGCSURI(gcsURI)
}
