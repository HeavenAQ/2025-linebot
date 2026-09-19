package app

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/line"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	linebot "github.com/line/line-bot-sdk-go/v7/linebot"
)

const chatQuestionQueued = "收到你的問題，分析完成後一起回答。"

// chatModeWelcome says what this mode can do and sets expectations about the
// wait: an analysis takes a while, and the answer comes back in one message.
const chatModeWelcome = "已進入和GPT對話模式，可以直接問我這個動作的問題。\n\n" +
	"你也可以在這裡上傳一支練習影片，我會分析後把骨架影片和說明一起回覆（大約需要半分鐘到一分鐘，請稍候）。\n" +
	"影片和問題誰先誰後都可以。如果等太久沒有收到回覆，隨便傳一則訊息給我，我就會把結果補給你。"

// chatAnswerDelayed heads an answer that missed its reply window. The video is
// not attached: by then only a push could carry it, and those are metered.
const chatAnswerDelayed = "剛才的影片分析完成了，骨架影片可以在學習網頁上看。\n\n"

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
	// The reply token is held for the finished answer rather than spent on an
	// acknowledgement: replies are free, pushes are not. The learner sees the
	// typing indicator in the meantime.
	app.showChatLoading(user.ID)
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
	answer, err := app.chatAnalysisAnswer(ctx, job, work)
	if err != nil {
		app.Logger.Error.Printf("chat analysis answer job=%s: %v", job.ID, err)
		answer = "分析完成了，但我這邊暫時無法產生說明。你可以再問我一次。"
	}

	// One reply carries both the overlay and the answer. If the window has
	// closed, the answer waits for the learner's next message instead of
	// costing a push.
	if err := app.replyChatAnalysis(job, work, answer); err != nil {
		app.Logger.Warn.Printf("chat analysis reply job=%s: %v", job.ID, err)
		if err := app.FirestoreClient.SetPendingAnswer(job.UserID, chatAnswerDelayed+answer); err != nil {
			app.Logger.Error.Printf("hold chat analysis answer job=%s: %v", job.ID, err)
		}
	}
	if err := app.FirestoreClient.AppendChatExchange(job.UserID, job.Skill, chatQuestion(job), answer); err != nil {
		app.Logger.Warn.Printf("append chat analysis exchange job=%s: %v", job.ID, err)
	}
}

// showChatLoading animates the typing indicator while the coach works. Every
// answer in this mode costs a model call -- two when tools are used -- which
// is long enough for silence to read as the bot having ignored the learner.
// The indicator is free, unlike a holding message, and clears as soon as the
// reply lands.
func (app *App) showChatLoading(userID string) {
	if err := app.LineBot.ShowLoading(context.Background(), userID, 60); err != nil {
		app.Logger.Warn.Printf("chat loading indicator: %v", err)
	}
}

func chatQuestion(job db.AnalysisJob) string {
	if question := strings.TrimSpace(job.Question); question != "" {
		return question
	}
	return "（上傳了一支影片）"
}

// replyChatAnalysis answers with the correction skeleton overlay -- the
// analysis render with no coaching text burned into it -- and the coach's
// explanation, in one reply.
func (app *App) replyChatAnalysis(job db.AnalysisJob, work db.Work, answer string) error {
	if job.ReplyToken == "" {
		return fmt.Errorf("no reply token")
	}
	video, err := app.chatOverlayVideo(job, work)
	if err != nil {
		app.Logger.Warn.Printf("chat analysis video job=%s: %v", job.ID, err)
	}
	return app.LineBot.ReplyChatAnalysis(job.ReplyToken, video, answer)
}

// chatOverlayVideo signs the overlay render for playback in the chat.
func (app *App) chatOverlayVideo(job db.AnalysisJob, work db.Work) (line.VideoContent, error) {
	overlay := work.SkeletonOverlayVideo
	if overlay.ObjectPath == "" {
		overlay = work.StudentVideo
	}
	if overlay.ObjectPath == "" {
		return line.VideoContent{}, fmt.Errorf("analysis produced no video")
	}
	signed, err := app.StorageClient.SignPlaybackURLIn(
		storageBucketFrom(overlay.GCSURI), overlay.ObjectPath, app.Config.GCP.ServiceAccountEmail,
	)
	if err != nil {
		return line.VideoContent{}, fmt.Errorf("sign overlay: %w", err)
	}
	thumbnail, err := app.StorageClient.SignThumbnailURL(work.Thumbnail, app.Config.GCP.ServiceAccountEmail)
	if err != nil {
		// A missing preview image is not worth withholding the video over.
		app.Logger.Warn.Printf("chat analysis thumbnail job=%s: %v", job.ID, err)
	}
	return line.VideoContent{URL: signed.SignedURL, ThumbnailURL: thumbnail.SignedURL}, nil
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
