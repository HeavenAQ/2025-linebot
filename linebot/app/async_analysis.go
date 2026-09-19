package app

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysis"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/obs"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	line "github.com/line/line-bot-sdk-go/v7/linebot"
	"google.golang.org/api/idtoken"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// enqueueVideoAnalysis accepts a video and starts its analysis. Passing a
// chatAnalysisRequest marks it as coming from the conversation: the pipeline's
// own coaching stage is skipped and the coach answers in chat instead.
func (a *App) enqueueVideoAnalysis(event *line.Event, session *db.UserSession, user *db.UserData, video []byte, replyToken string, chat ...chatAnalysisRequest) {
	source, question := db.AnalysisSourceUpload, ""
	if len(chat) > 0 {
		source, question = db.AnalysisSourceChat, chat[0].question
	}
	message, ok := event.Message.(*line.VideoMessage)
	if !ok {
		return
	}
	digest := sha256.Sum256([]byte(a.Config.GCP.Database.DataDB + ":" + user.ID + ":" + message.ID))
	id := hex.EncodeToString(digest[:])
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	// Check first so a duplicate LINE webhook cannot replace a finished record.
	var job db.AnalysisJob
	existing, err := a.FirestoreClient.AnalysisJobs().Doc(id).Get(ctx)
	if err == nil {
		if err = existing.DataTo(&job); err != nil {
			a.handleVideoAnalysisError(err, replyToken)
			return
		}
	} else if status.Code(err) == codes.NotFound {
		input := "analyses/input/" + user.ID + "/" + id + ".mp4"
		file := &storage.FileInfo{}
		file.Bucket.VideoPath, file.Local.VideoBlob = input, video
		if _, err = a.StorageClient.UploadVideo(file); err != nil {
			a.handleVideoAnalysisError(err, replyToken)
			return
		}
		thumb, err := a.createVideoThumbnail(video, user.ID)
		if err != nil {
			a.handleThumbnailCreationError(err, replyToken)
			return
		}
		defer os.RemoveAll(filepath.Dir(thumb))
		now := time.Now().In(time.FixedZone("Asia/Taipei", 8*3600))
		key := now.Format("2006-01-02-15-04-05") + "-" + id[:8]
		uploaded, err := a.uploadThumbnail(user, thumb, key)
		if err != nil {
			a.handleVideoAnalysisError(err, replyToken)
			return
		}
		// The thumbnail is durable BEFORE any card is sent; never replace that
		// object's contents later, since LINE caches preview images.
		job, err = a.FirestoreClient.CreateAnalysisJob(ctx, db.AnalysisJob{
			// Handedness is the learner's own, from their profile: the analysis
			// and the expert demonstrations both read it from one place.
			ID: id, UserID: user.ID, Skill: session.Skill, Handedness: user.Handedness.String(),
			WorkDate: key, InputObject: input, Thumbnail: uploaded.Path, Status: "queued", CreatedAt: now,
			Source: source, Question: question, ReplyToken: replyToken,
		})
		if err != nil {
			a.handleVideoAnalysisError(err, replyToken)
			return
		}
	} else {
		a.handleVideoAnalysisError(err, replyToken)
		return
	}
	if job.Status == "queued" {
		if err := a.AnalysisQueue.Enqueue(ctx, id); err != nil {
			// Durable outbox reconciliation retries publication; do not lose the
			// accepted video or ask the learner to upload it a second time.
			a.Logger.Error.Printf("analysis enqueue deferred job=%s: %v", id, err)
		}
	}
	if source == db.AnalysisSourceChat {
		return
	}
	if err := a.FirestoreClient.ResetSession(user.ID); err != nil {
		a.Logger.Warn.Println("reset upload session:", err)
	}
	updated, err := a.FirestoreClient.GetUserData(user.ID)
	if err != nil {
		a.handleVideoAnalysisError(err, replyToken)
		return
	}
	if err := a.sendPortfolio(event, updated, db.SkillStrToEnum(job.Skill), session.UserState,
		"影片已收到並加入學習歷程，正在分析中。稍後點擊查看結果，無需重新上傳。", true); err != nil {
		a.Logger.Error.Printf("send pending analysis portfolio: %v", err)
	}
}

// Cloud Run's bot endpoint is public for LINE webhooks. Internal workers must
// independently verify Google's signed OIDC token, exact audience and identity.
func (a *App) authorizeAnalysisTask(r *http.Request) bool {
	if a.AnalysisQueue == nil {
		return false
	}
	token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if token == "" || token == r.Header.Get("Authorization") {
		return false
	}
	claims, err := idtoken.Validate(r.Context(), token, a.Config.AnalysisServer.WorkerURL)
	return err == nil && claims.Claims["email"] == a.Config.AnalysisServer.TaskServiceAccount && claims.Claims["email_verified"] == true
}

func (a *App) HandleAnalysisTask(w http.ResponseWriter, r *http.Request) {
	if !a.authorizeAnalysisTask(r) {
		http.Error(w, "unauthorized", 401)
		return
	}
	var payload struct {
		JobID string `json:"job_id"`
	}
	if json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&payload) != nil || len(payload.JobID) != 64 || strings.ContainsAny(payload.JobID, "/.") {
		http.Error(w, "invalid job", 400)
		return
	}
	started := time.Now()
	// The job ID is the correlation ID: it names the learner's attempt in these
	// logs and, through gRPC metadata, in the analysis service's logs.
	ctx, cancel := context.WithTimeout(obs.WithRequestID(r.Context(), payload.JobID), 18*time.Minute)
	defer cancel()
	job, err := a.FirestoreClient.ClaimAnalysisJob(ctx, payload.JobID)
	if err != nil {
		http.Error(w, "job unavailable", 503)
		return
	}
	if job.Status == "completed" || job.Status == "failed" {
		w.WriteHeader(204)
		return
	}
	if job.Source == db.AnalysisSourceChatCoaching {
		a.runChatCoachingJob(ctx, w, job)
		return
	}
	var outcome *commons.AnalysisOutcome
	video, err := a.StorageClient.ReadAnalysisInput(ctx, job.InputObject)
	if err == nil {
		// A video sent in chat gets the plain analysis: the explaining is done
		// by the coach in the reply, against the learner's own question.
		outcome, err = a.AnalysisClient.AnalyzeVideo(ctx, job.ID, job.UserID, "line-upload.mp4",
			job.Skill, job.Handedness, video, job.Source == db.AnalysisSourceChat)
	}
	retry := err != nil && job.Attempts < 5 && time.Since(job.CreatedAt) < 24*time.Hour && !errors.Is(err, analysis.ErrSkillMismatch) && !errors.Is(err, analysis.ErrNoMatchingExpert) && status.Code(err) != codes.InvalidArgument && status.Code(err) != codes.FailedPrecondition
	failure := ""
	if err != nil {
		failure = "分析未能完成，請稍後重新上傳或聯絡老師。"
		if errors.Is(err, analysis.ErrSkillMismatch) {
			failure = "影片動作與選擇的技術不符，請確認發球或殺球後重新上傳。"
		}
		if errors.Is(err, analysis.ErrNoMatchingExpert) {
			failure = "目前沒有同慣用手的專家影片可供比較，本次不會跨左右手評分。請聯絡教練新增同手別的專家資料。"
		}
	}
	outcomeLabel, severity := "completed", obs.Info
	switch {
	case retry:
		outcomeLabel, severity = "retry", obs.Warning
	case err != nil:
		outcomeLabel, severity = "failed", obs.Error
	}
	fields := map[string]any{
		"job_id":           job.ID,
		"skill":            job.Skill,
		"attempt":          job.Attempts,
		"outcome":          outcomeLabel,
		"duration_seconds": time.Since(started).Seconds(),
	}
	if err != nil {
		fields["error"] = err.Error()
		fields["grpc_code"] = status.Code(err).String()
	}
	obs.Event(ctx, severity, "analysis job finished", fields)
	// Persist even if the RPC's context expired, so the lease does not remain
	// stuck. A failed persistence returns 503 and Cloud Tasks retries safely.
	finishCtx, finishCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer finishCancel()
	if e := a.FirestoreClient.FinishAnalysisJob(finishCtx, job, outcome, failure, retry); e != nil {
		http.Error(w, "save analysis failed", 503)
		return
	}
	if retry {
		http.Error(w, "retry analysis", 503)
		return
	}
	if job.Source == db.AnalysisSourceChat {
		if failure != "" {
			if _, replyErr := a.LineBot.SendGPTChattingModeReply(job.ReplyToken, failure); replyErr != nil {
				// The window closed; tell them with their next message rather
				// than spending a push on bad news.
				if holdErr := a.FirestoreClient.SetPendingAnswer(job.UserID, failure); holdErr != nil {
					a.Logger.Error.Printf("hold chat analysis failure job=%s: %v", job.ID, holdErr)
				}
			}
		} else {
			// The learner is waiting in the chat, so the answer is produced
			// here rather than left for them to ask again.
			answerCtx, answerCancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer answerCancel()
			a.answerChatAnalysis(answerCtx, job)
			// The attempt is in their portfolio like any other, so it should
			// carry the same coaching cues. That pass costs another 15-30 s,
			// which the learner has already been spared: it is queued after
			// they have their reply and only fills in the record.
			a.queueChatCoaching(answerCtx, job)
		}
	}
	w.WriteHeader(204)
}

// HandleClassStatsRebuild recomputes the class chart's aggregates from every
// portfolio. Cloud Scheduler calls it nightly to correct any drift.
func (a *App) HandleClassStatsRebuild(w http.ResponseWriter, r *http.Request) {
	if !a.authorizeAnalysisTask(r) {
		http.Error(w, "unauthorized", 401)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	started := time.Now()
	count, err := a.FirestoreClient.RebuildClassStats(ctx)
	if err != nil {
		obs.Event(ctx, obs.Error, "class stats rebuild failed", map[string]any{"error": err})
		http.Error(w, "rebuild failed", 503)
		return
	}
	obs.Event(ctx, obs.Info, "class stats rebuilt", map[string]any{
		"aggregates": count, "duration_seconds": time.Since(started).Seconds(),
	})
	w.WriteHeader(204)
}

// Reconcile the transactional outbox, not GPU progress. Handles a process dying
// between committing the pending record and publishing its deterministic task.
func (a *App) HandleAnalysisOutbox(w http.ResponseWriter, r *http.Request) {
	if !a.authorizeAnalysisTask(r) {
		http.Error(w, "unauthorized", 401)
		return
	}
	docs, err := a.FirestoreClient.AnalysisJobs().Where("status", "==", "queued").Limit(100).Documents(r.Context()).GetAll()
	if err != nil {
		http.Error(w, "outbox unavailable", 503)
		return
	}
	for _, doc := range docs {
		var job db.AnalysisJob
		if err := doc.DataTo(&job); err != nil {
			http.Error(w, "invalid outbox record", 503)
			return
		}
		if time.Since(job.CreatedAt) > 24*time.Hour {
			if err := a.FirestoreClient.FinishAnalysisJob(r.Context(), job, nil, "分析排程逾時，請重新上傳或聯絡老師。", false); err != nil {
				http.Error(w, "expire job failed", 503)
				return
			}
			continue
		}
		if err := a.AnalysisQueue.Enqueue(r.Context(), doc.Ref.ID); err != nil {
			http.Error(w, "publish failed", 503)
			return
		}
	}
	fmt.Fprint(w, "ok")
}

func (a *App) HandleAnalysisWarmup(w http.ResponseWriter, r *http.Request) {
	if !a.authorizeAnalysisTask(r) {
		http.Error(w, "unauthorized", 401)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()
	if err := a.AnalysisClient.Warmup(ctx); err != nil {
		http.Error(w, "GPU warmup failed", 503)
		return
	}
	w.WriteHeader(204)
}

func (a *App) HandleAnalysisCapacity(w http.ResponseWriter, r *http.Request) {
	if !a.authorizeAnalysisTask(r) {
		http.Error(w, "unauthorized", 401)
		return
	}
	var payload struct {
		Minimum *int `json:"minimum_instances"`
	}
	if json.NewDecoder(http.MaxBytesReader(w, r.Body, 256)).Decode(&payload) != nil || payload.Minimum == nil || *payload.Minimum < 0 || *payload.Minimum > 1 {
		http.Error(w, "invalid capacity", 400)
		return
	}
	if err := a.AnalysisQueue.SetGPUCapacity(r.Context(), a.Config.GCP.ProjectID, *payload.Minimum); err != nil {
		a.Logger.Error.Println(err)
		http.Error(w, "capacity update failed", 503)
		return
	}
	w.WriteHeader(204)
}
