package db

import (
	"context"
	"errors"
	"fmt"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var ErrJobBusy = errors.New("analysis already running")

// Where a video came from. A chat upload is analysed without the pipeline's
// coaching stage and answered by the coach in the conversation instead.
const (
	AnalysisSourceUpload = "upload"
	AnalysisSourceChat   = "chat"
	// AnalysisSourceChatCoaching fills in the coaching a chat upload skipped
	// to answer quickly. It runs against the same stored video and merges only
	// its cues into the attempt that is already recorded.
	AnalysisSourceChatCoaching = "chat-coaching"
)

type AnalysisJob struct {
	ID          string `firestore:"id"`
	UserID      string `firestore:"user_id"`
	Skill       string `firestore:"skill"`
	Handedness  string `firestore:"handedness"`
	WorkDate    string `firestore:"work_date"`
	InputObject string `firestore:"input_object"`
	Thumbnail   string `firestore:"thumbnail"`
	Status      string `firestore:"status"`
	// Source is "upload" for the usual portfolio flow, "chat" when the learner
	// sent the video while talking to the coach. A chat analysis skips the
	// pipeline's own coaching stage: the explaining happens in the reply.
	Source string `firestore:"source"`
	// Question is what the learner asked about this video, if they have asked
	// yet. The coach answers it once the analysis lands.
	Question string `firestore:"question"`
	// ReplyToken answers a chat analysis without spending the account's push
	// quota. LINE only honours it for about a minute, so a slow analysis falls
	// back to delivering the answer with the learner's next message.
	ReplyToken string    `firestore:"reply_token"`
	Attempts   int       `firestore:"attempts"`
	CreatedAt  time.Time `firestore:"created_at"`
	LeaseUntil time.Time `firestore:"lease_until"`
}

func (c *FirestoreClient) AnalysisJobs() *firestore.CollectionRef {
	// Preserve the deployment's existing Firestore isolation.
	return c.Client.Collection(c.Data.ID + "_analysis_jobs")
}

func (c *FirestoreClient) CreateAnalysisJob(ctx context.Context, job AnalysisJob) (AnalysisJob, error) {
	ref := c.AnalysisJobs().Doc(job.ID)
	err := c.Client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		existing, err := tx.Get(ref)
		if err == nil {
			return existing.DataTo(&job)
		}
		if status.Code(err) != codes.NotFound {
			return err
		}
		userRef := c.Data.Doc(job.UserID)
		if _, err := tx.Get(userRef); err != nil {
			return err
		}
		work := Work{DateTime: job.WorkDate, Handedness: job.Handedness, Thumbnail: job.Thumbnail,
			AnalysisID: job.ID, AnalysisStatus: "pending", Reflection: "尚未填寫心得",
			GradingOutcome: commons.GradingOutcome{ScoreStatus: "pending", GradingDetails: []commons.GradingDetail{}},
		}
		if err := tx.Create(ref, job); err != nil {
			return err
		}
		return tx.Update(userRef, []firestore.Update{{FieldPath: []string{"portfolio", job.Skill, job.WorkDate}, Value: work}})
	})
	return job, err
}

func (c *FirestoreClient) ClaimAnalysisJob(ctx context.Context, id string) (AnalysisJob, error) {
	var job AnalysisJob
	err := c.Client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		ref := c.AnalysisJobs().Doc(id)
		doc, err := tx.Get(ref)
		if err != nil {
			return err
		}
		if err := doc.DataTo(&job); err != nil {
			return err
		}
		if job.Status == "completed" || job.Status == "failed" {
			return nil
		}
		if time.Now().Before(job.LeaseUntil) {
			return ErrJobBusy
		}
		job.Status = "running"
		job.Attempts++
		job.LeaseUntil = time.Now().Add(19 * time.Minute)
		return tx.Set(ref, job)
	})
	return job, err
}

// Finish only the lease this worker owns, and only grading/media fields. Never
// overwrite a learner's reflection or another concurrently completed analysis.
func (c *FirestoreClient) FinishAnalysisJob(ctx context.Context, job AnalysisJob, outcome *commons.AnalysisOutcome, failure string, retry bool) error {
	return c.Client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		ref := c.AnalysisJobs().Doc(job.ID)
		doc, err := tx.Get(ref)
		if err != nil {
			return err
		}
		var current AnalysisJob
		if err := doc.DataTo(&current); err != nil {
			return err
		}
		if current.Status == "completed" || current.Status == "failed" {
			return nil
		}
		if current.Attempts != job.Attempts {
			return fmt.Errorf("analysis lease superseded")
		}
		// A completed attempt joins the class aggregate in the same transaction,
		// so a retried finish can never count it twice.
		var statRef *firestore.DocumentRef
		var stat ClassStat
		if outcome != nil {
			day, err := workDay(job.WorkDate)
			if err != nil {
				return err
			}
			if statRef, stat, err = c.readClassStat(tx, job.Skill, day); err != nil {
				return err
			}
		}
		state := "failed"
		if retry {
			state = "queued"
		}
		if outcome != nil {
			state = "completed"
		}
		if err := tx.Update(ref, []firestore.Update{{Path: "status", Value: state}, {Path: "lease_until", Value: time.Time{}}}); err != nil {
			return err
		}
		if retry {
			return nil
		}
		values := map[string]any{"analysis_status": state, "analysis_error": failure}
		if outcome != nil {
			stat.Add(outcome.Grade.TotalGrade)
			stat.UpdatedAt = time.Now()
			if err := tx.Set(statRef, stat); err != nil {
				return err
			}
			values["analysis_id"] = outcome.AnalysisID
			values["grading_outcome"] = outcome.Grade
			values["student_video"] = outcome.StudentVideo
			values["feedback_video"] = outcome.FeedbackVideo
			values["skeleton_overlay_video"] = outcome.SkeletonOverlayVideo
			values["expert"] = outcome.Expert
			values["timeline"] = outcome.Timeline
			values["coaching_cues"] = outcome.CoachingCues
			values["ai_note"] = outcome.OverallFeedback
			values["diagnostics"] = outcome.Diagnostics
			values["handedness"] = outcome.Handedness
		} else {
			values["grading_outcome.score_status"] = "failed"
		}
		updates := []firestore.Update{}
		for key, value := range values {
			path := []string{"portfolio", job.Skill, job.WorkDate, key}
			if key == "grading_outcome.score_status" {
				path = []string{"portfolio", job.Skill, job.WorkDate, "grading_outcome", "score_status"}
			}
			updates = append(updates, firestore.Update{FieldPath: path, Value: value})
		}
		return tx.Update(c.Data.Doc(job.UserID), updates)
	})
}

// SetAnalysisJobQuestion attaches the learner's question to a running chat
// analysis, for when they send the video first and ask afterwards.
func (c *FirestoreClient) SetAnalysisJobQuestion(ctx context.Context, jobID, question string) error {
	_, err := c.AnalysisJobs().Doc(jobID).Update(ctx, []firestore.Update{{Path: "question", Value: question}})
	return err
}

// PendingChatAnalysis returns the learner's chat analysis that is still
// running, if there is one.
func (c *FirestoreClient) PendingChatAnalysis(ctx context.Context, userID string) (*AnalysisJob, error) {
	iter := c.AnalysisJobs().Where("user_id", "==", userID).Where("source", "==", AnalysisSourceChat).Documents(ctx)
	defer iter.Stop()
	var newest *AnalysisJob
	for {
		doc, err := iter.Next()
		if err != nil {
			break
		}
		var job AnalysisJob
		if doc.DataTo(&job) != nil {
			continue
		}
		if job.Status == "completed" || job.Status == "failed" {
			continue
		}
		if newest == nil || job.CreatedAt.After(newest.CreatedAt) {
			copied := job
			newest = &copied
		}
	}
	return newest, nil
}

// MergeCoaching adds the coaching pass's results to an attempt already shown
// to the learner. Only the coaching fields move, so a second pass can never
// change a score, a video or the expert match after the fact.
func (c *FirestoreClient) MergeCoaching(ctx context.Context, job AnalysisJob, outcome *commons.AnalysisOutcome) error {
	if outcome == nil {
		return fmt.Errorf("no coaching outcome")
	}
	updates := []firestore.Update{}
	for _, path := range CoachingUpdatePaths(job, outcome) {
		value := any(outcome.CoachingCues)
		switch path[len(path)-1] {
		case "ai_note":
			value = outcome.OverallFeedback
		case "feedback_video":
			value = outcome.FeedbackVideo
		}
		updates = append(updates, firestore.Update{FieldPath: path, Value: value})
	}
	_, err := c.Data.Doc(job.UserID).Update(ctx, updates)
	return err
}

// CoachingUpdatePaths lists exactly the fields the coaching pass may write.
func CoachingUpdatePaths(job AnalysisJob, outcome *commons.AnalysisOutcome) [][]string {
	base := []string{"portfolio", job.Skill, job.WorkDate}
	field := func(name string) []string { return append(append([]string{}, base...), name) }
	paths := [][]string{field("coaching_cues"), field("ai_note")}
	if outcome != nil && outcome.FeedbackVideo.ObjectPath != "" {
		paths = append(paths, field("feedback_video"))
	}
	return paths
}

// MarkAnalysisJobState finishes a job that owns no portfolio record of its
// own, such as the coaching pass that only merges cues into an existing one.
func (c *FirestoreClient) MarkAnalysisJobState(ctx context.Context, jobID, state string) error {
	_, err := c.AnalysisJobs().Doc(jobID).Update(ctx, []firestore.Update{
		{Path: "status", Value: state},
		{Path: "lease_until", Value: time.Time{}},
	})
	return err
}
