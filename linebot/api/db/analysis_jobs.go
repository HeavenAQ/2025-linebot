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
)

type AnalysisJob struct {
	ID          string    `firestore:"id"`
	UserID      string    `firestore:"user_id"`
	Skill       string    `firestore:"skill"`
	Handedness  string    `firestore:"handedness"`
	WorkDate    string    `firestore:"work_date"`
	InputObject string    `firestore:"input_object"`
	Thumbnail   string    `firestore:"thumbnail"`
	Status      string    `firestore:"status"`
	Attempts    int       `firestore:"attempts"`
	CreatedAt   time.Time `firestore:"created_at"`
	LeaseUntil  time.Time `firestore:"lease_until"`
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
		if outcome != nil && c.CountsTowardClass(job.UserID) {
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
			// statRef is nil for a learner kept out of the class figures; their
			// portfolio still records the attempt in full.
			if statRef != nil {
				stat.Add(outcome.Grade.TotalGrade)
				stat.UpdatedAt = time.Now()
				if err := tx.Set(statRef, stat); err != nil {
					return err
				}
			}
			values["analysis_id"] = outcome.AnalysisID
			values["grading_outcome"] = outcome.Grade
			values["student_video"] = outcome.StudentVideo
			values["feedback_video"] = outcome.FeedbackVideo
			values["skeleton_overlay_video"] = outcome.SkeletonOverlayVideo
			values["expert"] = outcome.Expert
			values["timeline"] = outcome.Timeline
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

// MarkAnalysisJobState finishes a job that owns no portfolio record of its
// own, such as the coaching pass that only merges cues into an existing one.
func (c *FirestoreClient) MarkAnalysisJobState(ctx context.Context, jobID, state string) error {
	_, err := c.AnalysisJobs().Doc(jobID).Update(ctx, []firestore.Update{
		{Path: "status", Value: state},
		{Path: "lease_until", Value: time.Time{}},
	})
	return err
}
