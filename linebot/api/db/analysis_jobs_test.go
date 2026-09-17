package db

import (
	"context"
	"os"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
	"google.golang.org/api/option"
)

func TestAsyncWorkKeysPreserveTimestamp(t *testing.T) {
	tm, err := ParseWorkTime("2026-09-16-14-30-12-abc12345")
	require.NoError(t, err)
	require.Equal(t, 12, tm.Second())
}

// Creates only uniquely named test records and removes those exact records.
func TestLiveAsyncJobTransactions(t *testing.T) {
	if os.Getenv("RUN_LIVE_ASYNC_JOBS") != "1" {
		t.Skip("live Firestore opt-in")
	}
	ctx := context.Background()
	fc, err := firestore.NewClient(ctx, "nstc-linebot-2025", option.WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: os.Getenv("TEST_GCP_ACCESS_TOKEN")})))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, fc.Close()) })
	id := "async-test-" + time.Now().Format("20060102T150405.000000000")
	c := &FirestoreClient{Ctx: &ctx, Client: fc, Data: fc.Collection("async_integration_users")}
	userRef := c.Data.Doc(id)
	_, err = userRef.Set(ctx, UserData{ID: id, Portfolio: Portfolios{Smash: map[string]Work{}}})
	require.NoError(t, err)
	t.Cleanup(func() {
		_, e := userRef.Delete(ctx)
		require.NoError(t, e)
		_, e = c.AnalysisJobs().Doc(id).Delete(ctx)
		require.NoError(t, e)
		_, e = c.ClassStats().Doc(classStatID("smash", "2026-09-16")).Delete(ctx)
		require.NoError(t, e)
	})
	job := AnalysisJob{ID: id, UserID: id, Skill: "smash", WorkDate: "2026-09-16-14-00-00-test", Status: "queued", CreatedAt: time.Now()}
	created, err := c.CreateAnalysisJob(ctx, job)
	require.NoError(t, err)
	job.WorkDate = "different"
	duplicate, err := c.CreateAnalysisJob(ctx, job)
	require.NoError(t, err)
	require.Equal(t, created.WorkDate, duplicate.WorkDate)
	lease, err := c.ClaimAnalysisJob(ctx, id)
	require.NoError(t, err)
	_, err = c.ClaimAnalysisJob(ctx, id)
	require.ErrorIs(t, err, ErrJobBusy)
	_, err = userRef.Update(ctx, []firestore.Update{{FieldPath: []string{"portfolio", "smash", created.WorkDate, "reflection"}, Value: "learner note"}})
	require.NoError(t, err)
	require.NoError(t, c.FinishAnalysisJob(ctx, lease, &commons.AnalysisOutcome{AnalysisID: "result", Grade: commons.GradingOutcome{TotalGrade: 88}}, "", false))
	finished, err := c.ClaimAnalysisJob(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "completed", finished.Status)
	doc, err := userRef.Get(ctx)
	require.NoError(t, err)
	var user UserData
	require.NoError(t, doc.DataTo(&user))
	require.Equal(t, "learner note", user.Portfolio.Smash[created.WorkDate].Reflection)
	require.Equal(t, 88.0, user.Portfolio.Smash[created.WorkDate].GradingOutcome.TotalGrade)

	// Completion joins the class aggregate exactly once, even if finish is retried.
	require.NoError(t, c.FinishAnalysisJob(ctx, lease, &commons.AnalysisOutcome{AnalysisID: "result", Grade: commons.GradingOutcome{TotalGrade: 88}}, "", false))
	stats, err := c.GetClassSkillStats("smash")
	require.NoError(t, err)
	require.Equal(t, Stats{Avg: 88, Max: 88, Min: 88, Std: 0}, stats["2026-09-16"])

	rebuilt, err := c.RebuildClassStats(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, rebuilt)
	stats, err = c.GetClassSkillStats("smash")
	require.NoError(t, err)
	require.Equal(t, 88.0, stats["2026-09-16"].Avg)
}
