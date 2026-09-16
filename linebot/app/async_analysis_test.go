package app

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/analysisqueue"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
	"github.com/stretchr/testify/require"
)

func TestCloudTasksHeadersDoNotReplaceOIDCAuthentication(t *testing.T) {
	a := &App{AnalysisQueue: &analysisqueue.Client{}, Config: &config.Config{}}
	a.Config.AnalysisServer.WorkerURL = "https://worker.test"
	a.Config.AnalysisServer.TaskServiceAccount = "worker@project.iam.gserviceaccount.com"
	for _, token := range []string{"", "Bearer invalid"} {
		r := httptest.NewRequest("POST", "/internal/analysis/task", strings.NewReader(`{"job_id":"example"}`))
		r.Header.Set("Authorization", token)
		r.Header.Set("X-CloudTasks-TaskName", "forged")
		w := httptest.NewRecorder()
		a.HandleAnalysisTask(w, r)
		require.Equal(t, 401, w.Code)
	}
}
