package auth

import (
	"errors"
	"fmt"
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestExperimentGateFailsClosed(t *testing.T) {
	registered := &db.UserData{ID: "verified-line-user", RealName: "王小明", ExperimentNumber: "01",
		RegistrationVersion: 1, RegistrationCompletedAt: time.Now()}
	for _, tc := range []struct {
		name string
		user *db.UserData
		err  error
		code int
	}{
		{"new", nil, fmt.Errorf("lookup: %w", status.Error(codes.NotFound, "absent")), 403},
		{"legacy", &db.UserData{Name: "LINE name"}, nil, 403},
		{"registered", registered, nil, 200},
		{"outage", nil, errors.New("database unavailable"), 503},
		{"permission", nil, status.Error(codes.PermissionDenied, "denied"), 503},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			c.Request = httptest.NewRequest(http.MethodGet, "/api/db/playback?user_id=someone-else", nil)
			allowed := AllowRegisteredLearner(c, "verified-line-user", func(id string) (*db.UserData, error) {
				if id != "verified-line-user" {
					t.Fatal("trusted request-supplied identity")
				}
				return tc.user, tc.err
			})
			if allowed != (tc.code == 200) || recorder.Code != tc.code {
				t.Fatalf("allowed=%v code=%d", allowed, recorder.Code)
			}
			if !allowed && !c.IsAborted() {
				t.Fatal("request not aborted")
			}
		})
	}
}
