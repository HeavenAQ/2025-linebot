package auth

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/obs"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"net/http"
)

// Call only after LINE token verification. Never take the identity from a
// request body/query parameter, and never turn a database outage into a pass.
func AllowRegisteredLearner(c *gin.Context, userID string, lookup func(string) (*db.UserData, error)) bool {
	if userID == "" {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return false
	}
	user, err := lookup(userID)
	if err != nil && status.Code(err) != codes.NotFound {
		c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "暫時無法確認登記資料，請稍後再試。"})
		return false
	}
	if user == nil || !user.HasExperimentRegistration() {
		// Says which learner was turned away and why, so a refusal can be traced.
		fields := map[string]any{"user_id": userID, "found": user != nil, "lookup_error": err}
		if user != nil {
			fields["experiment_number"] = user.ExperimentNumber
			fields["registration_version"] = user.RegistrationVersion
			fields["registration_completed"] = !user.RegistrationCompletedAt.IsZero()
		}
		obs.Event(c.Request.Context(), obs.Warning, "registration required", fields)
		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
			"code": "registration_required", "error": db.RegistrationInstructions,
		})
		return false
	}
	return true
}
