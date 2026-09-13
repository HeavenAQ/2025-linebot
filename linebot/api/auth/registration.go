package auth

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
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
		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
			"code": "registration_required", "error": db.RegistrationInstructions,
		})
		return false
	}
	return true
}
