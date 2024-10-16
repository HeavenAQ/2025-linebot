package utils

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

func SetRandomEnv(t *testing.T) {
	t.Helper()
	os.Clearenv()

	// Set environment variables for the test
	t.Setenv("LINE_CHANNEL_SECRET", "test_line_channel_secret")
	t.Setenv("LINE_CHANNEL_TOKEN", "test_line_channel_token")
	t.Setenv("GCP_PROJECT_ID", "test_gcp_project_id")
	t.Setenv("GCP_CREDENTIALS", "test_gcp_credentials")
	t.Setenv("GOOGLE_DRIVE_ROOT_FOLDER", "test_google_drive_root_folder")
	t.Setenv("SECRET_VERSION", "test_secret_version")
	t.Setenv("FIREBASE_DATA_DB", "test_firebase_data_db")
	t.Setenv("FIREBASE_SESSION_DB", "test_firebase_session_db")
	t.Setenv("OPENAI_API_KEY", "test_openai_api_key")
	t.Setenv("OPENAI_ASSISTANT_ID", "test_openai_assistant_id")
	t.Setenv("POSE_ESTIMATION_SERVER_HOST", "test_pose_estimation_server_host")
	t.Setenv("POSE_ESTIMATION_SERVER_USER", "test_pose_estimation_server_user")
	t.Setenv("POSE_ESTIMATION_SERVER_PASSWORD", "test_pose_estimation_server_password")
	t.Setenv("PORT", "8080")
}

const (
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	digits   = "0123456789"
)

func RandomInt(min, max int64) int64 {
	return min + rand.Int63n(max-min+1)
}

func RandomFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func RandomAlphabetString(n int) string {
	var sb strings.Builder
	k := len(alphabet)

	for i := 0; i < n; i++ {
		c := alphabet[rand.Intn(k)]
		sb.WriteByte(c)
	}

	return sb.String()
}

func RandomUserName() string {
	return RandomAlphabetString(6)
}

func RandomPrice() int64 {
	return RandomInt(0, 1000)
}

func RandomDiscount() int64 {
	return RandomInt(0, 100)
}

func RandomLanguage() string {
	currencies := []string{"chn", "jp"}
	return currencies[rand.Intn(len(currencies))]
}

func RandomNumberString(n int) string {
	var sb strings.Builder
	k := len(digits)

	for i := 0; i < n; i++ {
		c := digits[rand.Intn(k)]
		sb.WriteByte(c)
	}

	return sb.String()
}
