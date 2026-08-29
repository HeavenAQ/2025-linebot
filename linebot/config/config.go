package config

import (
	"log"
	"strings"

	env "github.com/Netflix/go-env"
	"github.com/joho/godotenv"
)

type LineConfig struct {
	ChannelSecret string `env:"LINE_CHANNEL_SECRET"`
	ChannelToken  string `env:"LINE_CHANNEL_TOKEN"`
	// LoginChannelID is the LINE Login channel the LIFF app belongs to. ID
	// tokens name it as their audience, so it is what proves a token was
	// issued for this app and not replayed from another.
	LoginChannelID string `env:"LINE_LOGIN_CHANNEL_ID"`
}

type GCPConfig struct {
	ProjectID   string `env:"GCP_PROJECT_ID"`
	Credentials string `env:"GCP_CREDENTIALS"`
	// ServiceAccountEmail names the signer for playback URLs. On Cloud Run the
	// metadata credentials carry no private key, so signing goes through IAM
	// and needs to be told which account it is signing as.
	ServiceAccountEmail string `env:"GCP_SERVICE_ACCOUNT_EMAIL"`
	Storage             StorageConfig
	Secrets             SecretManagerConfig
	Database            FirestoreConfig
}

type StorageConfig struct {
	BucketName string `env:"GCS_BUCKET_NAME"`
}
type SecretManagerConfig struct {
	SecretVersion string `env:"GCP_SECRET_VERSION"`
}

type FirestoreConfig struct {
	DataDB    string `env:"FIREBASE_DATA_DB"`
	SessionDB string `env:"FIREBASE_SESSION_DB"`
}

type GPTConfig struct {
	APIKey string `env:"OPENAI_API_KEY"`
	// Model overrides gpt.DefaultModel. The bot no longer runs off a stored
	// OpenAI prompt, which pinned its own model and took the feature down when
	// that model was retired; the model and the system prompts are both in the
	// code now.
	Model string `env:"OPENAI_MODEL"`
}

type AnalysisServerConfig struct {
	Target   string `env:"ANALYSIS_GRPC_TARGET"`
	APIKey   string `env:"ANALYSIS_GRPC_API_KEY"`
	Insecure bool   `env:"ANALYSIS_GRPC_INSECURE"`
	// SkipCoaching turns off the one stage that leaves the analysis service:
	// coaching uploads sampled frames of the learner to a third-party model.
	// A deployment without consent for that sets it, and no image of a learner
	// is sent anywhere. Grading is unaffected -- it is entirely local.
	SkipCoaching bool `env:"ANALYSIS_SKIP_COACHING"`
	// StoragePrefix names this deployment inside the analysis service's
	// bucket. Deployments that share that service share its bucket, and they
	// share a LINE login channel too, so learner ids are identical in both and
	// cannot tell their recordings apart. Empty keeps the original unprefixed
	// layout, which is what the first deployment already has stored.
	StoragePrefix string `env:"ANALYSIS_STORAGE_PREFIX"`
}

// PreviewConfig guards the weekly 課前預習 push. The token is required: the
// endpoint messages every student on the roster, so it must not be callable by
// anyone who happens to find the URL. Leaving it unset disables the endpoint.
type PreviewConfig struct {
	PushToken string `env:"WEEKLY_PREVIEW_TOKEN"`
}

// LiffConfig points the bot at the web app. Reflections are written there now,
// so the bot has to be able to send learners to the right tab.
type LiffConfig struct {
	ReviewURL string `env:"LIFF_REVIEW_URL"`
}

// DefaultLiffReviewURL is the deployed review tab, used when nothing is set so
// the bot never hands a learner a broken link.
const DefaultLiffReviewURL = "https://linebot-liff-nstc-2025.heavian.work/personal?tab=review"

// ReviewURL is where learners write their weekly reflection.
func (c *Config) ReviewURL() string {
	if url := strings.TrimSpace(c.Liff.ReviewURL); url != "" {
		return url
	}
	return DefaultLiffReviewURL
}

type Config struct {
	Port           string `env:"PORT"`
	Line           LineConfig
	GCP            GCPConfig
	GPT            GPTConfig
	AnalysisServer AnalysisServerConfig
	Preview        PreviewConfig
	Liff           LiffConfig
}

func (c *Config) isConfigEmpty() bool {
	return (c.Port == "" &&
		c.Line.ChannelSecret == "" &&
		c.Line.ChannelToken == "" &&
		c.GCP.ProjectID == "" &&
		c.GCP.Credentials == "" &&
		c.GCP.Storage.BucketName == "" &&
		c.GCP.Secrets.SecretVersion == "" &&
		c.GCP.Database.DataDB == "" &&
		c.GCP.Database.SessionDB == "" &&
		c.GPT.APIKey == "" &&

		c.AnalysisServer.Target == "" &&
		c.AnalysisServer.APIKey == "")
}

func LoadConfig(path string) (*Config, error) {
	// try to load .env file
	err := godotenv.Load(path)
	if err != nil {
		// if error, log and continue without .env file
		log.Println("Error loading .env file")
		log.Println("Starting without .env file")
	}

	// unmarshal config from environment variables
	var config Config
	if _, err := env.UnmarshalFromEnviron(&config); err != nil {
		log.Panic("Error loading config")
	}
	if config.isConfigEmpty() {
		return nil, err
	}
	return &config, nil
}
