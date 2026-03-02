package config

import (
	"log"
	"os"

	"github.com/HeavenAQ/nstc-linebot-2025/api/secret"
	env "github.com/Netflix/go-env"
	"github.com/joho/godotenv"
)

type LineConfig struct {
	ChannelSecret string `env:"LINE_CHANNEL_SECRET"`
	ChannelToken  string `env:"LINE_CHANNEL_TOKEN"`
}

type GCPConfig struct {
	ProjectID string `env:"GCP_PROJECT_ID"`
	Storage   StorageConfig
	Secrets   SecretManagerConfig
	Database  FirestoreConfig
}

type StorageConfig struct {
	BucketName string `env:"GCS_BUCKET_NAME"`
}
type SecretManagerConfig struct {
	EnvFileSecretID      string `env:"GCP_ENV_SECRET_ID"`
	EnvFileSecretVersion string `env:"GCP_ENV_SECRET_VERSION"`
}

type FirestoreConfig struct {
	DatabaseID string `env:"FIREBASE_ID"`
	DataDB     string `env:"FIREBASE_DATA_DB"`
	SessionDB  string `env:"FIREBASE_SESSION_DB"`
}

type GPTConfig struct {
	APIKey   string `env:"OPENAI_API_KEY"`
	PromptID string `env:"OPENAI_PROMPT_ID"`
}

type Config struct {
	Port string `env:"PORT"`
	Line LineConfig
	GCP  GCPConfig
	GPT  GPTConfig
}

func (c *Config) isConfigEmpty() bool {
	return (c.Port == "" &&
		c.Line.ChannelSecret == "" &&
		c.Line.ChannelToken == "" &&
		c.GCP.ProjectID == "" &&
		c.GCP.Storage.BucketName == "" &&
		c.GCP.Database.DatabaseID == "" &&
		c.GCP.Database.DataDB == "" &&
		c.GCP.Database.SessionDB == "" &&
		c.GPT.APIKey == "")
}

func LoadConfig(path string) (*Config, error) {
	if err := BootstrapEnvFile(path); err != nil {
		return nil, err
	}

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

func BootstrapEnvFile(path string) error {
	secretID := firstNonEmptyEnv("GCP_ENV_SECRET_ID")
	if secretID == "" {
		return nil
	}

	projectID := firstNonEmptyEnv("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT")
	if projectID == "" {
		return secret.ErrMissingProjectID
	}

	version := firstNonEmptyEnv("GCP_ENV_SECRET_VERSION")
	if version == "" {
		version = "latest"
	}

	if path == "" {
		path = ".env"
	}

	secretName := secret.GetSecretString(projectID, secretID, version)
	return secret.DownloadSecretToFile(secretName, path)
}

func firstNonEmptyEnv(keys ...string) string {
	for _, key := range keys {
		if value := os.Getenv(key); value != "" {
			return value
		}
	}

	return ""
}
