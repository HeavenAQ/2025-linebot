package config

import (
	"log"

	env "github.com/Netflix/go-env"
	"github.com/joho/godotenv"
)

type LineConfig struct {
	ChannelSecret string `env:"LINE_CHANNEL_SECRET"`
	ChannelToken  string `env:"LINE_CHANNEL_TOKEN"`
}

type GCPConfig struct {
	ProjectID   string `env:"GCP_PROJECT_ID"`
	Credentials string `env:"GCP_CREDENTIALS"`
	Storage     StorageConfig
	Secrets     SecretManagerConfig
	Database    FirestoreConfig
}

type StorageConfig struct {
	BucketName string `env:"GCS_BUCKET_NAME"`
}
type SecretManagerConfig struct {
	SecretVersion string `env:"GCP_SECRET_VERSION"`
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
		c.GCP.Credentials == "" &&
		c.GCP.Storage.BucketName == "" &&
		c.GCP.Secrets.SecretVersion == "" &&
		c.GCP.Database.DatabaseID == "" &&
		c.GCP.Database.DataDB == "" &&
		c.GCP.Database.SessionDB == "" &&
		c.GPT.APIKey == "")
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
