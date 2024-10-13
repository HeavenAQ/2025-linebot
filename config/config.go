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

type GoogleDriveConfig struct {
	Credentials string `env:"GOOGLE_DRIVE_CREDENTIALS"`
	RootFolder  string `env:"GOOGLE_DRIVE_ROOT_FOLDER"`
}

type FirebaseConfig struct {
	Credentials string `env:"FIREBASE_CREDENTIALS"`
	ProjectID   string `env:"FIREBASE_PROJECT_ID"`
	DataDB      string `env:"FIREBASE_DATA_DB"`
	SessionDB   string `env:"FIREBASE_SESSION_DB"`
}

type GPTConfig struct {
	APIKey      string `env:"OPENAI_API_KEY"`
	AssistantID string `env:"OPENAI_ASSISTANT_ID"`
}

type PoseEstimationServerConfig struct {
	Host     string `env:"POSE_ESTIMATION_SERVER_HOST"`
	User     string `env:"POSE_ESTIMATION_SERVER_USER"`
	Password string `env:"POSE_ESTIMATION_SERVER_PASSWORD"`
}

type Config struct {
	Line        LineConfig
	GoogleDrive GoogleDriveConfig
	Firebase    FirebaseConfig
	GPT         GPTConfig
}

func LoadConfig() (Config, error) {
	// try to load .env file
	err := godotenv.Load()
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
	return config, nil
}
