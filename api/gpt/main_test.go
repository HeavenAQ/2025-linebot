package gpt_test

import (
	"log"
	"os"
	"strings"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

var (
	gptClient      *gpt.Client
	runIntegration bool
)

// setup OpenAI client for integration tests
func TestMain(m *testing.M) {
	cfg, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Println("Skipping GPT integration tests: failed to load config")
		os.Exit(0)
	}

	// Only run if real-looking credentials are provided
	if cfg.GPT.APIKey == "" || strings.HasPrefix(cfg.GPT.APIKey, "test_") {
		log.Println("Skipping GPT integration tests: missing or placeholder credentials")
		os.Exit(0)
	}

	runIntegration = true
	gptClient = gpt.NewGPTClient(cfg.GPT.APIKey, cfg.GPT.PromptID)
	os.Exit(m.Run())
}
