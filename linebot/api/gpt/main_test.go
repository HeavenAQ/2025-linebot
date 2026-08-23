package gpt_test

import (
	"log"
	"os"
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
	if os.Getenv("RUN_LIVE_OPENAI") != "1" {
		log.Println("Skipping OpenAI live tests; set RUN_LIVE_OPENAI=1 to enable.")
		// Tests that need no client (prompt building) still run.
		os.Exit(m.Run())
	}
	cfg, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load OpenAI integration-test config")
	}
	if cfg.GPT.APIKey == "" {
		log.Fatal("OPENAI_API_KEY is required for live OpenAI tests")
	}

	runIntegration = true
	gptClient = gpt.NewGPTClient(cfg.GPT.APIKey, cfg.GPT.Model)
	os.Exit(m.Run())
}
