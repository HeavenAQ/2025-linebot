package gpt_test

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/api/gpt"
	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

var gptClient *gpt.Client

// setup database
func TestMain(m *testing.M) {
	cfg, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load configurations")
	}

	gptClient = gpt.NewGPTClient(cfg.GPT.APIKey, cfg.GPT.AssistantID)

	os.Exit(m.Run())
}
