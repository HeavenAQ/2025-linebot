package secret

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/config"
)

var cfg *config.Config

// setup database
func TestMain(m *testing.M) {
	conf, err := config.LoadConfig("../../.env")
	if err != nil {
		log.Fatal("Failed to load configurations")
	}
	cfg = conf

	os.Exit(m.Run())
}
