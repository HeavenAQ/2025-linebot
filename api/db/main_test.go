package db

import (
	"log"
	"os"
	"testing"

	"github.com/HeavenAQ/config"
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
