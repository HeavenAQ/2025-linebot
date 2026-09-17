package review

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Config is the service configuration, read from the environment.
type Config struct {
	Port                string
	ProjectID           string
	BucketName          string
	ServiceAccountEmail string
	BatchID             string
	AccessCodes         map[string]string
	// FakeData serves seeded in-memory items instead of Firestore/GCS. Local
	// development only; enabled with LOCAL_FAKE_DATA=1.
	FakeData bool
}

// LoadConfig reads and validates configuration. It reports every missing or
// invalid setting at once so a misconfigured deploy fails fast and clearly.
func LoadConfig(getenv func(string) string) (Config, error) {
	cfg := Config{
		Port:                strings.TrimSpace(getenv("PORT")),
		ProjectID:           strings.TrimSpace(getenv("GCP_PROJECT_ID")),
		BucketName:          strings.TrimSpace(getenv("GCS_BUCKET_NAME")),
		ServiceAccountEmail: strings.TrimSpace(getenv("GCP_SERVICE_ACCOUNT_EMAIL")),
		BatchID:             strings.TrimSpace(getenv("BATCH_ID")),
		FakeData:            strings.TrimSpace(getenv("LOCAL_FAKE_DATA")) == "1",
	}
	if cfg.Port == "" {
		cfg.Port = "8080"
	}

	var problems []string
	if cfg.FakeData {
		if cfg.BatchID == "" {
			cfg.BatchID = "local-fake"
		}
	} else {
		required := []struct{ name, value string }{
			{"GCP_PROJECT_ID", cfg.ProjectID},
			{"GCS_BUCKET_NAME", cfg.BucketName},
			{"GCP_SERVICE_ACCOUNT_EMAIL", cfg.ServiceAccountEmail},
			{"BATCH_ID", cfg.BatchID},
		}
		for _, r := range required {
			if r.value == "" {
				problems = append(problems, r.name+" is required")
			}
		}
	}

	codes, err := ParseAccessCodes(getenv("ACCESS_CODES"))
	if err != nil {
		problems = append(problems, err.Error())
	}
	cfg.AccessCodes = codes

	if len(problems) > 0 {
		return Config{}, errors.New("invalid configuration: " + strings.Join(problems, "; "))
	}
	return cfg, nil
}

// ParseAccessCodes parses the ACCESS_CODES JSON object (expert_id -> code).
func ParseAccessCodes(raw string) (map[string]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("ACCESS_CODES is required")
	}
	var codes map[string]string
	if err := json.Unmarshal([]byte(raw), &codes); err != nil {
		return nil, errors.New("ACCESS_CODES must be a JSON object of expert_id to code strings")
	}
	if len(codes) == 0 {
		return nil, errors.New("ACCESS_CODES has no entries")
	}
	seen := make(map[string]string, len(codes))
	experts := 0
	for id, code := range codes {
		if !validExpertID(id) {
			return nil, fmt.Errorf("ACCESS_CODES has an invalid expert id %q (use letters, digits, _ or -)", id)
		}
		if strings.TrimSpace(code) != code || code == "" {
			return nil, fmt.Errorf("ACCESS_CODES entry %q has an empty code or surrounding whitespace", id)
		}
		if other, dup := seen[code]; dup {
			return nil, fmt.Errorf("ACCESS_CODES entries %q and %q share a code", other, id)
		}
		seen[code] = id
		if id != AdminID {
			experts++
		}
	}
	if experts == 0 {
		return nil, errors.New("ACCESS_CODES has no expert entries")
	}
	return codes, nil
}

func validExpertID(id string) bool {
	if id == "" || len(id) > 64 {
		return false
	}
	for _, c := range id {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
		default:
			return false
		}
	}
	return true
}
