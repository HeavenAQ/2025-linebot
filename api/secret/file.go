package secret

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

var ErrMissingProjectID = errors.New("missing GCP project ID for Secret Manager bootstrap")

func DownloadSecretToFile(secretName string, path string) error {
	payload, err := AccessSecretVersion(secretName)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create env file directory: %w", err)
	}

	if err := os.WriteFile(path, payload, 0o600); err != nil {
		return fmt.Errorf("failed to write env file: %w", err)
	}

	return nil
}
