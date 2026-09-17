package app

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
)

const tmpFolder = "/tmp/"
const thumbnailStorageRoot = "no-ai/analyses/thumbnail"

func thumbnailObjectPath(userID, timestamp string) string {
	return fmt.Sprintf("%s/%s/%s.jpeg", thumbnailStorageRoot, userID, timestamp)
}

func (app *App) createVideoThumbnail(video []byte, userID string) (string, error) {
	directory, err := os.MkdirTemp(tmpFolder, "linebot-thumbnail-")
	if err != nil {
		return "", err
	}
	input := filepath.Join(directory, userID+".mp4")
	output := filepath.Join(directory, userID+".jpeg")
	if err := os.WriteFile(input, video, 0600); err != nil {
		return "", err
	}
	command := exec.Command(
		"ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
		"-ss", "00:00:01", "-i", input, "-frames:v", "1", "-q:v", "3", output,
	)
	if data, err := command.CombinedOutput(); err != nil {
		return "", fmt.Errorf("create thumbnail: %w: %s", err, string(data))
	}
	return output, nil
}

func (app *App) uploadThumbnail(
	user *db.UserData, thumbnailPath, timestamp string,
) (*storage.UploadedFile, error) {
	fileInfo := storage.FileInfo{}
	fileInfo.Bucket.ThumbnailPath = thumbnailObjectPath(user.ID, timestamp)
	fileInfo.Local.ThumbnailPath = thumbnailPath
	return app.StorageClient.UploadThumbnail(&fileInfo)
}
