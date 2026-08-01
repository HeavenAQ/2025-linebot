package app

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	linebotsdk "github.com/line/line-bot-sdk-go/v7/linebot"
)

const tmpFolder = "/tmp/"

func (app *App) analyzeVideo(
	video []byte,
	requestID, userID, skill, handedness string,
) (*commons.AnalysisOutcome, error) {
	app.Logger.Info.Printf(
		"streaming video to analysis service request_id=%s skill=%s bytes=%d",
		requestID,
		skill,
		len(video),
	)
	return app.AnalysisClient.AnalyzeVideo(
		context.Background(), requestID, userID, "line-upload.mp4", skill, handedness, video,
	)
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
	fileInfo.Bucket.ThumbnailPath = fmt.Sprintf("%s/%s.jpeg", user.FolderPaths.Thumbnail, timestamp)
	fileInfo.Local.ThumbnailPath = thumbnailPath
	return app.StorageClient.UploadThumbnail(&fileInfo)
}

func (app *App) updateUserPortfolioVideo(
	user *db.UserData,
	session *db.UserSession,
	date string,
	analysis commons.AnalysisOutcome,
	thumbnail *storage.UploadedFile,
) error {
	portfolio := app.getUserPortfolio(user, session.Skill)
	thumbnailURL := "https://storage.googleapis.com/" + app.Config.GCP.Storage.BucketName + "/" + thumbnail.Path
	return app.FirestoreClient.CreateUserPortfolioVideo(
		user,
		portfolio,
		date,
		session,
		&storage.UploadedFile{Name: thumbnail.Name, Path: thumbnailURL},
		analysis,
	)
}

func (app *App) sendVideoUploadedReply(
	event *linebotsdk.Event, session *db.UserSession, user *db.UserData,
) error {
	return app.LineBot.SendPortfolio(
		event,
		user,
		db.SkillStrToEnum(session.Skill),
		session.Handedness,
		session.UserState,
		"影片分析完成，已加入學習歷程。",
		true,
	)
}
