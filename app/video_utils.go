package app

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	ffmpeg_go "github.com/u2takey/ffmpeg-go"
	"google.golang.org/api/drive/v3"
)

const tmpFolder = "/tmp/"

func (app *App) getVideoFolder(user *db.UserData, skill string) string {
	var folderId string
	switch skill {
	case "front_court_footwork":
		folderId = user.FolderIDs.FrontCourtFootwork
	case "drive":
		folderId = user.FolderIDs.Drive
	case "back_court_footwork":
		folderId = user.FolderIDs.BackCourtFootwork
	case "smash":
		folderId = user.FolderIDs.Smash
	case "netkill":
		folderId = user.FolderIDs.Netkill
	case "doubles_rotation":
		folderId = user.FolderIDs.DoublesRotation
	}
	return folderId
}

func (app *App) uploadVideoToDrive(user *db.UserData, session *db.UserSession, videoBlob []byte, thumbnailPath string, filename string) (*drive.File, *drive.File, error) {
	app.Logger.Info.Println("Getting folder ID...")
	folderID := app.getVideoFolder(user, session.Skill)

	app.Logger.Info.Printf("Uploading video to folder ID: %v...\n", folderID)
	fileInfo := storage.FileInfo{
		Drive: struct {
			ParentFolderID    string
			ThumbnailFolderID string
			Filename          string
		}{
			ParentFolderID:    folderID,
			ThumbnailFolderID: user.FolderIDs.Thumbnail,
			Filename:          fmt.Sprintf("%v.mp4", filename),
		},
		Local: struct {
			ThumbnailPath string
			VideoBlob     []byte
		}{
			ThumbnailPath: thumbnailPath,
			VideoBlob:     videoBlob,
		},
	}

	// Upload file
	driveFile, err := app.DriveClient.UploadVideo(&fileInfo)
	if err != nil {
		app.Logger.Error.Println("Failed to upload the video")
		return nil, nil, err
	}

	// Upload thumbnail
	thumbnailFile, err := app.DriveClient.UploadThumbnail(&fileInfo)
	if err != nil {
		app.Logger.Error.Println("Failed to upload the thumbnail")
		return nil, nil, err
	}
	return driveFile, thumbnailFile, nil
}

func (app *App) updateUserPortfolioVideo(user *db.UserData, session *db.UserSession, date string, driveFile *drive.File, thumbnailFile *drive.File) error {
	app.Logger.Info.Println("Updating user portfolio:")
	userPortfolio := app.getUserPortfolio(user, session.Skill)

	return app.FirestoreClient.CreateUserPortfolioVideo(
		user,
		userPortfolio,
		date,
		session,
		driveFile,
		thumbnailFile,
	)
}

func (app *App) sendVideoUploadedReply(event *linebot.Event, session *db.UserSession, user *db.UserData) error {
	app.Logger.Info.Println("Video uploaded successfully.")

	skill := db.SkillStrToEnum(session.Skill)
	err := app.LineBot.SendPortfolio(
		event,
		user,
		skill,
		session.UserState,
		"影片已成功上傳！",
		true,
	)
	return err
}

func (app *App) createTmpVideoFile(blob io.Reader, user *db.UserData) (string, error) {
	filename := tmpFolder + user.ID + ".mp4"
	file, err := os.Create(filename)
	if err != nil {
		return "", errors.New("failed to create tmp file for resizing")
	}
	defer file.Close()

	// Stream the video directly to disk to avoid memory duplication
	app.Logger.Info.Println("Copying video blob to disk")
	if _, err := io.Copy(file, blob); err != nil {
		return "", errors.New("failed to write video blob to disk")
	}

	return filename, nil
}

func (app App) rmTmpVideoFile(filename string) {
	app.Logger.Info.Println("Removing tmp video file")
	if err := os.Remove(filename); err != nil {
		app.Logger.Warn.Println("Failed to remove tmp video file:", err)
	}
}

func (app App) resizeVideo(user *db.UserData, videoPath string) (string, error) {
	// Use ffmpeg-go to resize the video
	app.Logger.Info.Println("Start Resizing video:")
	app.Logger.Info.Println("Resizing video")
	outputFilename := tmpFolder + "resized_" + user.ID + ".mp4"
	err := ffmpeg_go.Input(videoPath).
		Filter("scale", ffmpeg_go.Args{"1080:1920"}).
		Output(outputFilename, ffmpeg_go.KwArgs{
			"vsync":   "0",  // avoid audio sync issues
			"threads": "1",  // use 1 thread to avoid memory issues
			"b:v":     "1M", // set video bitrate to 1 Mbps
			"an":      "",   // remove audio
		}).
		Run()
	if err != nil {
		return "", errors.New("failed to resize video")
	}

	app.Logger.Info.Println("Video resized successfully.")
	return outputFilename, nil
}

func (app *App) createVideoThumbnail(event *linebot.Event, user *db.UserData, blob []byte) (string, error) {
	// create a tmp file to store video blob
	app.Logger.Info.Println("Creating a tmp file to store video blob ...")
	replyToken := event.ReplyToken
	filename := tmpFolder + user.ID + ".mp4"
	file, err := os.Create(filename)
	if err != nil {
		app.Logger.Error.Println("Error creating tmp file for video:", err)
		app.handleThumbnailCreationError(err, replyToken)
		return "", err
	}
	defer file.Close()

	// write video blob to the tmp file
	app.Logger.Info.Println("Writing video blob to tmp file")
	if _, err := io.Copy(file, bytes.NewReader(blob)); err != nil {
		app.Logger.Error.Println("Error writing video blob to tmp file:", err)
		_, err := app.LineBot.SendDefaultErrorReply(replyToken)
		app.handleThumbnailCreationError(err, replyToken)
		return "", err
	}

	// Using ffmpeg to create video thumbnail
	app.Logger.Info.Println("Extracting thumbnail from the video")
	outFileName := tmpFolder + user.ID + ".jpeg"

	var stderr bytes.Buffer
	err = ffmpeg_go.Input(filename, ffmpeg_go.KwArgs{
		"ss": "00:00:01", // place ss before input file to avoid seeking issues
	}).
		Output(outFileName, ffmpeg_go.KwArgs{
			"vframes": 1,              // extract exactly 1 frame
			"vcodec":  "mjpeg",        // make it a jpeg file
			"vf":      "scale=320:-1", // scale the image to 320px width, keep aspect ratio
		}).
		WithErrorOutput(&stderr). // Capture stderr for debugging
		Run()
	if err != nil {
		app.Logger.Error.Println("Error extracting thumbnail from video:", err)
		app.Logger.Error.Println("ffmpeg stderr:", stderr.String())
		return "", err
	}

	// Asynchronously remove the original file
	go func() {
		if err := os.Remove(filename); err != nil {
			app.Logger.Info.Println("Failed to remove temp file:", err)
		}
	}()
	return outFileName, nil
}

func uploadError(app App, event *linebot.Event, err error, message string) {
	app.Logger.Error.Println(message, err)
	app.LineBot.SendDefaultErrorReply(event.ReplyToken)
}
