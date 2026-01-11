package app

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
	poseestimation "github.com/HeavenAQ/nstc-linebot-2025/api/pose_estimation"
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
	"github.com/go-resty/resty/v2"
	"github.com/line/line-bot-sdk-go/v7/linebot"
	ffmpeg_go "github.com/u2takey/ffmpeg-go"
)

const tmpFolder = "./tmp/"

type AnalyzedResult struct {
	SkeletonVideo string   `json:"skeleton_video"`
	Score         string   `json:"score"`
	Suggestions   []string `json:"suggestions"`
}

func (app *App) getVideoFolder(user *db.UserData, skill string) string {
	var folderId string
	switch skill {
	case "serve":
		folderId = user.FolderPaths.Serve
	case "smash":
		folderId = user.FolderPaths.Smash
	case "clear":
		folderId = user.FolderPaths.Clear
	}
	return folderId
}

func (app *App) uploadVideoToBucket(user *db.UserData, session *db.UserSession, skeletonVideo []byte, thumbnailPath string, filename string) (*storage.UploadedFile, *storage.UploadedFile, error) {
	app.Logger.Info.Println("Getting folder ID...")
	folderID := app.getVideoFolder(user, session.Skill)

	app.Logger.Info.Printf("Uploading video to folder ID: %v...\n", folderID)
	fileInfo := storage.FileInfo{
		Bucket: struct {
			VideoPath     string
			ThumbnailPath string
		}{},
		Local: struct {
			ThumbnailPath string
			VideoBlob     []byte
		}{
			ThumbnailPath: thumbnailPath,
			VideoBlob:     skeletonVideo,
		},
	}

	// Build Cloud Storage object paths
	// Videos are stored under the skill-specific folder with .mp4 extension
	// Thumbnails are stored under the user's thumbnail folder with .jpeg extension
	fileInfo.Bucket.VideoPath = fmt.Sprintf("%s%s.mp4", folderID, filename)
	fileInfo.Bucket.ThumbnailPath = fmt.Sprintf("%s/%s.jpeg", user.FolderPaths.Thumbnail, filename)

	// Upload video file
	driveFile, err := app.StorageClient.UploadVideo(&fileInfo)
	if err != nil {
		app.Logger.Error.Println("Failed to upload the video")
		return nil, nil, err
	}

	// Upload thumbnail
	thumbnailFile, err := app.StorageClient.UploadThumbnail(&fileInfo)
	if err != nil {
		app.Logger.Error.Println("Failed to upload the thumbnail")
		return nil, nil, err
	}
	return driveFile, thumbnailFile, nil
}

func (app *App) updateUserPortfolioVideo(user *db.UserData, session *db.UserSession, date string, grade commons.GradingOutcome, skeletonFile *storage.UploadedFile, comparisonFile *storage.UploadedFile, thumbnailFile *storage.UploadedFile) error {
	app.Logger.Info.Println("Updating user portfolio:")
	userPortfolio := app.getUserPortfolio(user, session.Skill)

	// Build direct GCS URLs for LINE/LIFF consumption
	bucket := app.Config.GCP.Storage.BucketName
	videoURL := "https://storage.googleapis.com/" + bucket + "/" + skeletonFile.Path
	comparisonURL := "https://storage.googleapis.com/" + bucket + "/" + comparisonFile.Path
	thumbURL := "https://storage.googleapis.com/" + bucket + "/" + thumbnailFile.Path

	urlVideo := &storage.UploadedFile{Name: skeletonFile.Name, Path: videoURL}
	urlComparison := &storage.UploadedFile{Name: comparisonFile.Name, Path: comparisonURL}
	urlThumb := &storage.UploadedFile{Name: thumbnailFile.Name, Path: thumbURL}

	return app.FirestoreClient.CreateUserPortfolioVideo(
		user,
		userPortfolio,
		date,
		session,
		urlVideo,
		urlComparison,
		urlThumb,
		grade,
	)
}

func (app *App) sendVideoUploadedReply(event *linebot.Event, session *db.UserSession, user *db.UserData) error {
	app.Logger.Info.Println("Video uploaded successfully.")

	skill := db.SkillStrToEnum(session.Skill)
	err := app.LineBot.SendPortfolio(
		event,
		user,
		skill,
		session.Handedness,
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
			"vsync":   "0", // avoid audio sync issues
			"threads": "1", // use 1 thread to avoid memory issues
			"an":      "",  // remove audio
			"b:v":     "1M",
		}).
		Run()
	if err != nil {
		return "", errors.New("failed to resize video")
	}

	app.Logger.Info.Println("Video resized successfully.")
	return outputFilename, nil
}

func (app *App) analyzeVideo(videoBlob []byte, skill string, handedness string) (*poseestimation.VideoAnalysisResponse, error) {
	app.Logger.Info.Println("Analyzing video:")

	// set up request body with video data
	url := app.Config.PoseEstimationServer.Host + "/upload"
	app.Logger.Info.Println("Sending video to AI server, URL: " + url)
	client := resty.New()
	client.SetTimeout(1 * time.Minute)

	maxRetries := 6
	delay := 10 * time.Second
	var resp *poseestimation.VideoAnalysisResponse
	var err error
	for i := 0; i < maxRetries; i++ {
		// init client
		client := poseestimation.NewClient(
			app.Config.PoseEstimationServer.User,
			app.Config.PoseEstimationServer.Password,
			url,
			videoBlob,
		)

		// send video to AI server
		resp, err = client.ProcessVideo(skill, handedness)
		if err == nil {
			break
		}

		// retry if failed
		app.Logger.Error.Println("Error processing video:", err)
		app.Logger.Error.Println("Retrying in", delay)
		time.Sleep(delay)
	}

	// Check if we have a valid response
	if err != nil {
		app.Logger.Error.Printf("AI Server Response: %v\n", err.Error())
		return nil, err
	}
	if resp == nil {
		return nil, fmt.Errorf("failed to get response from AI server after %d retries", maxRetries)
	}

	return resp, nil
}

func (app *App) createVideoThumbnail(event *linebot.Event, user *db.UserData, videoPath string) (string, error) {
	// create a tmp file to store video blob
	filename := tmpFolder + user.ID + ".mp4"

	// Using ffmpeg to create video thumbnail
	app.Logger.Info.Println("Extracting thumbnail from the video")
	outFileName := tmpFolder + user.ID + ".jpeg"

	if err := os.Remove(outFileName); !errors.Is(err, os.ErrNotExist) {
		app.Logger.Error.Println("Something goes wrong when removing previous thumbnail")
	}

	var stderr bytes.Buffer
	err := ffmpeg_go.Input(videoPath, ffmpeg_go.KwArgs{
		"ss": "00:00:01", // place ss before input file to avoid seeking issues
	}).
		Output(outFileName, ffmpeg_go.KwArgs{
			"vframes": 1,       // extract exactly 1 frame
			"vcodec":  "mjpeg", // make it a jpeg file
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

func (app *App) stitchVideoWithExpertVideo(user *db.UserData, encodedVideo string, skill string, handedness string) (string, string) {
	// extract video data from base64 string
	videoData, err := base64.StdEncoding.DecodeString(encodedVideo)
	if err != nil {
		app.Logger.Error.Println("Error decoding video data:", err)
		return "", ""
	}

	// save it to a tmp file
	videoPath := tmpFolder + user.ID + "_video.mp4"
	videoFile, err := os.Create(videoPath)
	if err != nil {
		app.Logger.Error.Println("Error creating tmp file for video:", err)
	}
	videoFile.Write(videoData)

	//  run ffmpeg with cmd to stitch the video with expert video
	expertVideoPath := fmt.Sprintf("./pro_videos/pro_%v_%v.mp4", handedness, skill)
	outputPath := tmpFolder + user.ID + "_stitched.mp4"
	cmd := exec.Command(
		"ffmpeg",
		"-i", videoPath,
		"-i", expertVideoPath,
		"-filter_complex",
		"[0:v]scale=-1:960[video1];[1:v]scale=-1:960[video2];[video1][video2]hstack[stacked];[stacked]pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
		"-c:v", "mpeg4", // Use the MPEG-4 codec
		"-q:v", "5", // Adjust quality (lower is better; 2-5 recommended)
		"-pix_fmt", "yuv420p", // Ensure pixel format matches
		"-r", "29.79", // Match the frame rate
		"-metadata:s:v", "rotate=0", // Ensure no rotation metadata
		"-y", // Overwrite the output file
		outputPath,
	) // Capture the output for debugging
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	// Run the command
	err = cmd.Run()
	if err != nil {
		app.Logger.Error.Println("Error stitching video with expert video:", err)
		app.Logger.Error.Println("ffmpeg stderr:", stderr.String())
		return "", videoPath
	}

	app.Logger.Info.Println("Video stitched successfully.")
	return outputPath, videoPath
}

// uploadComparisonVideoToBucket uploads the stitched comparison video to the user's skill folder
func (app *App) uploadComparisonVideoToBucket(user *db.UserData, session *db.UserSession, comparisonVideo []byte, filename string) (*storage.UploadedFile, error) {
	folderID := app.getVideoFolder(user, session.Skill)
	fileInfo := storage.FileInfo{}
	fileInfo.Bucket.VideoPath = fmt.Sprintf("%s%s_comparison.mp4", folderID, filename)
	fileInfo.Local.VideoBlob = comparisonVideo
	videoFile, err := app.StorageClient.UploadVideo(&fileInfo)
	if err != nil {
		app.Logger.Error.Println("Failed to upload the comparison video")
		return nil, err
	}
	return videoFile, nil
}
