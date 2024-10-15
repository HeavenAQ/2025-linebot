package storage

import (
	"bytes"
	"context"
	"os"

	"google.golang.org/api/drive/v3"
	"google.golang.org/api/option"
)

type GoogleDriveClient struct {
	srv          *drive.Service
	RootFolderID string
}

type UserFolders struct {
	UserID            string
	UserName          string
	RootFolderID      string
	ServeFolderID     string
	SmashFolderID     string
	ClearFolderID     string
	ThumbnailFolderID string
}

func NewGoogleDriveClient(credentials []byte, rootFolderID string) (*GoogleDriveClient, error) {
	ctx := context.Background()

	// init google drive service
	srv, err := drive.NewService(ctx, option.WithCredentialsJSON(credentials))
	if err != nil {
		return nil, err
	}

	return &GoogleDriveClient{
		srv, rootFolderID,
	}, nil
}

func (client *GoogleDriveClient) CreateUserFolders(userID string, userName string) (*UserFolders, error) {
	folderNames := []string{
		userID,
		"serve",
		"smash",
		"clear",
		"thumbnail",
	}

	userFolders := UserFolders{
		UserID:   userID,
		UserName: userName,
	}

	for _, folderName := range folderNames {
		var parents []string
		if folderName == userID {
			parents = []string{client.RootFolderID}
		} else {
			parents = []string{userFolders.RootFolderID}
		}

		folder, err := client.srv.Files.Create(&drive.File{
			Name:     folderName,
			MimeType: "application/vnd.google-apps.folder",
			Parents:  parents,
		}).Do()
		if err != nil {
			return nil, err
		}

		switch folderName {
		case userID:
			userFolders.RootFolderID = folder.Id
		case "serve":
			userFolders.ServeFolderID = folder.Id
		case "smash":
			userFolders.SmashFolderID = folder.Id
		case "clear":
			userFolders.ClearFolderID = folder.Id
		case "thumbnail":
			userFolders.ThumbnailFolderID = folder.Id
		}
	}

	return &userFolders, nil
}

type FileInfo struct {
	Drive struct {
		ParentFolderID    string
		ThumbnailFolderID string
		Filename          string
	}
	Local struct {
		ThumbnailPath string
		VideoBlob     []byte
	}
}

func (client *GoogleDriveClient) UploadVideo(fileInfo *FileInfo) (*drive.File, error) {
	// upload video file to google drive
	blob := bytes.NewReader(fileInfo.Local.VideoBlob)
	driveFile, err := client.srv.Files.Create(&drive.File{
		Name:    fileInfo.Drive.Filename,
		Parents: []string{fileInfo.Drive.ParentFolderID},
	}).Media(blob).Do()
	if err != nil {
		return nil, err
	}

	return driveFile, nil
}

func (client *GoogleDriveClient) UploadThumbnail(fileInfo *FileInfo) (*drive.File, error) {
	// upload video thumbnail to google drive
	thumbnailData, err := os.ReadFile(fileInfo.Local.ThumbnailPath)
	thumbnailFile, err := client.srv.Files.Create(&drive.File{
		Name:    fileInfo.Drive.Filename + "_thumbnail",
		Parents: []string{fileInfo.Drive.ThumbnailFolderID},
	}).Media(bytes.NewReader(thumbnailData)).Do()
	if err != nil {
		return nil, err
	}

	return thumbnailFile, nil
}

func (client *GoogleDriveClient) DeleteFile(fileID string) error {
	err := client.srv.Files.Delete(fileID).Do()
	if err != nil {
		return err
	}
	return nil
}
