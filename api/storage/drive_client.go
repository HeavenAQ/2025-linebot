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
	UserID                     string
	UserName                   string
	RootFolderID               string
	SmashFolderID              string
	DriveFolderID              string
	NetkillFolderID            string
	FrontCourtFootworkFolderID string
	BackCourtFootworkFolderID  string
	DoublesRotationFolderID    string
	ThumbnailFolderID          string
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

func (client *GoogleDriveClient) createFolder(
	folderName string,
	parents []string,
) (folderID string, err error) {
	folder, err := client.srv.Files.Create(&drive.File{
		Name:     folderName,
		MimeType: "application/vnd.google-apps.folder",
		Parents:  parents,
	}).Do()
	if err != nil {
		return "", err
	}

	return folder.Id, nil
}

func (client *GoogleDriveClient) asyncCreateFolder(folderName string, parents []string, idChannel chan<- string, errorChannel chan<- error) {
	// Attempt folder creation
	folderID, err := client.createFolder(folderName, parents)
	if err != nil {
		select {
		case errorChannel <- err:
		default:
		}
		return
	}

	// Send successful folder ID
	idChannel <- folderID
}

func (client *GoogleDriveClient) asyncCreateFolders(userID string, folderNames []string, userFolders *UserFolders) (<-chan string, <-chan error) {
	idChannel := make(chan string, len(folderNames))
	errorChannel := make(chan error, 1)

	for i, folderName := range folderNames {
		go func(i int, folderName string) {
			// Determine parent folder
			var parents []string
			if folderName == userID {
				parents = []string{client.RootFolderID}
			} else {
				parents = []string{userFolders.RootFolderID}
			}

			// Attempt folder creation
			client.asyncCreateFolder(folderName, parents, idChannel, errorChannel)
		}(i, folderName)
	}
	return idChannel, errorChannel
}

func (client *GoogleDriveClient) checkAsyncFolderCreation(idChannel <-chan string, errorChannel <-chan error, userFolders *UserFolders) error {
	userFolderIDAddrs := []*string{
		&userFolders.FrontCourtFootworkFolderID,
		&userFolders.BackCourtFootworkFolderID,
		&userFolders.SmashFolderID,
		&userFolders.NetkillFolderID,
		&userFolders.DriveFolderID,
		&userFolders.DoublesRotationFolderID,
		&userFolders.ThumbnailFolderID,
	}

	for i := range userFolderIDAddrs {
		select {
		case err := <-errorChannel:
			if err != nil {
				return err
			}
		case res := <-idChannel:
			*userFolderIDAddrs[i] = res
		}
	}
	return nil
}

func (client *GoogleDriveClient) CreateUserFolders(userID string, userName string) (*UserFolders, error) {
	folderNames := []string{
		"Front Court Footwork",
		"Back Court Footwork",
		"Smash",
		"Netkill",
		"Drive",
		"Doubles Rotation",
		"Thumbnail",
	}

	// Initialize user folders struct
	userFolders := UserFolders{
		UserID:                     userID,
		UserName:                   userName,
		SmashFolderID:              "",
		DriveFolderID:              "",
		RootFolderID:               "",
		NetkillFolderID:            "",
		FrontCourtFootworkFolderID: "",
		BackCourtFootworkFolderID:  "",
		DoublesRotationFolderID:    "",
		ThumbnailFolderID:          "",
	}

	// Create user's root folder first
	var userRootFolder string
	userRootFolder, err := client.createFolder(userID, []string{client.RootFolderID})
	if err != nil {
		return nil, err
	}
	userFolders.RootFolderID = userRootFolder

	// Create folders in Google Drive concurrently
	idChannel, errChannel := client.asyncCreateFolders(userID, folderNames, &userFolders)
	err = client.checkAsyncFolderCreation(idChannel, errChannel, &userFolders)
	if err != nil {
		return nil, err
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
