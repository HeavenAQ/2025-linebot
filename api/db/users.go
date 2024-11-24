package db

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	googleDrive "google.golang.org/api/drive/v3"
)

type UserData struct {
	Portfolio    Portfolios   `json:"portfolio"`
	FolderIDs    FolderIDs    `json:"folderIDs"`
	GPTThreadIDs GPTThreadIDs `json:"gptThreadIDs"`
	Name         string       `json:"name"`
	ID           string       `json:"id"`
	Handedness   Handedness   `json:"handedness"`
}

type FolderIDs struct {
	Root               string `json:"root"`
	Smash              string `json:"smash"`
	Drive              string `json:"drive"`
	Netkill            string `json:"netkill"`
	FrontCourtFootwork string `json:"front_court_footwork"`
	BackCourtFootwork  string `json:"back_court_footwork"`
	DoublesRotation    string `json:"doubles_rotation"`
	Thumbnail          string `json:"thumbnail"`
}

type Portfolios struct {
	Smash              map[string]Work `json:"smash"`
	Drive              map[string]Work `json:"drive"`
	Netkill            map[string]Work `json:"netkill"`
	FrontCourtFootwork map[string]Work `json:"front_court_footwork"`
	BackCourtFootwork  map[string]Work `json:"back_court_footwork"`
	DoublesRotation    map[string]Work `json:"double_rotation"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "smash":
		return p.Smash
	case "drive":
		return p.Drive
	case "netkill":
		return p.Netkill
	case "front_court_footwork":
		return p.FrontCourtFootwork
	case "footwork":
		return p.BackCourtFootwork
	case "doubles_rotation":
		return p.DoublesRotation
	default:
		return nil
	}
}

type GPTThreadIDs struct {
	DoublesRotation string `json:"doubles_rotation"`
}

type Work struct {
	DateTime   string `json:"date"`
	Thumbnail  string `json:"thumbnail"`
	Video      string `json:"video"`
	Reflection string `json:"reflection"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders, gptThreads *GPTThreadIDs) (*UserData, error) {
	ref := client.Data.Doc(userFolders.UserID)
	newUserTemplate := &UserData{
		Name:       userFolders.UserName,
		ID:         userFolders.UserID,
		Handedness: Right,
		FolderIDs: FolderIDs{
			Root:               userFolders.RootFolderID,
			Smash:              userFolders.SmashFolderID,
			Drive:              userFolders.DriveFolderID,
			Netkill:            userFolders.NetkillFolderID,
			FrontCourtFootwork: userFolders.FrontCourtFootworkFolderID,
			BackCourtFootwork:  userFolders.BackCourtFootworkFolderID,
			DoublesRotation:    userFolders.DoublesRotationFolderID,
			Thumbnail:          userFolders.ThumbnailFolderID,
		},
		Portfolio: Portfolios{
			Smash:              map[string]Work{},
			Drive:              map[string]Work{},
			Netkill:            map[string]Work{},
			FrontCourtFootwork: map[string]Work{},
			BackCourtFootwork:  map[string]Work{},
			DoublesRotation:    map[string]Work{},
		},
		GPTThreadIDs: GPTThreadIDs{
			DoublesRotation: gptThreads.DoublesRotation,
		},
	}

	_, err := ref.Set(*client.Ctx, newUserTemplate)
	if err != nil {
		return nil, fmt.Errorf("error creating user data: %w", err)
	}
	return newUserTemplate, nil
}

func (client *FirestoreClient) GetUserData(userID string) (*UserData, error) {
	docsnap, err := client.Data.Doc(userID).Get(*client.Ctx)
	if err != nil {
		return nil, fmt.Errorf("error getting user data: %w", err)
	}
	user := &UserData{}
	err = docsnap.DataTo(user)
	if err != nil {
		return nil, fmt.Errorf("error converting user data: %w", err)
	}

	return user, nil
}

func (client *FirestoreClient) updateUserData(user *UserData) error {
	_, err := client.Data.Doc(user.ID).Set(*client.Ctx, *user)
	if err != nil {
		return fmt.Errorf("error updating user data: %w", err)
	}
	return nil
}

func (client *FirestoreClient) UpdateUserHandedness(user *UserData, handedness Handedness) error {
	user.Handedness = handedness
	return client.updateUserData(user)
}

func (client *FirestoreClient) CreateUserPortfolioVideo(user *UserData, userPortfolio *map[string]Work, date string, session *UserSession, driveFile *googleDrive.File, thumbnailFile *googleDrive.File) error {
	id := driveFile.Id
	work := Work{
		DateTime:   date,
		Reflection: "尚未填寫心得",
		Video:      id,
		Thumbnail:  thumbnailFile.Id,
	}
	(*userPortfolio)[date] = work
	err := client.UpdateUserSession(user.ID, *session)
	if err != nil {
		return fmt.Errorf("error updating user session: %w", err)
	}

	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserPortfolioReflection(
	user *UserData,
	userPortfolio *map[string]Work,
	date string,
	reflection string,
) error {
	targetWork := (*userPortfolio)[date]
	work := Work{
		DateTime:   targetWork.DateTime,
		Reflection: reflection,
		Video:      targetWork.Video,
		Thumbnail:  targetWork.Thumbnail,
	}
	(*userPortfolio)[date] = work

	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadID(user *UserData, skill string, threadID string) error {
	switch skill {
	case "strategy":
		user.GPTThreadIDs.DoublesRotation = threadID
	}
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadIDs(user *UserData, threadIDs *GPTThreadIDs) error {
	user.GPTThreadIDs = *threadIDs
	return client.updateUserData(user)
}
