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
	Root      string `json:"root"`
	Lift      string `json:"lift"`
	Drop      string `json:"drop"`
	Netplay   string `json:"netplay"`
	Clear     string `json:"clear"`
	Footwork  string `json:"footwork"`
	Strategy  string `json:"strategy"`
	Thumbnail string `json:"thumbnail"`
}

type Portfolios struct {
	Lift     map[string]Work `json:"lift"`
	Drop     map[string]Work `json:"drop"`
	Netplay  map[string]Work `json:"netplay"`
	Clear    map[string]Work `json:"clear"`
	Footwork map[string]Work `json:"footwork"`
	Strategy map[string]Work `json:"strategy"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "lift":
		return p.Lift
	case "drop":
		return p.Drop
	case "netplay":
		return p.Netplay
	case "clear":
		return p.Clear
	case "footwork":
		return p.Footwork
	case "strategy":
		return p.Strategy
	default:
		return nil
	}
}

type GPTThreadIDs struct {
	Strategy string `json:"strategy"`
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
			Root:      userFolders.RootFolderID,
			Lift:      userFolders.LiftFolderID,
			Drop:      userFolders.DropFolderID,
			Netplay:   userFolders.NetplayFolderID,
			Clear:     userFolders.ClearFolderID,
			Footwork:  userFolders.FootworkFolderID,
			Strategy:  userFolders.StrategyFolderID,
			Thumbnail: userFolders.ThumbnailFolderID,
		},
		Portfolio: Portfolios{
			Lift:     map[string]Work{},
			Drop:     map[string]Work{},
			Netplay:  map[string]Work{},
			Clear:    map[string]Work{},
			Footwork: map[string]Work{},
			Strategy: map[string]Work{},
		},
		GPTThreadIDs: GPTThreadIDs{
			Strategy: gptThreads.Strategy,
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
		user.GPTThreadIDs.Strategy = threadID
	}
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadIDs(user *UserData, threadIDs *GPTThreadIDs) error {
	user.GPTThreadIDs = *threadIDs
	return client.updateUserData(user)
}
