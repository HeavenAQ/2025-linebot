package db

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
)

type UserData struct {
	Portfolio    Portfolios   `json:"portfolio"`
	FolderPaths  FolderPaths  `json:"folderPaths"`
	GPTThreadIDs GPTThreadIDs `json:"gptThreadIDs"`
	Name         string       `json:"name"`
	ID           string       `json:"id"`
	Handedness   Handedness   `json:"handedness"`
}

type FolderPaths struct {
	Root               string `json:"root"`
	Smash              string `json:"smash"`
	BackhandDrive      string `json:"backhand_drive"`
	ForehandDrive      string `json:"forehand_drive"`
	BackhandNetKill    string `json:"backhand_netkill"`
	ForehandNetKill    string `json:"forehand_netkill"`
	FrontCourtFootwork string `json:"frontcourt_footwork"`
	BackCourtFootwork  string `json:"backcourt_footwork"`
	Thumbnail          string `json:"thumbnail"`
}

type Portfolios struct {
	Smash              map[string]Work `json:"smash"`
	BackhandDrive      map[string]Work `json:"backhand_drive"`
	ForehandDrive      map[string]Work `json:"forehand_drive"`
	BackhandNetKill    map[string]Work `json:"backhand_netkill"`
	ForehandNetKill    map[string]Work `json:"forehand_netkill"`
	FrontCourtFootwork map[string]Work `json:"frontcourt_footwork"`
	BackCourtFootwork  map[string]Work `json:"backcourt_footwork"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "smash":
		return p.Smash
	case "backhand_drive":
		return p.BackhandDrive
	case "forehand_drive":
		return p.ForehandDrive
	case "backhand_netkill":
		return p.BackhandNetKill
	case "forehand_netkill":
		return p.ForehandNetKill
	case "frontcourt_footwork":
		return p.FrontCourtFootwork
	case "backcourt_footwork":
		return p.BackCourtFootwork
	default:
		return nil
	}
}

type GPTThreadIDs struct {
	Chat string `json:"chat"`
}

type Work struct {
	DateTime   string `json:"date"`
	Thumbnail  string `json:"thumbnail"`
	Video      string `json:"video"`
	Reflection string `json:"reflection"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders, gptThreads *GPTThreadIDs) (*UserData, error) {
	ref := client.Data.Doc(userFolders.UserID)

	if gptThreads == nil {
		gptThreads = &GPTThreadIDs{}
	}

	// In GCS, folders are just path prefixes
	rootPath := userFolders.RootPath
	newUserTemplate := &UserData{
		Name:       userFolders.UserName,
		ID:         userFolders.UserID,
		Handedness: Right,
		FolderPaths: FolderPaths{
			Root:               rootPath,
			Smash:              rootPath + "smash/",
			BackhandDrive:      rootPath + "backhand_drive/",
			ForehandDrive:      rootPath + "forehand_drive/",
			BackhandNetKill:    rootPath + "backhand_netkill/",
			ForehandNetKill:    rootPath + "forehand_netkill/",
			FrontCourtFootwork: rootPath + "frontcourt_footwork/",
			BackCourtFootwork:  rootPath + "backcourt_footwork/",
			Thumbnail:          rootPath + "thumbnails/",
		},
		Portfolio: Portfolios{
			Smash:              map[string]Work{},
			BackhandDrive:      map[string]Work{},
			ForehandDrive:      map[string]Work{},
			BackhandNetKill:    map[string]Work{},
			ForehandNetKill:    map[string]Work{},
			FrontCourtFootwork: map[string]Work{},
			BackCourtFootwork:  map[string]Work{},
		},
		GPTThreadIDs: GPTThreadIDs{
			Chat: gptThreads.Chat,
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

func (client *FirestoreClient) CreateUserPortfolioVideo(user *UserData, userPortfolio *map[string]Work, date string, session *UserSession, videoFile *storage.UploadedFile, thumbnailFile *storage.UploadedFile) error {
	if userPortfolio == nil {
		return fmt.Errorf("error creating user portfolio video: missing portfolio for skill %q", session.Skill)
	}
	if *userPortfolio == nil {
		*userPortfolio = make(map[string]Work)
	}

	work := Work{
		DateTime:   date,
		Reflection: "尚未填寫心得",
		Video:      videoFile.Path,
		Thumbnail:  thumbnailFile.Path,
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
	if userPortfolio == nil || *userPortfolio == nil {
		return fmt.Errorf("error updating user portfolio reflection: portfolio does not exist for date %q", date)
	}

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

func (client *FirestoreClient) UpdateUserGPTThreadID(user *UserData, threadID string) error {
	user.GPTThreadIDs.Chat = threadID
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadIDs(user *UserData, threadIDs *GPTThreadIDs) error {
	user.GPTThreadIDs = *threadIDs
	return client.updateUserData(user)
}
