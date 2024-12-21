package db

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
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
	Serve     string `json:"serve"`
	Smash     string `json:"smash"`
	Clear     string `json:"clear"`
	Thumbnail string `json:"thumbnail"`
}

type Portfolios struct {
	Serve map[string]Work `json:"serve"`
	Smash map[string]Work `json:"smash"`
	Clear map[string]Work `json:"clear"`
}

type GPTThreadIDs struct {
	Serve string `json:"serve"`
	Smash string `json:"smash"`
	Clear string `json:"clear"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "serve":
		return p.Serve
	case "smash":
		return p.Smash
	case "clear":
		return p.Clear
	default:
		return nil
	}
}

type Work struct {
	DateTime                string                 `json:"date"`
	Thumbnail               string                 `json:"thumbnail"`
	SkeletonVideo           string                 `json:"video"`
	SkeletonComparisonVideo string                 `json:"comparisonVideo"`
	Reflection              string                 `json:"reflection"`
	PreviewNote             string                 `json:"previewNote"`
	AINote                  string                 `json:"aiNote"`
	GradingOutcome          commons.GradingOutcome `json:"gradingOutcome"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders, gptThreads *GPTThreadIDs) (*UserData, error) {
	ref := client.Data.Doc(userFolders.UserID)
	newUserTemplate := &UserData{
		Name:       userFolders.UserName,
		ID:         userFolders.UserID,
		Handedness: Right,
		FolderIDs: FolderIDs{
			Root:      userFolders.RootFolderID,
			Serve:     userFolders.ServeFolderID,
			Smash:     userFolders.SmashFolderID,
			Clear:     userFolders.ClearFolderID,
			Thumbnail: userFolders.ThumbnailFolderID,
		},
		Portfolio: Portfolios{
			Serve: map[string]Work{},
			Smash: map[string]Work{},
			Clear: map[string]Work{},
		},
		GPTThreadIDs: GPTThreadIDs{
			Serve: gptThreads.Serve,
			Smash: gptThreads.Smash,
			Clear: gptThreads.Clear,
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

func (client *FirestoreClient) CreateUserPortfolioVideo(
	user *UserData,
	userPortfolio *map[string]Work,
	date string,
	session *UserSession,
	driveFile *googleDrive.File,
	thumbnailFile *googleDrive.File,
	aiGrading commons.GradingOutcome,
) error {
	id := driveFile.Id
	work := Work{
		DateTime:                date,
		GradingOutcome:          aiGrading,
		Reflection:              "尚未填寫心得",
		PreviewNote:             "尚未填寫課前檢視要點",
		AINote:                  "尚未詢問 AI 改善建議",
		SkeletonVideo:           id,
		SkeletonComparisonVideo: "",
		Thumbnail:               thumbnailFile.Id,
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
		DateTime:                targetWork.DateTime,
		GradingOutcome:          targetWork.GradingOutcome,
		Reflection:              reflection,
		PreviewNote:             targetWork.PreviewNote,
		SkeletonVideo:           targetWork.SkeletonVideo,
		SkeletonComparisonVideo: targetWork.SkeletonComparisonVideo,
		Thumbnail:               targetWork.Thumbnail,
		AINote:                  targetWork.AINote,
	}
	(*userPortfolio)[date] = work

	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserPortfolioPreviewNote(
	user *UserData,
	userPortfolio *map[string]Work,
	date string,
	previewNote string,
) error {
	targetWork := (*userPortfolio)[date]
	work := Work{
		DateTime:                targetWork.DateTime,
		GradingOutcome:          targetWork.GradingOutcome,
		Reflection:              targetWork.Reflection,
		PreviewNote:             previewNote,
		SkeletonVideo:           targetWork.SkeletonVideo,
		SkeletonComparisonVideo: targetWork.SkeletonComparisonVideo,
		Thumbnail:               targetWork.Thumbnail,
		AINote:                  targetWork.AINote,
	}
	(*userPortfolio)[date] = work
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadID(user *UserData, skill string, threadID string) error {
	switch skill {
	case "serve":
		user.GPTThreadIDs.Serve = threadID
	case "smash":
		user.GPTThreadIDs.Smash = threadID
	case "clear":
		user.GPTThreadIDs.Clear = threadID
	}
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadIDs(user *UserData, threadIDs *GPTThreadIDs) error {
	user.GPTThreadIDs = *threadIDs
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserPortfolioAINote(
	user *UserData,
	userPortfolio *map[string]Work,
	date string,
	aiNote string,
) error {
	targetWork := (*userPortfolio)[date]
	work := Work{
		DateTime:                targetWork.DateTime,
		GradingOutcome:          targetWork.GradingOutcome,
		Reflection:              targetWork.Reflection,
		PreviewNote:             targetWork.PreviewNote,
		SkeletonVideo:           targetWork.SkeletonVideo,
		SkeletonComparisonVideo: targetWork.SkeletonComparisonVideo,
		Thumbnail:               targetWork.Thumbnail,
		AINote:                  aiNote,
	}
	(*userPortfolio)[date] = work
	return client.updateUserData(user)
}
