package db

import (
	"fmt"

	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

type UserData struct {
	Portfolio          Portfolios         `json:"portfolio" firestore:"portfolio"`
	FolderPaths        FolderPaths        `json:"folder_paths" firestore:"folder_paths"`
	GPTConversationIDs GPTConversationIDs `json:"gpt_conversation_ids" firestore:"gpt_conversation_ids"`
	Name               string             `json:"name" firestore:"name"`
	ID                 string             `json:"id" firestore:"id"`
	Handedness         Handedness         `json:"handedness" firestore:"handedness"`
}

type FolderPaths struct {
	Root      string `json:"root" firestore:"root"`
	Serve     string `json:"serve" firestore:"serve"`
	Smash     string `json:"smash" firestore:"smash"`
	Clear     string `json:"clear" firestore:"clear"`
	Thumbnail string `json:"thumbnail" firestore:"thumbnail"`
}

type Portfolios struct {
	Serve map[string]Work `json:"serve" firestore:"serve"`
	Smash map[string]Work `json:"smash" firestore:"smash"`
	Clear map[string]Work `json:"clear" firestore:"clear"`
}

type GPTConversationIDs struct {
	Serve string `json:"serve" firestore:"serve"`
	Smash string `json:"smash" firestore:"smash"`
	Clear string `json:"clear" firestore:"clear"`
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
	DateTime                string                 `json:"date" firestore:"date"`
	Thumbnail               string                 `json:"thumbnail" firestore:"thumbnail"`
	SkeletonVideo           string                 `json:"skeleton_video" firestore:"skeleton_video"`
	SkeletonComparisonVideo string                 `json:"skeleton_comparison_video" firestore:"skeleton_comparison_video"`
	Reflection              string                 `json:"reflection" firestore:"reflection"`
	PreviewNote             string                 `json:"preview_note" firestore:"preview_note"`
	AINote                  string                 `json:"ai_note" firestore:"ai_note"`
	GradingOutcome          commons.GradingOutcome `json:"grading_outcome" firestore:"grading_outcome"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders, gptConvs *GPTConversationIDs) (*UserData, error) {
	ref := client.Data.Doc(userFolders.UserID)
	newUserTemplate := &UserData{
		Name:       userFolders.UserName,
		ID:         userFolders.UserID,
		Handedness: Right,
		FolderPaths: FolderPaths{
			Root:      userFolders.RootPath,
			Serve:     userFolders.RootPath + "serve/",
			Smash:     userFolders.RootPath + "smash/",
			Clear:     userFolders.RootPath + "clear/",
			Thumbnail: userFolders.RootPath + "thumbnail",
		},
		Portfolio: Portfolios{
			Serve: map[string]Work{},
			Smash: map[string]Work{},
			Clear: map[string]Work{},
		},
		GPTConversationIDs: GPTConversationIDs{
			Serve: gptConvs.Serve,
			Smash: gptConvs.Smash,
			Clear: gptConvs.Clear,
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
	skeletonFile *storage.UploadedFile,
	comparisonFile *storage.UploadedFile,
	thumbnailFile *storage.UploadedFile,
	aiGrading commons.GradingOutcome,
) error {
	work := Work{
		DateTime:                date,
		GradingOutcome:          aiGrading,
		Reflection:              "尚未填寫心得",
		PreviewNote:             "尚未填寫課前檢視要點",
		AINote:                  "尚未詢問 AI 改善建議",
		SkeletonVideo:           skeletonFile.Path,
		SkeletonComparisonVideo: comparisonFile.Path,
		Thumbnail:               thumbnailFile.Path,
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

func (client *FirestoreClient) UpdateUserGPTConversationID(user *UserData, skill string, id string) error {
	switch skill {
	case "serve":
		user.GPTConversationIDs.Serve = id
	case "smash":
		user.GPTConversationIDs.Smash = id
	case "clear":
		user.GPTConversationIDs.Clear = id
	}
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTConversationIDs(user *UserData, ids *GPTConversationIDs) error {
	user.GPTConversationIDs = *ids
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

func (client *FirestoreClient) ListUsers() (*[]UserData, error) {
	iter := client.Data.Documents(*client.Ctx)
	var all []UserData
	for {
		doc, err := iter.Next()
		if err != nil {
			break
		}

		var u UserData
		if err := doc.DataTo(&u); err != nil {
			return nil, fmt.Errorf("[db.users] decode failed id=%s err=%v", doc.Ref.ID, err)
		}
		all = append(all, u)
	}
	return &all, nil
}
