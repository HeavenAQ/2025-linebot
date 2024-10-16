package db

import (
	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	googleDrive "google.golang.org/api/drive/v3"
)

type UserData struct {
	Portfolio  Portfolio  `json:"portfolio"`
	FolderIDs  FolderIDs  `json:"folderIDs"`
	Name       string     `json:"name"`
	ID         string     `json:"id"`
	Handedness Handedness `json:"handedness"`
}

type FolderIDs struct {
	Root      string `json:"root"`
	Serve     string `json:"serve"`
	Smash     string `json:"smash"`
	Clear     string `json:"clear"`
	Thumbnail string `json:"thumbnail"`
}

type Portfolio struct {
	Serve map[string]Work `json:"serve"`
	Smash map[string]Work `json:"smash"`
	Clear map[string]Work `json:"clear"`
}

func (p *Portfolio) GetSkillPortfolio(skill string) map[string]Work {
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
	DateTime      string  `json:"date"`
	Thumbnail     string  `json:"thumbnail"`
	SkeletonVideo string  `json:"video"`
	Reflection    string  `json:"reflection"`
	PreviewNote   string  `json:"previewNote"`
	AINote        string  `json:"aiNote"`
	Rating        float32 `json:"rating"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders) (*UserData, error) {
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
		Portfolio: Portfolio{
			Serve: map[string]Work{},
			Smash: map[string]Work{},
			Clear: map[string]Work{},
		},
	}

	_, err := ref.Set(*client.Ctx, newUserTemplate)
	if err != nil {
		return nil, err
	}
	return newUserTemplate, nil
}

func (client *FirestoreClient) GetUserData(userId string) (*UserData, error) {
	docsnap, err := client.Data.Doc(userId).Get(*client.Ctx)
	if err != nil {
		return nil, err
	}
	user := &UserData{}
	docsnap.DataTo(user)
	return user, nil
}

func (client *FirestoreClient) updateUserData(user *UserData) error {
	_, err := client.Data.Doc(user.ID).Set(*client.Ctx, *user)
	if err != nil {
		return err
	}
	return nil
}

func (client *FirestoreClient) UpdateUserHandedness(user *UserData, handedness Handedness) error {
	user.Handedness = handedness
	return client.updateUserData(user)
}

func (client *FirestoreClient) CreateUserPortfolioVideo(user *UserData, userPortfolio *map[string]Work, session *UserSession, driveFile *googleDrive.File, thumbnailFile *googleDrive.File, aiRating float32, aiSuggestions string) error {
	id := driveFile.Id
	date := driveFile.Name
	work := Work{
		DateTime:      date,
		Rating:        aiRating,
		Reflection:    "尚未填寫心得",
		PreviewNote:   "尚未填寫課前檢視要點",
		AINote:        aiSuggestions,
		SkeletonVideo: id,
		Thumbnail:     thumbnailFile.Id,
	}
	(*userPortfolio)[date] = work
	client.updateUserSession(user.ID, *session)
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserPortfolioReflection(user *UserData, userPortfolio *map[string]Work, date string, reflection string) error {
	targetWork := (*userPortfolio)[date]
	work := Work{
		DateTime:      targetWork.DateTime,
		Rating:        targetWork.Rating,
		Reflection:    reflection,
		PreviewNote:   targetWork.PreviewNote,
		SkeletonVideo: targetWork.SkeletonVideo,
		Thumbnail:     targetWork.Thumbnail,
		AINote:        targetWork.AINote,
	}
	(*userPortfolio)[date] = work

	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserPortfolioPreviewNote(user *UserData, userPortfolio *map[string]Work, date string, previewNote string) error {
	targetWork := (*userPortfolio)[date]
	work := Work{
		DateTime:      targetWork.DateTime,
		Reflection:    targetWork.Reflection,
		Rating:        targetWork.Rating,
		AINote:        targetWork.AINote,
		PreviewNote:   previewNote,
		SkeletonVideo: targetWork.SkeletonVideo,
		Thumbnail:     targetWork.Thumbnail,
	}
	(*userPortfolio)[date] = work
	return client.updateUserData(user)
}
