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
	Root                    string `json:"root"`
	JumpingClear            string `json:"jumping_clear"`
	FrontCourtHighPointDrop string `json:"front_court_high_point_drop"`
	DefensiveClear          string `json:"defensive_clear"`
	FrontCourtLowPointLift  string `json:"front_court_low_point_lift"`
	JumpingSmash            string `json:"jumping_smash"`
	MidCourtChasseToBack    string `json:"mid_court_chasse_to_back"`
	ForwardCrossStep        string `json:"forward_cross_step"`
	MidCourtBackCrossStep   string `json:"mid_court_back_cross_step"`
	DefensiveSlideStep      string `json:"defensive_slide_step"`
	Thumbnail               string `json:"thumbnail"`
}

type Portfolios struct {
	JumpingClear            map[string]Work `json:"jumping_clear"`
	FrontCourtHighPointDrop map[string]Work `json:"front_court_high_point_drop"`
	DefensiveClear          map[string]Work `json:"defensive_clear"`
	FrontCourtLowPointLift  map[string]Work `json:"front_court_low_point_lift"`
	JumpingSmash            map[string]Work `json:"jumping_smash"`
	MidCourtChasseToBack    map[string]Work `json:"mid_court_chasse_to_back"`
	ForwardCrossStep        map[string]Work `json:"forward_cross_step"`
	MidCourtBackCrossStep   map[string]Work `json:"mid_court_back_cross_step"`
	DefensiveSlideStep      map[string]Work `json:"defensive_slide_step"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "jumping_clear":
		return p.JumpingClear
	case "front_court_high_point_drop":
		return p.FrontCourtHighPointDrop
	case "defensive_clear":
		return p.DefensiveClear
	case "front_court_low_point_lift":
		return p.FrontCourtLowPointLift
	case "jumping_smash":
		return p.JumpingSmash
	case "mid_court_chasse_to_back":
		return p.MidCourtChasseToBack
	case "forward_cross_step":
		return p.ForwardCrossStep
	case "mid_court_back_cross_step":
		return p.MidCourtBackCrossStep
	case "defensive_slide_step":
		return p.DefensiveSlideStep
	default:
		return nil
	}
}

type GPTThreadIDs struct {
	JumpingClear            string `json:"jumping_clear"`
	FrontCourtHighPointDrop string `json:"front_court_high_point_drop"`
	DefensiveClear          string `json:"defensive_clear"`
	FrontCourtLowPointLift  string `json:"front_court_low_point_lift"`
	JumpingSmash            string `json:"jumping_smash"`
	MidCourtChasseToBack    string `json:"mid_court_chasse_to_back"`
	ForwardCrossStep        string `json:"forward_cross_step"`
	MidCourtBackCrossStep   string `json:"mid_court_back_cross_step"`
	DefensiveSlideStep      string `json:"defensive_slide_step"`
}

type Work struct {
	DateTime   string `json:"date"`
	Thumbnail  string `json:"thumbnail"`
	Video      string `json:"video"`
	Reflection string `json:"reflection"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders, gptThreads *GPTThreadIDs) (*UserData, error) {
	ref := client.Data.Doc(userFolders.UserID)

	// In GCS, folders are just path prefixes
	rootPath := userFolders.RootPath
	newUserTemplate := &UserData{
		Name:       userFolders.UserName,
		ID:         userFolders.UserID,
		Handedness: Right,
		FolderPaths: FolderPaths{
			Root:                    rootPath,
			JumpingClear:            rootPath + "jumping_clear/",
			FrontCourtHighPointDrop: rootPath + "front_court_high_point_drop/",
			DefensiveClear:          rootPath + "defensive_clear/",
			FrontCourtLowPointLift:  rootPath + "front_court_low_point_lift/",
			JumpingSmash:            rootPath + "jumping_smash/",
			MidCourtChasseToBack:    rootPath + "mid_court_chasse_to_back/",
			ForwardCrossStep:        rootPath + "forward_cross_step/",
			MidCourtBackCrossStep:   rootPath + "mid_court_back_cross_step/",
			DefensiveSlideStep:      rootPath + "defensive_slide_step/",
			Thumbnail:               rootPath + "thumbnails/",
		},
		Portfolio: Portfolios{
			JumpingClear:            map[string]Work{},
			FrontCourtHighPointDrop: map[string]Work{},
			DefensiveClear:          map[string]Work{},
			FrontCourtLowPointLift:  map[string]Work{},
			JumpingSmash:            map[string]Work{},
			MidCourtChasseToBack:    map[string]Work{},
			ForwardCrossStep:        map[string]Work{},
			MidCourtBackCrossStep:   map[string]Work{},
			DefensiveSlideStep:      map[string]Work{},
		},
		GPTThreadIDs: GPTThreadIDs{
			JumpingClear:            gptThreads.JumpingClear,
			FrontCourtHighPointDrop: gptThreads.FrontCourtHighPointDrop,
			DefensiveClear:          gptThreads.DefensiveClear,
			FrontCourtLowPointLift:  gptThreads.FrontCourtLowPointLift,
			JumpingSmash:            gptThreads.JumpingSmash,
			MidCourtChasseToBack:    gptThreads.MidCourtChasseToBack,
			ForwardCrossStep:        gptThreads.ForwardCrossStep,
			MidCourtBackCrossStep:   gptThreads.MidCourtBackCrossStep,
			DefensiveSlideStep:      gptThreads.DefensiveSlideStep,
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
	case "jumping_clear":
		user.GPTThreadIDs.JumpingClear = threadID
	case "front_court_high_point_drop":
		user.GPTThreadIDs.FrontCourtHighPointDrop = threadID
	case "defensive_clear":
		user.GPTThreadIDs.DefensiveClear = threadID
	case "front_court_low_point_lift":
		user.GPTThreadIDs.FrontCourtLowPointLift = threadID
	case "jumping_smash":
		user.GPTThreadIDs.JumpingSmash = threadID
	case "mid_court_chasse_to_back":
		user.GPTThreadIDs.MidCourtChasseToBack = threadID
	case "forward_cross_step":
		user.GPTThreadIDs.ForwardCrossStep = threadID
	case "mid_court_back_cross_step":
		user.GPTThreadIDs.MidCourtBackCrossStep = threadID
	case "defensive_slide_step":
		user.GPTThreadIDs.DefensiveSlideStep = threadID
	}
	return client.updateUserData(user)
}

func (client *FirestoreClient) UpdateUserGPTThreadIDs(user *UserData, threadIDs *GPTThreadIDs) error {
	user.GPTThreadIDs = *threadIDs
	return client.updateUserData(user)
}
