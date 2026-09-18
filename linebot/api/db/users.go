package db

import (
	"fmt"
	"time"

	"cloud.google.com/go/firestore"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/HeavenAQ/nstc-linebot-2025/api/storage"
	"github.com/HeavenAQ/nstc-linebot-2025/commons"
)

type UserData struct {
	RealName                string      `json:"real_name" firestore:"real_name"`
	ExperimentNumber        string      `json:"experiment_number" firestore:"experiment_number"`
	RegistrationVersion     int         `json:"registration_version" firestore:"registration_version"`
	RegistrationCompletedAt time.Time   `json:"registration_completed_at" firestore:"registration_completed_at"`
	Portfolio               Portfolios  `json:"portfolio" firestore:"portfolio"`
	FolderPaths             FolderPaths `json:"folder_paths" firestore:"folder_paths"`
	Name                    string      `json:"name" firestore:"name"`
	ID                      string      `json:"id" firestore:"id"`
	Handedness              Handedness  `json:"handedness" firestore:"handedness"`
}

type FolderPaths struct {
	Root      string `json:"root" firestore:"root"`
	Serve     string `json:"serve" firestore:"serve"`
	Smash     string `json:"smash" firestore:"smash"`
	Clear     string `json:"clear" firestore:"clear"`
	Lift      string `json:"lift" firestore:"lift"`
	Thumbnail string `json:"thumbnail" firestore:"thumbnail"`
}

type Portfolios struct {
	Serve map[string]Work `json:"serve" firestore:"serve"`
	Smash map[string]Work `json:"smash" firestore:"smash"`
	Clear map[string]Work `json:"clear" firestore:"clear"`
	Lift  map[string]Work `json:"lift" firestore:"lift"`
}

func (p *Portfolios) GetSkillPortfolio(skill string) map[string]Work {
	switch skill {
	case "serve":
		return p.Serve
	case "smash":
		return p.Smash
	case "clear":
		return p.Clear
	case "lift":
		return p.Lift
	default:
		return nil
	}
}

type Work struct {
	AnalysisStatus       string                 `json:"analysis_status" firestore:"analysis_status"`
	AnalysisError        string                 `json:"analysis_error" firestore:"analysis_error"`
	DateTime             string                 `json:"date" firestore:"date"`
	Handedness           string                 `json:"handedness" firestore:"handedness"`
	Thumbnail            string                 `json:"thumbnail" firestore:"thumbnail"`
	Reflection           string                 `json:"reflection" firestore:"reflection"`
	Preview              string                 `json:"preview" firestore:"preview"`
	AINote               string                 `json:"ai_note" firestore:"ai_note"`
	GradingOutcome       commons.GradingOutcome `json:"grading_outcome" firestore:"grading_outcome"`
	AnalysisID           string                 `json:"analysis_id" firestore:"analysis_id"`
	StudentVideo         commons.MediaRef       `json:"student_video" firestore:"student_video"`
	FeedbackVideo        commons.MediaRef       `json:"feedback_video" firestore:"feedback_video"`
	SkeletonOverlayVideo commons.MediaRef       `json:"skeleton_overlay_video" firestore:"skeleton_overlay_video"`
	Expert               commons.ExpertMatch    `json:"expert" firestore:"expert"`
	Timeline             []commons.PhaseMarker  `json:"timeline" firestore:"timeline"`
	CoachingCues         []commons.CoachingCue  `json:"coaching_cues" firestore:"coaching_cues"`
	Diagnostics          map[string]float64     `json:"diagnostics" firestore:"diagnostics"`
}

func (client *FirestoreClient) CreateUserData(userFolders *storage.UserFolders) (*UserData, error) {
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
			Lift:      userFolders.RootPath + "lift/",
			Thumbnail: userFolders.RootPath + "thumbnail",
		},
		Portfolio: Portfolios{
			Serve: map[string]Work{},
			Smash: map[string]Work{},
			Clear: map[string]Work{},
			Lift:  map[string]Work{},
		},
	}

	_, err := ref.Create(*client.Ctx, newUserTemplate)
	if status.Code(err) == codes.AlreadyExists {
		return client.GetUserData(userFolders.UserID)
	}
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

func (client *FirestoreClient) UpdateUserHandedness(user *UserData, handedness Handedness) error {
	user.Handedness = handedness
	_, err := client.Data.Doc(user.ID).Update(*client.Ctx, []firestore.Update{{Path: "handedness", Value: handedness}})
	return err
}

func (client *FirestoreClient) UpdateUserPortfolioReflection(
	user *UserData,
	userPortfolio *map[string]Work,
	skill string,
	date string,
	reflection string,
) error {
	targetWork := (*userPortfolio)[date]
	targetWork.Reflection = reflection
	(*userPortfolio)[date] = targetWork

	_, err := client.Data.Doc(user.ID).Update(*client.Ctx, []firestore.Update{{
		FieldPath: []string{"portfolio", skill, date, "reflection"}, Value: reflection,
	}})
	return err
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
