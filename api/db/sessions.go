package db

type UserState int8

const (
	WritingReflection UserState = iota
	WritingPreviewNote
	ChattingWithGPT
	ChattingWithTeacher
	ViewingDashboard
	ViewingExpertVideos
	AnalyzingVideo
	None
)

type UserSession struct {
	Skill     string    `json:"skill"`
	UserState UserState `json:"userState"`
}

func (client *FirestoreClient) GetUserSession(userID string) (*UserSession, error) {
	session, err := client.Sessions.Doc(userID).Get(client.Ctx)
	if err != nil {
		return nil, err
	}
	var userSessioon UserSession
	session.DataTo(&userSessioon)
	return &userSessioon, nil
}

func (client *FirestoreClient) updateUserSession(userID string, newSessionContent UserSession) error {
	_, err := client.Sessions.Doc(userID).Set(client.Ctx, newSessionContent)
	if err != nil {
		return err
	}
	return nil
}

func (client *FirestoreClient) CreateUerSession(userID string) (*UserSession, error) {
	newSession := UserSession{
		UserState: None,
		Skill:     "",
	}
	err := client.updateUserSession(userID, newSession)
	if err != nil {
		return nil, err
	}
	return &newSession, nil
}

func (client *FirestoreClient) UpdateSessionUserState(userID string, state UserState) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}
	userSession.UserState = state
	return client.updateUserSession(userID, *userSession)
}

func (client *FirestoreClient) UpdateSessionUserSkill(userID string, skill string) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}
	userSession.Skill = skill
	return client.updateUserSession(userID, *userSession)
}
