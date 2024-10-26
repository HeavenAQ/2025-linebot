package db

import (
	"fmt"
)

type UserSession struct {
	Skill      string     `json:"skill"`
	UserState  UserState  `json:"userState"`
	ActionStep ActionStep `json:"actionStep"`
}

func (client *FirestoreClient) GetUserSession(userID string) (*UserSession, error) {
	session, err := client.Sessions.Doc(userID).Get(*client.Ctx)
	if err != nil {
		return nil, fmt.Errorf("error getting user session: %w", err)
	}

	var userSessioon UserSession
	err = session.DataTo(&userSessioon)
	if err != nil {
		return nil, fmt.Errorf("error converting user session data: %w", err)
	}
	return &userSessioon, nil
}

func (client *FirestoreClient) UpdateUserSession(userID string, newSessionContent UserSession) error {
	_, err := client.Sessions.Doc(userID).Set(*client.Ctx, newSessionContent)
	if err != nil {
		return fmt.Errorf("error updating user session: %w", err)
	}
	return nil
}

func (client *FirestoreClient) CreateUserSession(userID string) (*UserSession, error) {
	newSession := UserSession{
		UserState:  None,
		Skill:      "",
		ActionStep: Empty,
	}
	err := client.UpdateUserSession(userID, newSession)
	if err != nil {
		return nil, err
	}
	return &newSession, nil
}

func (client *FirestoreClient) UpdateSessionUserState(userID string, state UserState, step ActionStep) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}

	userSession.UserState = state
	userSession.ActionStep = step
	return client.UpdateUserSession(userID, *userSession)
}

func (client *FirestoreClient) UpdateSessionUserSkill(userID string, skill string) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}

	userSession.Skill = skill
	return client.UpdateUserSession(userID, *userSession)
}

func (client *FirestoreClient) ResetSession(userID string) error {
	userSession := UserSession{
		Skill:      "",
		UserState:  None,
		ActionStep: Empty,
	}
	err := client.UpdateUserSession(userID, userSession)
	if err != nil {
		return err
	}
	return nil
}

func (client *FirestoreClient) UpdateSessionActionStep(userID string, step ActionStep) error {
	userSession, err := client.GetUserSession(userID)
	if err != nil {
		return err
	}

	userSession.ActionStep = step
	return client.UpdateUserSession(userID, *userSession)
}
