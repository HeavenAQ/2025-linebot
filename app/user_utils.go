package app

import "github.com/HeavenAQ/nstc-linebot-2025/api/db"

func (app *App) createUser(userID string) *db.UserData {
	username, err := app.LineBot.GetUserName(userID)
	if err != nil {
		app.Logger.Error.Println("\n\tError getting new user's name:", err)
	}
	userFolders, err := app.DriveClient.CreateUserFolders(userID, username)
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's folders:", err)
	}
	userData, err := app.FirestoreClient.CreateUserData(userFolders)
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's data:", err)
	}
	return userData
}

func (app *App) createUserIfNotExist(userID string) (user *db.UserData) {
	user, err := app.FirestoreClient.GetUserData(userID)
	if err != nil {
		app.Logger.Warn.Println("\n\tUser not found, creating new user...")
		userData := app.createUser(userID)
		user = userData
		app.Logger.Info.Println("\n\tNew user created successfully.")
	}
	return
}

func (app *App) createUserSessionIfNotExist(userID string) (session *db.UserSession) {
	session, err := app.FirestoreClient.GetUserSession(userID)
	if err != nil {
		app.Logger.Warn.Println("\n\tUser session not found, creating new session")
		session, err = app.FirestoreClient.CreateUserSession(userID)
		if err != nil {
			app.Logger.Error.Println("\n\tError creating new user session:", err)
		}
	}
	return
}

func (app *App) resetUserSession(userID string) {
	err := app.FirestoreClient.ResetSession(
		userID,
		db.UserSession{
			UserState: db.None,
			Skill:     "",
		},
	)
	if err != nil {
		app.Logger.Error.Println("\n\tError resetting user session:", err)
	}
}
