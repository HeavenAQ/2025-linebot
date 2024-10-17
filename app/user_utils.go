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

	// create GPT threads for users
	gptThreadIDs := app.createUserGPTThreads()
	err = app.FirestoreClient.UpdateUserGPTThreadIDs(userData, gptThreadIDs)
	if err != nil {
		app.Logger.Error.Println("\n\tError updating user's GPT threads:", err)
	}
	return userData
}

func (app *App) createUserGPTThreads() *db.GPTThreadIDs {
	serve, err := app.GPTClient.CreateThread()
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's GPT threads:", err)
	}

	badmintonClear, err := app.GPTClient.CreateThread()
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's GPT threads:", err)
	}

	smash, err := app.GPTClient.CreateThread()
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's GPT threads:", err)
	}

	return &db.GPTThreadIDs{
		Serve: serve.ID,
		Clear: badmintonClear.ID,
		Smash: smash.ID,
	}
}

func (app *App) createUserIfNotExist(userID string) *db.UserData {
	user, err := app.FirestoreClient.GetUserData(userID)
	if err != nil {
		app.Logger.Warn.Println("\n\tUser not found, creating new user...")
		userData := app.createUser(userID)
		user = userData

		app.Logger.Info.Println("\n\tNew user created successfully.")
	}

	return user
}

func (app *App) createUserSessionIfNotExist(userID string) *db.UserSession {
	session, err := app.FirestoreClient.GetUserSession(userID)
	if err != nil {
		app.Logger.Warn.Println("\n\tUser session not found, creating new session")
		session, err = app.FirestoreClient.CreateUserSession(userID)
		if err != nil {
			app.Logger.Error.Println("\n\tError creating new user session:", err)
		}
	}

	return session
}

func (app *App) resetUserSession(userID string) {
	err := app.FirestoreClient.ResetSession(
		userID,
		db.UserSession{
			UserState:  db.None,
			ActionStep: db.Empty,
			Skill:      "",
		},
	)
	if err != nil {
		app.Logger.Error.Println("\n\tError resetting user session:", err)
	}
}
