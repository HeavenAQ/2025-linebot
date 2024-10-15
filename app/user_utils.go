package app

import "github.com/HeavenAQ/api/db"

func (app *App) createUser(userId string) *db.UserData {
	username, err := app.LineBot.GetUserName(userId)
	if err != nil {
		app.Logger.Error.Println("\n\tError getting new user's name:", err)
	}
	userFolders, err := app.DriveClient.CreateUserFolders(userId, username)
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's folders:", err)
	}
	userData, err := app.FirestoreClient.CreateUserData(userFolders)
	if err != nil {
		app.Logger.Error.Println("\n\tError creating new user's data:", err)
	}
	return userData
}

func (app *App) createUserIfNotExist(userId string) (user *db.UserData) {
	user, err := app.FirestoreClient.GetUserData(userId)
	if err != nil {
		app.Logger.Warn.Println("\n\tUser not found, creating new user...")
		userData := app.createUser(userId)
		user = userData
		app.Logger.Info.Println("\n\tNew user created successfully.")
	}
	return
}
