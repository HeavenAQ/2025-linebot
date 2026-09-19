package app

import (
	"fmt"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/HeavenAQ/nstc-linebot-2025/api/db"
)

// EnsureUserData returns the learner's record, creating it on first use. The
// LIFF registration form reaches a learner who has never sent the bot a
// message, so the folders and the document may not exist yet.
func (app *App) EnsureUserData(userID string) (*db.UserData, error) {
	user, err := app.FirestoreClient.GetUserData(userID)
	if err == nil {
		return user, nil
	}
	if status.Code(err) != codes.NotFound {
		return nil, err
	}
	if created := app.createUser(userID); created != nil {
		return created, nil
	}
	return nil, fmt.Errorf("could not create user %s", userID)
}

func (app *App) createUser(userID string) *db.UserData {
	// Retrieve user's name from LINE
	app.Logger.Info.Println("Getting the user's name")
	username, err := app.LineBot.GetUserName(userID)
	if err != nil {
		app.Logger.Error.Println("Error getting new user's name:", err)
		return nil
	}
	app.Logger.Info.Println("User name has been retrieved")

	// Create user's folders
	app.Logger.Info.Println("Creating the user's folders")
	userFolders, err := app.StorageClient.CreateUserFolders(userID, username)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's folders:", err)
		return nil
	}
	app.Logger.Info.Println("User's folders has been created")

	// Store user's data in database
	app.Logger.Info.Println("Add the user's data to database")
	userData, err := app.FirestoreClient.CreateUserData(userFolders)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's data:", err)
	}
	app.Logger.Info.Println("User's data has been added")
	return userData
}

func (app *App) createUserSessionIfNotExist(userID string) *db.UserSession {
	session, err := app.FirestoreClient.GetUserSession(userID)
	if err != nil {
		app.Logger.Warn.Println("User session not found, creating new session")
		session, err = app.FirestoreClient.CreateUserSession(userID)
		if err != nil {
			app.Logger.Error.Println("Error creating new user session:", err)
		}
	}

	return session
}
