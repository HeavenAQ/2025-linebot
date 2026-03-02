package app

import "github.com/HeavenAQ/nstc-linebot-2025/api/db"

func (app *App) createUser(userID string) *db.UserData {
	// Retrieve user's name from LINE
	app.Logger.Info.Println("Getting the user's name")
	username, err := app.LineBot.GetUserName(userID)
	if err != nil {
		app.Logger.Error.Println("Error getting new user's name:", err)
		return nil
	}
	app.Logger.Info.Println("User name has been retrieved")

	// Create user's folder structure in bucket
	app.Logger.Info.Println("Creating the user's folder structure")
	userFolders, err := app.BucketClient.CreateUserFolders(userID, username)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's folder structure:", err)
		return nil
	}
	app.Logger.Info.Println("User's folder structure has been created")

	// Create a single GPT chat thread for the user.
	app.Logger.Info.Println("Creating the user's GPT chat thread")
	gptThreadIDs, err := app.createUserGPTThread()
	if err != nil {
		app.Logger.Error.Println("Error creating user's GPT chat thread:", err)
		return nil
	}
	app.Logger.Info.Println("User's GPT chat thread has been created")

	// Store user's data in database
	app.Logger.Info.Println("Add the user's data to database")
	userData, err := app.FirestoreClient.CreateUserData(userFolders, gptThreadIDs)
	if err != nil {
		app.Logger.Error.Println("Error creating new user's data:", err)
		return nil
	}
	app.Logger.Info.Println("User's data has been added")
	return userData
}

func (app *App) createUserGPTThread() (*db.GPTThreadIDs, error) {
	app.Logger.Info.Println("Creating GPT thread...")
	thread, err := app.GPTClient.CreateThread()
	if err != nil {
		app.Logger.Error.Println("Error creating GPT thread:", err)
		return nil, err
	}

	return &db.GPTThreadIDs{
		Chat: thread.ID,
	}, nil
}

func (app *App) createUserIfNotExist(userID string) *db.UserData {
	user, err := app.FirestoreClient.GetUserData(userID)
	if err != nil {
		app.Logger.Warn.Println("User not found, creating new user...")
		userData := app.createUser(userID)
		if userData == nil {
			app.Logger.Error.Println("Failed to create new user")
			return nil
		}
		user = userData

		app.Logger.Info.Println("New user created successfully.")
	}

	return user
}

func (app *App) ensureUserGPTThread(user *db.UserData) (string, error) {
	threadID := user.GPTThreadIDs.Chat
	if threadID != "" {
		return threadID, nil
	}

	thread, err := app.GPTClient.CreateThread()
	if err != nil {
		return "", err
	}

	if err := app.FirestoreClient.UpdateUserGPTThreadID(user, thread.ID); err != nil {
		return "", err
	}

	return thread.ID, nil
}

func (app *App) createUserSessionIfNotExist(userID string) *db.UserSession {
	session, err := app.FirestoreClient.GetUserSession(userID)
	if err != nil {
		app.Logger.Warn.Println("User session not found, creating new session")
		session, err = app.FirestoreClient.CreateUserSession(userID)
		if err != nil {
			app.Logger.Error.Println("Error creating new user session:", err)
			return nil
		}
	}

	return session
}

func (app *App) getUserPortfolio(user *db.UserData, skill string) *map[string]db.Work {
	switch skill {
	case "smash":
		return &user.Portfolio.Smash
	case "backhand_drive":
		return &user.Portfolio.BackhandDrive
	case "forehand_drive":
		return &user.Portfolio.ForehandDrive
	case "backhand_netkill":
		return &user.Portfolio.BackhandNetKill
	case "forehand_netkill":
		return &user.Portfolio.ForehandNetKill
	case "frontcourt_footwork":
		return &user.Portfolio.FrontCourtFootwork
	case "backcourt_footwork":
		return &user.Portfolio.BackCourtFootwork
	default:
		return nil
	}
}
